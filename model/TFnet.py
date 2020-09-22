import torch
from torch import nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import os
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class TFnet(nn.Module):
    def __init__(self, field_num, feature_sizes, embedding_dim, n_epochs, batch_size, learning_rate, weight_decay,
                 numerical, use_cuda, m, deep_node, deep_layer):
        super(TFnet, self).__init__()
        self.deep_layer = deep_layer
        self.deep_node = deep_node
        self.m = m
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.numerical = numerical
        self.use_cuda = use_cuda
        self.embedding_dim = embedding_dim
        self.feature_sizes = feature_sizes
        self.field_num = field_num
        self.interaction = field_num * (field_num - 1) // 2
        self.left = []
        self.right = []
        for i in range(field_num):
            for j in range(i+1, field_num):
                self.left.append(i)
                self.right.append(j)

        self.fm_1st_embedding = nn.ModuleList([
            nn.Embedding(feature_size, 1) for feature_size in feature_sizes
        ])
        self.fm_2nd_embedding = nn.ModuleList([
            nn.Embedding(feature_size, embedding_dim) for feature_size in feature_sizes
        ])

        self.T3 = nn.Parameter(torch.randn(embedding_dim, m, embedding_dim))
        self.T2 = nn.Parameter(torch.randn(m, embedding_dim * embedding_dim))
        self.gc = nn.Parameter(torch.randn(self.interaction))

        '''
           deep part for higher order feature interaction 
        '''
        setattr(self, 'init_fi_dropout', nn.Dropout(0.5))
        setattr(self, 'init_fi_linear', nn.Linear(self.interaction * self.m, deep_node))
        setattr(self, 'init_fi_batchNorm', nn.BatchNorm1d(deep_node, momentum=0.05))
        # setattr(self, 'init_fi_dropout2', nn.Dropout(0.5))
        # for i in range(1, self.deep_layer+1):
        #     setattr(self, str(i) + '_fi_dropout', nn.Dropout(0.5))
        #     setattr(self, str(i) + '_fi_linear', nn.Linear(deep_node, deep_node))
        #     setattr(self, str(i) + '_fi_batchNorm', nn.BatchNorm1d(deep_node, momentum=0.05))

        '''
           deep part for embedding concat 
        '''
        setattr(self, 'init_emb_dropout', nn.Dropout(0.5))
        setattr(self, 'init_emb_linear', nn.Linear(self.embedding_dim * self.field_num, deep_node))
        setattr(self, 'init_emb_batchNorm', nn.BatchNorm1d(deep_node, momentum=0.05))

        '''
           final concat linear layer
        '''
        self.fc = nn.Linear(deep_node * 2, 1)

    def forward(self, xi, xv):
        Tzero = torch.zeros(xi.shape[0], 1, dtype=torch.long)
        if self.use_cuda:
            Tzero = Tzero.cuda()

        fm_2nd_emb_arr = [(torch.sum(emb(Tzero), 1).t() * xv[:, i]).t() if i < self.numerical else
                          emb(xi[:, i - self.numerical]) for i, emb in enumerate(self.fm_2nd_embedding)]

        fm_2nd_order_tensor = torch.stack(fm_2nd_emb_arr, dim=1)  # dim: B x 39 x D
        vi = [fm_2nd_emb_arr[i] for i in self.left]
        vj = [fm_2nd_emb_arr[i] for i in self.right]
        vi = torch.stack(vi, dim=1)  # bqd
        vj = torch.stack(vj, dim=1)  # bqd
        # ga 有问题， ga 是一个m维的向量，应该是bm 然后softmax
        ga = torch.einsum('bqd,dmd,bqd->bm', vi, self.T3, vj)
        ga = F.softmax(ga, dim=1)  # bm
        # T1 有问题， T1 应该是bdmd 每一个row 对应一个T1，总共有batch个
        T1 = torch.einsum('bm,mn->bmn', ga, self.T2)
        # T1 = T1.reshape(-1, self.m, self.embedding_dim, self.embedding_dim)
        s = torch.einsum('bqd,bmn,bqd->bqm', vi, T1, vj)
        sh = torch.einsum('bqm,q->bqm', s, self.gc)
        sh = sh.reshape(-1, self.interaction * self.m)
        th = getattr(self, 'init_fi_dropout')(sh)
        th = getattr(self, 'init_fi_linear')(th)
        th = getattr(self, 'init_fi_batchNorm')(th)
        th = torch.relu(th)

        emb = torch.cat(fm_2nd_emb_arr, 1)
        emb = getattr(self, 'init_emb_dropout')(emb)
        emb = getattr(self, 'init_emb_linear')(emb)
        emb = getattr(self, 'init_emb_batchNorm')(emb)
        emb = torch.relu(emb)

        outputs = torch.cat((th, emb), dim=1)
        outputs = self.fc(outputs)
        outputs = torch.sigmoid(outputs)
        return torch.sum(outputs, 1)

    def fit(self, xi_train, xv_train, y_train, xi_valid=None, xv_valid=None, y_valid=None):
        is_valid = False
        xi_train = np.array(xi_train)
        xv_train = np.array(xv_train)
        y_train = np.array(y_train)
        x_size = xi_train.shape[0]
        if xi_valid:
            xi_valid = np.array(xi_valid)
            xv_valid = np.array(xv_valid)
            y_valid = np.array(y_valid)
            x_valid_size = xi_valid.shape[0]
            is_valid = True

        print("init_weight")
        # init_weight()

        model = self.train()

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = F.binary_cross_entropy

        train_result = []
        valid_result = []
        num_total = 0
        num_1st_order_embeddings = 0
        num_2nd_order_embeddings = 0
        print("=========================")
        for name, param in model.named_parameters():
            print(name, param.data.shape)
            num_total += np.prod(param.data.shape)
            if '1st' in name:
                num_1st_order_embeddings += np.prod(param.data.shape)
            if '2nd' in name:
                num_2nd_order_embeddings += np.prod(param.data.shape)
        print('Summation of feature sizes: %s' % (sum(self.feature_sizes)))
        print("Number of total parameters: %d" % num_total)

        for epoch in range(self.n_epochs):
            total_loss = .0
            batch_iter = x_size // self.batch_size
            print("batch_iter:", batch_iter)
            epoch_begin_time = time()
            batch_begin_time = time()
            for i in range(batch_iter + 1):
                offset = i * self.batch_size
                end = min(x_size, offset + self.batch_size)
                if offset == end:
                    break
                batch_xi = Variable(torch.LongTensor(xi_train[offset:end]))
                batch_xv = Variable(torch.FloatTensor(xv_train[offset:end]))
                batch_y = Variable(torch.FloatTensor(y_train[offset:end]))
                if self.use_cuda:
                    batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(). batch_y.cuda()
                optimizer.zero_grad()
                outputs = model(batch_xi, batch_xv)
                # print("outputs:", outputs.shape)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.data.item()
                if i % 100 == 99:
                    eval = self.evaluate(batch_xi, batch_xv, batch_y)
                    print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                          (epoch + 1, i + 1, total_loss / 100.0, eval, time() - batch_begin_time))
                    total_loss = 0.0
                    batch_begin_time = time()
            train_loss, train_eval = self.eval_by_batch(xi_train, xv_train, y_train, x_size)
            train_result.append(train_eval)
            print('Training [%d] loss: %.6f metric: %.6f  time: %.1f s' %
                  (
                      epoch + 1, train_loss, train_eval, time() - epoch_begin_time))
            if is_valid:
                valid_loss, valid_eval = self.eval_by_batch(xi_valid, xv_valid, y_valid, x_valid_size)
                valid_result.append(valid_eval)
                print('Validation [%d] loss: %.6f metric: %.6f time: %.1f s' %
                      (epoch + 1, valid_loss, valid_eval,
                       time() - epoch_begin_time))
            print('*' * 50)

    def eval_by_batch(self, xi, xv, y, x_size):
        total_loss = 0.0
        y_pred = []
        batch_size = 2048
        batch_iter = x_size // batch_size
        criterion = F.binary_cross_entropy
        model = self.eval()
        with torch.no_grad():
            for i in range(batch_iter + 1):
                offset = i * batch_size
                end = min(x_size, offset + batch_size)
                if offset == end:
                    break
                batch_xi = Variable(torch.LongTensor(xi[offset:end]))
                batch_xv = Variable(torch.FloatTensor(xv[offset:end]))
                batch_y = Variable(torch.FloatTensor(y[offset:end]))
                if self.use_cuda:
                    batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
                outputs = model(batch_xi, batch_xv)
                pred = torch.sigmoid(outputs).cpu()
                y_pred.extend(pred.data.numpy())
                loss = criterion(outputs, batch_y)
                total_loss += loss.data.item() * (end - offset)
        total_metric = roc_auc_score(y, y_pred)
        return total_loss / x_size, total_metric

    def evaluate(self, xi, xv, y):
        y_pred = self.inner_predict_proba(xi, xv)
        return roc_auc_score(y.cpu().data.numpy(), y_pred)

    def inner_predict_proba(self, xi, xv):
        model = self.eval()
        pred = model(xi, xv).cpu()
        return pred.data.numpy()


