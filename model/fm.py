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


class fm(nn.Module):
    def __init__(self, field_num, feature_sizes, embedding_dim, n_epochs, batch_size, learning_rate, weight_decay,
                 numerical, use_cuda):
        super(fm, self).__init__()
        self.field_num = field_num
        self.feature_sizes = feature_sizes
        self.embedding_dim = embedding_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.numerical = numerical
        self.use_cuda = use_cuda

        self.bias = nn.Parameter(torch.Tensor([0.01]))

        self.fm_1st_embedding = nn.ModuleList([
            nn.Embedding(feature_size, 1) for feature_size in feature_sizes
        ])
        self.fm_2nd_embedding = nn.ModuleList([
            nn.Embedding(feature_size , embedding_dim) for feature_size in feature_sizes
        ])

    def forward(self, xi, xv):
        """

        :param xi: embedding index of categories feature, dim: batch x 26
        :param xv: real value of continuous feature, dim:  batch x 13
        :return: predict value of click through rate
        """

        # for all continuous feature, embedding index always is 0
        Tzero = torch.zeros(xi.shape[0], 1, dtype=torch.long)
        if self.use_cuda:
            Tzero = Tzero.cuda()

        fm_1st_emb_arr = [(torch.sum(emb(Tzero), 1).t() * xv[:, i]).t() if i < self.numerical else
                          emb(xi[:, i - self.numerical]) for i, emb in enumerate(self.fm_1st_embedding)]
        fm_1st_order = torch.cat(fm_1st_emb_arr, 1) # dim: B x 39

        fm_2nd_emb_arr = [(torch.sum(emb(Tzero), 1).t() * xv[:, i]).t() if i < self.numerical else
                          emb(xi[:, i - self.numerical]) for i, emb in enumerate(self.fm_2nd_embedding)]

        fm_2nd_order_tensor = torch.stack(fm_2nd_emb_arr, dim=1) # dim: B x 39 x D

        outer_fm = torch.einsum('bfd,bld->bfld',fm_2nd_order_tensor, fm_2nd_order_tensor)
        fm_2nd_order = 0.5 * (torch.sum(torch.sum(outer_fm, 1), 1) - torch.sum(torch.einsum('bijk->bik', outer_fm), 1))
        total_sum = torch.sum(fm_1st_order, 1) + torch.sum(fm_2nd_order, 1) + self.bias
        return total_sum

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
        criterion = F.binary_cross_entropy_with_logits

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
        print('Number of 1st order embeddings: %d' % num_1st_order_embeddings)
        print('Number of 2nd order embeddings: %d' % num_2nd_order_embeddings)
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
        criterion = F.binary_cross_entropy_with_logits
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
        pred = torch.sigmoid(model(xi, xv)).cpu()
        return pred.data.numpy()
