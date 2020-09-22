
import sys
import random
import argparse
import pickle
import numpy as np
from model import fm,TFnet
from utils import data_preprocess
import torch

parser = argparse.ArgumentParser(description="Hyperparameter tuning")
parser.add_argument('-gpu', default=1, type=int, help='gpu id')
parser.add_argument('-use_cuda', default=0, type=int, help='use cuda or not')
parser.add_argument('-embedding_dim', default=10, type=int, help='Embedding dimension')
parser.add_argument('-n_epochs', default=10, type=int, help='number of epochs')
parser.add_argument('-batch_size', default=2048, type=int, help='batch size')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-l2', default=3e-7, type=float, help='L2 penalty')
parser.add_argument('-is_fw', default=1, type=int, help='is use field weight model')
parser.add_argument('-is_deep', default=1, type=int, help='use deep layer or not')

pars = parser.parse_args()

criteo_num_feat_dim = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
# result_dict = data_preprocess.read_data('data/train.csv', 'data/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
# with open('data/result_dict.pickle', 'wb') as f:
#     pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
# test_dict = data_preprocess.read_data('data/valid.csv', 'data/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
# with open('data/test_dict.pickle', 'wb') as f:
#     pickle.dump(test_dict, f, pickle.HIGHEST_PROTOCOL)

result_dict = {}
test_dict = {}

with open('data/result_dict.pickle', 'rb') as f:
    result_dict = pickle.load(f)

with open('data/test_dict.pickle', 'rb') as f:
    test_dict = pickle.load(f)
# with torch.cuda.device(pars.gpu):
#     model = fm.fm(field_num=39, feature_sizes=result_dict['feature_sizes'], embedding_dim=pars.embedding_dim,
#                   n_epochs=pars.n_epochs, batch_size=pars.batch_size, learning_rate=pars.lr, weight_decay=pars.l2,
#                   numerical=13, use_cuda=pars.use_cuda)
#     if pars.use_cuda:
#         model = model.cuda()
#         model.fit(result_dict['index'], result_dict['value'], result_dict['label'], test_dict['index'],
#                   test_dict['value'], test_dict['value'])
if __name__ =='__main__':
        model = TFnet.TFnet(field_num=39, feature_sizes=result_dict['feature_sizes'], embedding_dim=pars.embedding_dim,
                      n_epochs=pars.n_epochs, batch_size=pars.batch_size, learning_rate=pars.lr, weight_decay=pars.l2,
                      numerical=13, use_cuda=pars.use_cuda, m=4, deep_node=512, deep_layer=3)
        # model = fm.fm(field_num=39, feature_sizes=result_dict['feature_sizes'], embedding_dim=pars.embedding_dim,
        #               n_epochs=pars.n_epochs, batch_size=pars.batch_size, learning_rate=pars.lr, weight_decay=pars.l2,
        #               numerical=13, use_cuda=pars.use_cuda, is_fwfm=pars.is_fw, is_deep=pars.is_deep)
        model.fit(result_dict['index'], result_dict['value'], result_dict['label'], test_dict['index'],
                       test_dict['value'], test_dict['label'])