import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import numpy as np
from time import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class fwfm(nn.Module):
    def __init__(self, field_num, feature_sizes, embedding_dim, n_epochs, batch_size, learning_rate, weight_decay,
                 numerical, use_cuda):
        super(fwfm, self).__init__()

