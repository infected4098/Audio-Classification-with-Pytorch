#이거 그대로 쓰면 됨

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
#X.shape = [#batch_size, #timesteps, dim_h]
#y.shape = [#batch_size, #labels] ->reshape
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class DeepLSTM(nn.Module):
    def __init__(self, var_dim, dim_h, n_layer, batch_size):
        super().__init__()
        self.n_layer = n_layer
        self.dim_h = dim_h
        self.batch_size = batch_size
        self.act = nn.ReLU()
        self.variable_dim = var_dim #mfcc에서는 feature size가 13.
        self.flat = nn.Flatten()
        self.lstm = nn.LSTM(self.variable_dim, self.dim_h, num_layers = self.n_layer, batch_first = True) #(batch_size, #timesteps, #features)
        self.bn = nn.BatchNorm2d(self.dim_h) #(#timesteps, #batchsize, dim_h)
        self.in_feat = self.dim_h*self.n_layer
        self.linear_1 = nn.Linear(self.in_feat, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 14)

    def forward(self, x):
        comp, (h_n, c_n) = self.lstm(x)
        h_n = h_n.permute(1, 0, 2) #(batch_size, #n_layers, #dim_h)
        #h_n = torch.cat((h_n[:, -1, :], h_n[:, -2, :]), dim = 0)
        #h_n = h_n[:, -1, :]
        flat = self.flat(h_n) #(-1, #dim_h * #n_layer)
        lin = self.linear_1(flat) #(-1, )
        lin = self.act(lin)
        lin_ = self.linear_2(lin)
        lin_ = self.act(lin_)
        logits = self.linear_3(lin_)

        return logits
#deeplstm = DeepLSTM(80, 64, 4, 16).to(device) #맨 앞 13은 mfcc의 coef 개수.
#deeplstm(torch.zeros([16, 41, 80]))
#print(deeplstm(torch.zeros([16, 49, 13]).to(device)))