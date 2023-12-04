import torch
from torch import nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


#(batch_size,  1, var_dim, sequence_len)
#mel : (batch_size ,1, 128, sequence_len)
#mfcc: (batch_size, 1, 13, sequence_len)
class DeepCNN(nn.Module):
    def __init__(self, dim_h_lst, input_dim, kernel_size = (3, 3)):
        super().__init__()
        self.dim_h_lst = dim_h_lst
        self.kernel_size = kernel_size
        self.act = nn.ReLU()
        self.var_dim = input_dim[0]
        self.sequence_len = input_dim[1]
        variable_dim = 1
        self.stacks = nn.Sequential()
        self.flat = nn.Flatten()
        for i, dim_h in enumerate(self.dim_h_lst):
          self.stacks.add_module("CNN layer_" + str(i), nn.Conv2d(variable_dim, self.dim_h_lst[i], kernel_size = self.kernel_size, stride = (1, 1), padding = "same"))
          self.stacks.add_module("BN layer_"+str(i), nn.BatchNorm2d(self.dim_h_lst[i]))
          self.stacks.add_module("RELU layer_" + str(i), self.act)
          variable_dim = dim_h_lst[i]
        self.in_feat = self.dim_h_lst[-1]*self.var_dim*self.sequence_len
        self.linear_1 = nn.Linear(self.in_feat, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 14)

    def forward(self, x):
        comp = self.stacks(x) #(-1, 64, H, W)
        flat = self.flat(comp) #(-1, #resulting dimension product)
        input_dim = flat.shape[1]
        lin = self.linear_1(flat)
        lin = self.act(lin)
        lin_ = self.linear_2(lin)
        lin_ = self.act(lin_)
        logits = self.linear_3(lin_)

        return logits
