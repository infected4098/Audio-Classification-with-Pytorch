import torch
from torch import nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


#(batch_size,  1, var_dim, sequence_len)
#mel : (batch_size ,1, 44, 128)
#mfcc: (batch_size, 1, 13, 49)


class NewCNN_res(nn.Module):
    def __init__(self, dim_h, input_dim, kernel_size = (3, 3)):
        super().__init__()
        self.dim_h = dim_h
        self.kernel_size = kernel_size
        self.act = nn.ReLU()
        self.var_dim = input_dim[0]
        self.sequence_len = input_dim[1]
        variable_dim = 1
        self.stacks = nn.Sequential()
        self.resid = nn.Conv2d(variable_dim, 64, kernel_size = 1, stride = (1, 1), padding = "same")
        self.resid_bn64_1 = nn.BatchNorm2d(64)
        self.resid_bn64_2 = nn.BatchNorm2d(64)
        self.resid_bn64_3 = nn.BatchNorm2d(64)
        self.flat = nn.Flatten()
        self.input_conv = nn.Conv2d(variable_dim, self.dim_h, kernel_size = self.kernel_size, stride = (1, 1), padding = "same")
        self.bn_32 = nn.BatchNorm2d(self.dim_h)
        self.maxpool = nn.MaxPool2d((2, 3))
        self.next_conv = nn.Conv2d(self.dim_h, 64, kernel_size=self.kernel_size, stride=(1, 1),
                                    padding="same")
        self.bn_64 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d((2, 3))
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.in_feat = 64*self.var_dim*self.sequence_len
        self.linear_1 = nn.Linear(in_features=64, out_features=32)
        self.act2 = nn.LeakyReLU()
        #nn.LazyLinear(128))
        #self.linear_1 = nn.Linear(self.in_feat, 128)
        self.linear_2 = nn.Linear(32, 14)
        self.linear_3 = nn.Linear(in_features=14, out_features=14)
        # self.linear_3 = nn.Linear(64, 14)

        self.resid_2 = nn.Conv2d(64, 64, kernel_size = 1, stride = (1, 1), padding = "same")
        self.resid_bn64 = nn.BatchNorm2d(64)

        self.test = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size = (3,3), padding=(1,1), stride=(1,1)),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1),
                                            stride=(2, 2)),
                                  nn.LeakyReLU(0.2),

                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1),
                                            stride=(1, 1)),
                                  nn.LeakyReLU(0.2),

                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1),
                                            stride=(2, 2)),
                                  nn.LeakyReLU(0.2),

                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1),
                                            stride=(2, 2)),
                                  nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.test(x)
        x = self.flatten(x).squeeze(2).squeeze(2)
        x = self.linear_1(x)
        x = self.act2(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        # identity = x
        # identity = self.resid(identity)
        # #identity = self.resid_bn64_1(identity)
        # x = self.input_conv(x)
        # x = self.resid_bn64_2(x)
        # x += identity
        #
        # #x = self.maxpool(x)
        # x = self.act(x)
        #
        # identity_2 = x
        # identity_2 = self.resid_2(identity_2)
        # #identity_2 = self.resid_bn64_3(identity_2)
        # x = self.next_conv(x)
        # x += identity_2
        # x = self.bn_64(x)
        # #x = self.maxpool(x)
        # x = self.act(x)
        # x = self.flatten(x).squeeze(2).squeeze(2)
        # lin = self.linear_1(x)
        # lin = self.act(lin)
        # lin_ = self.linear_2(lin)
        # # lin_ = self.act(lin_)
        # # logits = self.linear_3(lin_)

        return x
