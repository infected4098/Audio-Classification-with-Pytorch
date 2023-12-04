import torch
from torch import nn
import numpy as np
import scipy.io.wavfile as wav
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os
m_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/audio_train" #디렉토리 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#오디오 파일의 정보가 들어 있는 데이터프레임 불러오기
label_df = pd.read_csv("C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/real_df.csv")

# Hyperparameters
CONFIG = {
    'lr': 0.001,
    'epochs': 30,
    'min_batch': 16,
    'weight_decay': 1e-4
}

X_mel_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/Xmel_torch.npy"
y_mel_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/ymel_torch.npy"


class audio_dataset(Dataset):
    def __init__(self, X_data, y_data):

        self.X_data = X_data #(#data, #coefs, #timetsteps)
        self.y_data = y_data
        self.X_torch = torch.FloatTensor(self.X_data).unsqueeze(dim = 1).to(device) #(#data, 1, #coefs, #timetsteps)
        self.y_torch = torch.LongTensor(self.y_data).to(device) #(#data, #labels)
        self.x_len = self.X_torch.shape[0]
        self.y_len = self.y_torch.shape[0]
        assert self.x_len == self.y_len

    def __getitem__(self, index):
        #index = np.random.randint(0, self.x_len) #데이터셋 생성 과정에서 randomness 추가
        return self.X_torch[index], self.y_torch[index]
    def __len__(self):
        return self.x_len


class lstm_dataset(Dataset):
    def __init__(self, X_data, y_data):

        self.X_data = X_data #(#data, #dim, #timesteps)
        self.y_data = y_data
        self.X_torch = torch.FloatTensor(self.X_data).to(device) #(#data, #dim, #timesteps)
        self.X_torch = self.X_torch.permute(0, 2, 1)
        self.y_torch = torch.LongTensor(self.y_data).to(device) #(#data, #labels)


        self.x_len = self.X_torch.shape[0]
        self.y_len = self.y_torch.shape[0]
        assert self.x_len == self.y_len

    def __getitem__(self, index):
        #index = np.random.randint(0, self.x_len) #데이터셋 생성 과정에서 randomness 추가
        return self.X_torch[index], self.y_torch[index] #(-1, #timesteps, #dim)
    def __len__(self):
        return self.x_len

X_mel_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/Xmel_torch.npy"
y_mel_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/ymel_torch.npy"
X_data = np.load(X_mel_path, allow_pickle = True)
y_data = np.load(y_mel_path)
X_data_ = []
for i in range(X_data.shape[0]):
    X_data_.append(np.array(X_data[i]))
X_data = np.array(X_data_)
#print(X_data)
