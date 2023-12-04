import matplotlib.pyplot as plt
import numpy as np
cnn_mfcc_loss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/cnn_mfcc_loss.npy"
cnn_mfcc_valloss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/cnn_mfcc_valloss.npy"
lstm_mfcc_loss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/lstm_mfcc_loss.npy"
lstm_mfcc_valloss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/lstm_mfcc_valloss.npy"
resnet_mfcc_loss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/resnet_mfcc_loss.npy"
resnet_mfcc_valloss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/resnet_mfcc_valloss.npy"
cnn_mel_loss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/cnn_mel_loss.npy"
cnn_mel_valloss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/cnn_mel_valloss.npy"
lstm_mel_loss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/lstm_mel_loss.npy"
lstm_mel_valloss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/lstm_mel_valloss.npy"
resnet_mel_loss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/resnet_mel_loss.npy"
resnet_mel_valloss = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/resnet_mel_valloss.npy"
import torch
def test_model(model, X_test, y_test, device, cnn = True):
    model.eval()
    inputs = X_test
    labels = y_test
    if cnn:
        with torch.no_grad():

            inputs = torch.FloatTensor(inputs).unsqueeze(dim = 1).to(device) #(#data, 1, #coefs, #timetsteps)
            labels = torch.Tensor(labels).to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim = 1)
            #labels = torch.argmax(labels, dim = 1)
    else:
        with torch.no_grad():
            inputs = torch.FloatTensor(inputs).to(device)  # (#data, #dim, #timesteps)
            inputs = inputs.permute(0, 2, 1)
            labels = torch.FloatTensor(labels).to(device)  # (#data, #labels)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            #labels = torch.argmax(labels)
    return np.array(preds.cpu()), np.array(labels.cpu())


