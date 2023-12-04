import torch
from torch import nn
import torch.optim as optim
from configdata import CONFIG
from LSTM import DeepLSTM
from configdata import lstm_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("we are using device: ", device)
torch.manual_seed(156)

save_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch"
X_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/Xdata.npy"
y_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/ydata.npy"
X_mel_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/Xdata_mel.npy"
y_mel_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/ydata_mel.npy"


X_data = np.load(X_path) #(batch_size, 44, 128)
y_data = np.load(y_path) #(batch_size, 14)
X_len = X_data.shape[0]
train_size = 0.7
idx = np.random.permutation(X_len)
train_idx = idx[:round(train_size*X_len)]; test_idx = idx[round(train_size*X_len):]
X_train = X_data[train_idx].astype(np.float32); X_test = X_data[test_idx].astype(np.float32);
y_train = y_data[train_idx].astype(np.float32); y_test = y_data[test_idx].astype(np.float32) ;
audio_train = lstm_dataset(X_train, y_train)
audio_test = lstm_dataset(X_test, y_test)

train_loader = DataLoader(audio_train, batch_size = CONFIG["min_batch"], shuffle = True)
test_loader = DataLoader(audio_test, batch_size = CONFIG["min_batch"], shuffle = True)

model_2 = DeepLSTM(13, 64, 3, 16).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_2.parameters(), lr = CONFIG["lr"])

loss_history = []
val_loss_history = []
init_loss = 999999

for epoch in tqdm(range(CONFIG["epochs"])):
    run_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model_2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()

        if i % 500 == 499:
            loss_history.append(run_loss) #경석형 물어보기
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {run_loss / 500:.3f}')
            run_loss = 0.0
            with torch.no_grad():
                val_loss = 0.0
                for k, (val_inputs, val_labels) in enumerate(test_loader):
                    val_output = model_2(val_inputs)
                    v_loss = criterion(val_output, val_labels)
                    val_loss += v_loss
                print(f'[{epoch + 1}, {i + 1:5d}] val loss: {val_loss / 500:.3f}')
                val_loss_history.append(val_loss.item()/500)

            if val_loss < init_loss:
                torch.save(model_2, os.path.join(save_path, 'lstm_mfcc.pt'))

                init_loss = val_loss

val_loss_history = val_loss_history
loss_history = loss_history
print("finished training")

np.save("C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/lstm_mfcc_valloss.npy", val_loss_history)
np.save("C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/lstm_mfcc_loss.npy", loss_history)
#torch.save(model_1.state_dict(), "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/cnn_mfcc.pt")


lstm_mfcc = torch.load(os.path.join(save_path, 'lstm_mfcc.pt'))
from evaluate import test_model
preds, labels = test_model(lstm_mfcc, X_test, y_test, device, cnn = False)
print(sum(preds == labels)/preds.shape[0])