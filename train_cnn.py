import torch
from torch import nn
import torch.optim as optim
from configdata import CONFIG
from deepcnn import DeepCNN
from configdata import audio_dataset
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

X_data = np.load(X_path)
y_data = np.load(y_path)
X_len = X_data.shape[0]
train_size = 0.7
idx = np.random.permutation(X_len)
train_idx = idx[:round(train_size*X_len)]; test_idx = idx[round(train_size*X_len):]
X_train = X_data[train_idx].astype(np.float32); X_test = X_data[test_idx].astype(np.float32);
y_train = y_data[train_idx].astype(np.float32); y_test = y_data[test_idx].astype(np.float32) ;
audio_train = audio_dataset(X_train, y_train)
audio_test = audio_dataset(X_test, y_test)
train_loader = DataLoader(audio_train, batch_size = CONFIG["min_batch"], shuffle = True)
test_loader = DataLoader(audio_test, batch_size = CONFIG["min_batch"], shuffle = True)

model_1 = DeepCNN([16, 32], (13, 49), (3, 3)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1.parameters(), lr = CONFIG["lr"])

loss_history = []
val_loss_history = []
init_loss = 999999

for epoch in tqdm(range(CONFIG["epochs"])):
    run_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model_1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()

        if i % 500 == 499:
            loss_history.append(run_loss/500) #경석형 물어보기
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {run_loss / 500:.3f}')
            run_loss = 0.0
            with torch.no_grad():
                val_loss = 0.0
                for k, (val_inputs, val_labels) in enumerate(test_loader):
                    val_output = model_1(val_inputs)
                    v_loss = criterion(val_output, val_labels)
                    val_loss += v_loss
                print(f'[{epoch + 1}, {i + 1:5d}] val loss: {val_loss / 500:.3f}')
                val_loss_history.append(val_loss.item()/500)

            if val_loss < init_loss:
                torch.save(model_1, os.path.join(save_path, 'cnn_mfcc.pt'))

                init_loss = val_loss

val_loss_history = val_loss_history
loss_history = loss_history
print("finished training")

np.save("C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/cnn_mfcc_valloss.npy", val_loss_history)
np.save("C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/cnn_mfcc_loss.npy", loss_history)
#torch.save(model_1.state_dict(), "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/cnn_mfcc.pt")


cnn_mfcc = torch.load(os.path.join(save_path, 'cnn_mfcc.pt'))
from evaluate import test_model
preds, labels = test_model(cnn_mfcc, X_test, y_test, device, cnn = True)
print(sum(preds == labels)/preds.shape[0])

