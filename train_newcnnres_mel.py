import torch
from torch import nn
from configdata import CONFIG
import torch.optim as optim
from newcnn_res import NewCNN_res
from configdata import audio_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("we are using device: ", device)
torch.manual_seed(156)

save_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch"
X_mel_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/Xmel_torch.npy"
y_mel_path = "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/ymel_torch.npy"
X_data = np.load(X_mel_path, allow_pickle = True)
y_data = np.load(y_mel_path)
X_data_ = []
for i in range(X_data.shape[0]):
    X_data_.append(np.array(X_data[i]))
X_data = np.array(X_data_)




X_len = X_data.shape[0]
train_size = 0.7
idx = np.random.permutation(X_len)
train_idx = idx[:round(train_size*X_len)]; test_idx = idx[round(train_size*X_len):]
X_train = X_data[train_idx].astype(np.float32); X_test = X_data[test_idx].astype(np.float32);
y_train = y_data[train_idx].astype(np.float32); y_test = y_data[test_idx].astype(np.float32) ;
audio_train = audio_dataset(X_train, y_train)
audio_test = audio_dataset(X_test, y_test)
train_loader = DataLoader(audio_train, batch_size = CONFIG["min_batch"], shuffle = True, drop_last=True)
test_loader = DataLoader(audio_test, batch_size = CONFIG["min_batch"], shuffle = False)



newcnn_mel = NewCNN_res(64, (80, 41)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(newcnn_mel.parameters(), lr = CONFIG["lr"])

loss_history = []
val_loss_history = []
init_loss = 999999

for epoch in tqdm(range(CONFIG["epochs"])):
    run_loss = 0.0
    for i, data in enumerate(train_loader):
        newcnn_mel.train()
        inputs, labels = data
        optimizer.zero_grad()
        outputs = newcnn_mel(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()

        if i % 50 == 49:
            loss_history.append(run_loss/50) #경석형 물어보기
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {run_loss / 50:.3f}')
            run_loss = 0.0

            with torch.no_grad():
                newcnn_mel.eval()
                val_loss = 0.0
                for k, (val_inputs, val_labels) in enumerate(test_loader):
                    val_output = newcnn_mel(val_inputs)
                    v_loss = criterion(val_output, val_labels)
                    val_loss += v_loss
                print(f'[{epoch + 1}, {i + 1:5d}] val loss: {val_loss / k:.3f}')
                val_loss_history.append(val_loss.item()/k)

            if val_loss < init_loss:
                torch.save(newcnn_mel, os.path.join(save_path, 'newcnnres_mel.pt'))

                init_loss = val_loss

val_loss_history = val_loss_history
loss_history = loss_history
print("finished training")

np.save("C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/newcnnres_mel_valloss.npy", val_loss_history)
np.save("C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/newcnnres_mel_loss.npy", loss_history)
#torch.save(model_1.state_dict(), "C:/Users/infected4098/Desktop/LYJ/Audio Clf with Pytorch/cnn_mfcc.pt")


resnet_mel = torch.load(os.path.join(save_path, 'newcnnres_mel.pt'))
from evaluate import test_model
preds, labels = test_model(resnet_mel, X_test, y_test, device, cnn = True)
print(sum(preds == labels)/preds.shape[0])

