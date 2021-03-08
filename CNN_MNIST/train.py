import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from model import LeNet

model = LeNet()
Epoch = 5
batch_size = 64
lr = 0.001
train_data = torchvision.datasets.MNIST(root='./data/', train=True, transform=torchvision.transforms.ToTensor(), download=False)
train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
torch.set_grad_enabled(True)
model.train()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(Epoch):
    running_loss = 0.0
    acc = 0.0
    for step, data in enumerate(train_loader):
        x, y = data
        optimizer.zero_grad()
        y_pred = model(x.to(device, torch.float))
        loss = loss_function(y_pred, y.to(device, torch.long))
        loss.backward()
        running_loss += float(loss.data.cpu())
        pred = y_pred.argmax(dim=1)
        acc += (pred.data.cpu() == y.data).sum()
        optimizer.step()
        if step % 100 == 99:
            loss_avg = running_loss / (step + 1)
            acc_avg = float(acc / ((step + 1) * batch_size))
            print('Epoch', epoch + 1, ',step', step + 1, '| Loss_avg: %.4f' % loss_avg, '|Acc_avg:%.4f' % acc_avg)

torch.save(model, './LeNet.pkl')