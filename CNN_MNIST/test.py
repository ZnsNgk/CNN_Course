import torch
import torchvision
import torch.utils.data as Data

test_data = torchvision.datasets.MNIST(root='./data/', train=False, transform=torchvision.transforms.ToTensor(), download=False)
test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = torch.load('./LeNet.pkl',map_location=torch.device(device))
net.to(device)
torch.set_grad_enabled(False)
net.eval()
length = test_data.data.size(0)
acc = 0.0

for i, data in enumerate(test_loader):
    x, y = data
    y_pred = net(x.to(device, torch.float))
    pred = y_pred.argmax(dim=1)
    acc += (pred.data.cpu() == y.data).sum()
    print('Predict:', int(pred.data.cpu()), '|Ground Truth:', int(y.data))
acc = (acc / length) * 100
print('Accuracy: %.2f' %acc, '%')