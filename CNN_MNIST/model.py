import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.R1 = nn.ReLU()
        self.S2 = nn.MaxPool2d(kernel_size=2)
        self.C3 = nn.Conv2d(6, 16, 5, 1, 0)
        self.R2 = nn.ReLU()
        self.S4 = nn.MaxPool2d(2)
        self.C5 = nn.Conv2d(16, 120, 5, 1, 0)
        self.R3 = nn.ReLU()
        self.F6 = nn.Linear(in_features=120, out_features=84)
        self.R4 = nn.ReLU()
        self.OUT = nn.Linear(84, 10)
    def forward(self, x):
        x = self.C1(x)
        x = self.R1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.R2(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.R3(x)
        x = x.view(x.size(0), -1)
        x = self.F6(x)
        x = self.R4(x)
        x = self.OUT(x)
        return x

if __name__ == "__main__":
    model = LeNet()
    a = torch.randn(1, 1, 28, 28)
    b = model(a)
    print(b)