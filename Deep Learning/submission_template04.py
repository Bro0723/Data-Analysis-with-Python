import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 定义网络层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        # 计算展平后的特征图尺寸
        # 输入尺寸为32x32，经过conv1和pool1后变为(32-5+1)/2 = 14
        # 经过conv2和pool2后变为(14-3+1)/2 = 6
        # 因此展平后的特征图尺寸为5 * 6 * 6
        self.fc1 = nn.Linear(in_features=5 * 6 * 6, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        # 实现前向传播
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
model = ConvNet()

# 检查GPU可用性并将模型移至GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
