import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Определение слоев
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Flatten слой не является отдельным модулем в PyTorch, поэтому его определение происходит в методе forward
        # Мы можем предварительно вычислить количество входных признаков для первого полносвязного слоя,
        # зная размерность выхода после второго пулингового слоя.
        # После conv1 и maxpool1: (32-5+1)/2 = 14
        # После conv2 и maxpool2: (14-3+1)/2 = 6
        # Итак, на вход fc1 поступает 5*6*6 элементов
        self.fc1 = nn.Linear(5 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Размерность x ~ [batch_size, 3, 32, 32]
        
        # Первый сверточный слой + ReLU + макс-пулинг
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Второй сверточный слой + ReLU + макс-пулинг
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Преобразование тензора в плоский вектор
        x = x.view(-1, 5 * 6 * 6)
        
        # Полносвязный слой + ReLU
        x = F.relu(self.fc1(x))
        
        # Выходной полносвязный слой
        x = self.fc2(x)
        
        return x
