import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Сверточный слой 1: входные каналы = 1, выходные каналы = 32, ядро = 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Сверточный слой 2: выходные каналы = 32, выходные каналы = 64, ядро = 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Полносвязный слой 1
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Полносвязный слой 2
        self.fc2 = nn.Linear(128, 10)  # 10 классов на выходе

    def forward(self, x):
        # Применяем первый сверточный слой + ReLU
        x = F.relu(self.conv1(x))
        # Применяем пулинг
        x = F.max_pool2d(x, 2)
        # Применяем второй сверточный слой + ReLU
        x = F.relu(self.conv2(x))
        # Применяем пулинг
        x = F.max_pool2d(x, 2)
        # Переводим данные в одномерный вид для полносвязных слоев
        x = x.view(-1, 64 * 7 * 7)
        # Применяем полносвязный слой 1 + ReLU
        x = F.relu(self.fc1(x))
        # Выходной слой
        x = self.fc2(x)
        return x
