import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Определяем класс сверточной нейронной сети
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Определяем сверточные и пулинговые слои
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Полносвязные слои
        self.fc1 = nn.Linear(in_features=self._calculate_flatten_size(), out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def _calculate_flatten_size(self):
        # Вычисляем размер входа в первый полносвязный слой после всех сверточных и пулинговых операций
        with torch.no_grad():
            x = torch.zeros((1, 3, 32, 32))  # Входное изображение
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            return int(np.prod(x.size()[1:]))  # Произведение всех размерностей, кроме первой

    def forward(self, x):
        # Первый сверточный блок
        x = self.pool1(F.relu(self.conv1(x)))

        # Второй сверточный блок
        x = self.pool2(F.relu(self.conv2(x)))

        # Преобразуем данные из 4D в 2D
        x = x.view(x.size(0), -1)

        # Первый полносвязный слой
        x = F.relu(self.fc1(x))

        # Второй полносвязный слой (выходной слой)
        x = self.fc2(x)

        return x
        
