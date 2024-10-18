import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from IPython.display import clear_output
    return ConvNet()
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
        # Эта ячейка не должна выдавать ошибку.
# Если при исполнении ячейки возникает ошибка, то в вашей реализации нейросети есть баги.
img = torch.Tensor(np.random.random((32, 3, 32, 32)))
model = ConvNet()
out = model(img)
# conv1
assert model.conv1.kernel_size == (5, 5), "неверный размер ядра у conv1"
assert model.conv1.in_channels == 3, "неверный размер in_channels у conv1"
assert model.conv1.out_channels == 3, "неверный размер out_channels у conv1"

# pool1
assert model.pool1.kernel_size == (2, 2), "неверный размер ядра у pool1"

# conv2
assert model.conv2.kernel_size == (3, 3), "неверный размер ядра у conv2"
assert model.conv2.in_channels == 3, "неверный размер in_channels у conv2"
assert model.conv2.out_channels == 5, "неверный размер out_channels у conv2"

# pool2
assert model.pool1.kernel_size == (2, 2), "неверный размер ядра у pool2"

# fc1
assert model.fc1.out_features == 100, "неверный размер out_features у fc1"
# fc2
assert model.fc2.out_features == 10, "неверный размер out_features у fc2"# conv1
assert model.conv1.kernel_size == (5, 5), "неверный размер ядра у conv1"
assert model.conv1.in_channels == 3, "неверный размер in_channels у conv1"
assert model.conv1.out_channels == 3, "неверный размер out_channels у conv1"

# pool1
assert model.pool1.kernel_size == (2, 2), "неверный размер ядра у pool1"

# conv2
assert model.conv2.kernel_size == (3, 3), "неверный размер ядра у conv2"
assert model.conv2.in_channels == 3, "неверный размер in_channels у conv2"
assert model.conv2.out_channels == 5, "неверный размер out_channels у conv2"

# pool2
assert model.pool1.kernel_size == (2, 2), "неверный размер ядра у pool2"

# fc1
assert model.fc1.out_features == 100, "неверный размер out_features у fc1"
# fc2
assert model.fc2.out_features == 10, "неверный размер out_features у fc2"
