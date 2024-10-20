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

        # Flatten не является отдельным модулем, его реализация будет в методе forward
        # Вычисляем количество входных признаков для fc1
        # После conv1 и maxpool1: (32-5+1)/2 = 14
        # После conv2 и maxpool2: (14-3+1)/2 = 6
        # Итак, на вход fc1 поступает 5*6*6 элементов
        self.fc1 = nn.Linear(5 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 10)  # CIFAR-10 имеет 10 классов

    def forward(self, x):
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

# Создаем случайный батч изображений
img = torch.Tensor(np.random.random((32, 3, 32, 32)))
# Создаем экземпляр модели
model = ConvNet()
# Прогоняем батч через модель
out = model(img)  # Добавлена закрывающая скобка

# Проверяем размерность выхода
print(out.shape)  # Должно быть [32, 10], так как у нас 32 изображения и 10 классов
