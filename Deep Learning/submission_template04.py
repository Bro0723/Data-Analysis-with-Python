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
        # Эта ячейка не должна выдавать ошибку.
# Если при исполнении ячейки возникает ошибка, то в вашей реализации нейросети есть баги.
img = torch.Tensor(np.random.random((32, 3, 32, 32)))
model = ConvNet()
out = model(img)
# conv1
assert model.conv1.kernel_size == (5, 5), "Неверный размер ядра у conv1"
assert model.conv1.in_channels == 3, "Неверный размер in_channels у conv1"
assert model.conv1.out_channels == 3, "Неверный размер out_channels у conv1"

# pool1
assert model.pool1.kernel_size in (2, (2, 2)), "Неверный размер ядра у pool1"

# conv2
assert model.conv2.kernel_size == (3, 3), "Неверный размер ядра у conv2"
assert model.conv2.in_channels == 3, "Неверный размер in_channels у conv2"
assert model.conv2.out_channels == 5, "Неверный размер out_channels у conv2"

# pool2
assert model.pool2.kernel_size in (2, (2, 2)), "Неверный размер ядра у pool2"

# fc1
assert model.fc1.out_features == 100, "Неверный размер out_features у fc1"

# fc2
assert model.fc2.out_features == 10, "Неверный размер out_features у fc2"
