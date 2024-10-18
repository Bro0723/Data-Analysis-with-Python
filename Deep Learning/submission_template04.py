class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Первый сверточный слой
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        
        # Первый пулинговый слой
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Второй сверточный слой
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        
        # Второй пулинговый слой
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Полносвязный слой после flatten
        self.fc1 = nn.Linear(in_features=5 * 6 * 6, out_features=100)
        
        # Выходной полносвязный слой
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        # Применение первого сверточного слоя и ReLU
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Применение второго сверточного слоя и ReLU
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Преобразование многомерного тензора в одномерный
        x = x.view(-1, 5 * 6 * 6)
        
        # Применение первого полносвязного слоя и ReLU
        x = F.relu(self.fc1(x))
        
        # Применение выходного полносвязного слоя
        x = self.fc2(x)
        
        return x
