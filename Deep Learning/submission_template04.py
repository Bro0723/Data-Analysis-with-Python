import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class CustomCNN(nn.Module):  
    def __init__(self):  
        super(CustomCNN, self).__init__()  
          
        # conv1: 3 filters of size (5, 5)  
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=0)  
          
        # maxpool1: kernel size of 2  
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
          
        # conv2: 5 filters of size (3, 3)  
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1)  
          
        # maxpool2: kernel size of 2  
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
          
        # flatten layer (not explicitly defined in PyTorch, handled in forward method)  
          
        # fc1: 100 neurons  
        self.fc1 = nn.Linear(5 * 7 * 7, 100)  # Calculated based on input size and pooling operations  
          
        # fc2: 10 neurons  
        self.fc2 = nn.Linear(100, 10)  
      
    def forward(self, x):  
        # conv1 -> ReLU -> maxpool1  
        x = self.conv1(x)  
        x = F.ReLU(x)  
        x = self.maxpool1(x)  
          
        # conv2 -> ReLU -> maxpool2  
        x = self.conv2(x)  
        x = F.ReLU(x)  
        x = self.maxpool2(x)  
          
        # flatten  
        x = x.view(x.size(0), -1)  # Flatten the tensor  
          
        # fc1 -> ReLU  
        x = self.fc1(x)  
        x = F.ReLU(x)  
          
        # fc2  
        x = self.fc2(x)  
          
        return x  
