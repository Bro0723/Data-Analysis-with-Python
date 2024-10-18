import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class CustomCNN(nn.Module):  
    def __init__(self):  
        super(CustomCNN, self).__init__()  
          
        # Define the convolutional layers  
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=0)  
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1)  
          
        # Define the pooling layers  
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
          
        # Define the fully connected layers  
        self.fc1 = nn.Linear(5 * 7 * 7, 100)  # 5 filters, 7x7 feature map after pooling  
        self.fc2 = nn.Linear(100, 10)  
      
    def forward(self, x):  
        # Convolution and pooling  
        x = self.conv1(x)  
        x = F.ReLU(x)  
        x = self.maxpool1(x)  
          
        x = self.conv2(x)  
        x = F.ReLU(x)  
        x = self.maxpool2(x)  
          
        # Flatten the tensor for fully connected layers  
        x = x.view(x.size(0), -1)  
          
        # Fully connected layers  
        x = self.fc1(x)  
        x = F.ReLU(x)  
        x = self.fc2(x)  
          
        return x  
  
# Instantiate the network and check its architecture  
net = CustomCNN()  
print(net)  
  
# Create a dummy input tensor and run it through the network  
dummy_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels (RGB), 32x32 image  
output = net(dummy_input)  
  
# Print the output shape  
print(output.shape)  # Should be [1, 10]
