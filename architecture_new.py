import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.b1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1))
        self.b2 = nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.Dropout1 = nn.Dropout(0.2)
        #16x16x32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.b3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.b4 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.Dropout2 = nn.Dropout(0.3)
        #8x8x64
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.b5 = nn.BatchNorm2d(num_features=128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.b6 = nn.BatchNorm2d(num_features=128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.Dropout3 = nn.Dropout(0.4)
        #4x4x128
        self.fc1 = nn.Linear(in_features=4 * 4 * 128, out_features=128)
        self.b7 = nn.BatchNorm1d(num_features=128)
        self.Dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.b1(x)
        x = F.relu(self.conv2(x))
        x = self.b2(x)
        x = self.pool1(x)
        x = self.Dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.b3(x)
        x = F.relu(self.conv4(x))
        x = self.b4(x)
        x = self.pool2(x)
        x = self.Dropout2(x)
        x = F.relu(self.conv5(x))
        x = self.b5(x)
        x = F.relu(self.conv6(x))
        x = self.b6(x)
        x = self.pool3(x)
        x = self.Dropout3(x)
        x = x.view(-1, 4 * 4 * 128)  # reshape x
        x = F.relu(self.fc1(x))
        x = self.b7(x)
        x = self.Dropout4(x)
        x = self.fc2(x)
        return x
