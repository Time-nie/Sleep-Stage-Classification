import torch.nn.functional as F
import torch.optim
from torch import nn


class TinySleepNetCNN(nn.Module):
    def __init__(self, conv1_ch=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv1_ch, 50, 6)
        self.batchnorm1 = nn.BatchNorm1d(conv1_ch)
        self.maxpool1 = nn.MaxPool1d(8, 8)
        self.dropout1 = nn.Dropout()
        self.conv2 = nn.Conv1d(conv1_ch, 8, 1)
        self.batchnorm2 = nn.BatchNorm1d(8)
        self.conv3 = nn.Conv1d(8, 8, 1)
        self.batchnorm3 = nn.BatchNorm1d(8)
        self.conv4 = nn.Conv1d(8, 8, 1)
        self.batchnorm4 = nn.BatchNorm1d(8)
        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.flatten = nn.Flatten()
        self.dropout2 = nn.Dropout()

    def forward(self, x):
        old_shape = x.shape
        x = x.chunk(x.shape[0], 0)
        x = torch.cat(x, 1).squeeze(0)
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.dropout1(self.maxpool1(x))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dropout2(x)
        x = x.unsqueeze(0).chunk(old_shape[0], 1)
        x = torch.cat(x)
        return x


class TinySleepNet(nn.Module):
    def __init__(self, conv1_ch=128):
        super().__init__()
        self.cnn = TinySleepNetCNN(conv1_ch=conv1_ch)
        self.lstm = nn.LSTM(input_size=120, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class EmbedSleepNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 50, 6)
        self.maxpool1 = nn.MaxPool1d(8, 8)
        self.conv2 = nn.Conv1d(32, 8, 1)
        self.conv3 = nn.Conv1d(8, 8, 1)
        self.conv4 = nn.Conv1d(8, 8, 1)
        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        old_shape = x.shape
        x = x.chunk(x.shape[0], 0)
        x = torch.cat(x, 1).squeeze(0)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.maxpool1(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = x.unsqueeze(0).chunk(old_shape[0], 1)
        x = torch.cat(x)
        return x


class EmbedSleepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = EmbedSleepNetCNN()
        self.fc1 = nn.Linear(120, 16)
        self.seq = nn.Sequential(
            nn.Conv2d(1, 2, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2, 4, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 8, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 4, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 1, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc1(x)
        x = self.seq(x.unsqueeze(1)).squeeze(1)
        x = self.fc2(x)
        return x
