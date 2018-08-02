import torch
import torch.nn as nn
import torch.nn.functional as F

class ReproductionNet(nn.Module):
    
    def __init__(self):
        super(ReproductionNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv3 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv4 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5 = nn.Conv2d(8, 8, 5, padding=2)
        
        self.dense = nn.Conv2d(8, 48, (144, 1))
        
        self.pool = nn.AvgPool2d((1, 151))
        
        self.fc1 = nn.Linear(48, 24)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        
        x = F.elu(self.dense(x))
        x = x.squeeze()
        
        x = F.elu(self.pool(x))
        x = x.view(-1, 48)
        
        return F.softmax(self.fc1(x), dim=1)


class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 5, (24, 5), stride=(2, 1))
        self.conv2 = nn.Conv2d(5, 1, (12, 20), stride=(1, 10))
        
        self.fc1 = nn.Linear(50 * 13, 50)
        self.fc2 = nn.Linear(50, 24)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 50 * 13)
        x = F.elu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
    
class ConvLstm(nn.Module):
    def __init__(self):
        super(ConvLstm, self).__init__()
        self.conv = nn.Conv1d(144, 24, 20, stride=10)
        self.lstm = nn.LSTM(input_size=24, hidden_size=24, batch_first=True)
        
    def forward(self, x):
        x = F.elu(self.conv(x))
        x = x.transpose(1, 2)
        x = self.lstm(x)[0][:, -1, :].squeeze()
        return F.softmax(x, dim=1)
    
    
class ConvBiLstm(nn.Module):
    def __init__(self):
        super(ConvBiLstm, self).__init__()
        # Convs, standard
        self.conv1 = nn.Conv2d(1, 10, 5, padding=2)
        self.conv2 = nn.Conv2d(10, 10, 5, padding=2)
        self.conv3 = nn.Conv2d(10, 10, 10, stride=2, padding=4)
        
        # Flatten each frame into 48-length vector
        self.dense = nn.Conv2d(10, 48, (72, 1))
        
        # bi-rnn in time
        self.lstm = nn.GRU(input_size=48, hidden_size=48, batch_first=True, num_layers=2, bidirectional=True)
        
        # Pool bi-rnn outputs across frames w/ 10 convolutions
        self.conv_pool = nn.Conv2d(1, 10, (1, 75))
        
        # Linear
        self.fc1 = nn.Linear(96, 48)
        self.fc3 = nn.Linear(48, 24)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        
        x = F.elu(self.dense(x))
        
        x = x.squeeze()
        x = x.transpose(1, 2)
        x = self.lstm(x)[0].transpose(1, 2)
        
        x = x.unsqueeze(1)
        x = F.elu(self.conv_pool(x))
        
        # Take average of each conv's activation for each pitch
        x = x.squeeze()
        x = F.max_pool1d(x.transpose(1, 2), 10)
        x = x.squeeze()
        
        x = F.elu(self.fc1(x))
        return F.softmax(self.fc3(x), dim=1)
    
    

class ConvPlusOne(nn.Module):
    def __init__(self):
        super(ConvPlusOne, self).__init__()
        self.conv1 = nn.Conv1d(144, 24, 3, padding=1)
        self.fc1 = nn.Linear(24, 24)
        
    def forward(self, x):
        x = F.avg_pool1d(F.elu(self.conv1(x)), 151)
        x = x.view(-1, 24)
        x = F.elu(self.fc1(x))
        return F.softmax(x, dim=1)
    
    
class ShallowConvNet(nn.Module):
    
    def __init__(self):
        super(ShallowConvNet, self).__init__()
        self.conv1 = nn.Conv1d(144, 24, 1)
        
    def forward(self, x):
        x = F.avg_pool1d(F.elu(self.conv1(x)), 151)
        x = x.view(-1, 24)
        return F.softmax(x, dim=1)