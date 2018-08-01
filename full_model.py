import torch
import torch.nn as nn
import torch.nn.functional as F

class ReproductionNet(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv3 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv4 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5 = nn.Conv2d(8, 8, 5, padding=2)
        
        self.dense = nn.Conv2d(8, 48, (144, 1))
        
        self.pool = nn.AvgPool2d((48, 1))
        
        self.fc1 = nn.Linear(48, 24)
        
    def forward(self, x):
        print(x.size())
        
        x.unsqueeze_(1)
        
        print(x.size())
        
        x = F.elu(self.conv1(x))
        
        print(x.size())
        
        x = F.elu(self.conv2(x))
        
        print(x.size())
        
        x = F.elu(self.conv3(x))
        
        print(x.size())
        
        x = F.elu(self.conv4(x))
        
        print(x.size())
        
        x = F.elu(self.conv5(x))
        
        print(x.size())
        
        x = F.elu(self.dense(x))
        
        print(x.size())
        
        x = F.elu(self.pool(x))
        
        print(x.size())
        
        x = x.view(-1, 48)
        
        print(x.size())
        
        return F.softmax(self.fc1(x), dim=1)

    
class ShallowConvNet(nn.Module):
    
    def __init__(self):
        super(ShallowConvNet, self).__init__()
        self.conv1 = nn.Conv1d(144, 24, 1)
        
    def forward(self, x):
        x = F.avg_pool1d(F.relu(self.conv1(x)), 151)
        x = x.view(-1, 24)
        return F.softmax(x, dim=1)