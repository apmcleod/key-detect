import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv1d(12, 32, 1)
        
        self.fc1 = nn.Linear(32, 100)
        self.fc2 = nn.Linear(100, 24)
        
    def forward(self, x):
        x = F.avg_pool1d(F.relu(x), 300)
        x = F.relu(self.fc1(x))
        return self.fc2(x)