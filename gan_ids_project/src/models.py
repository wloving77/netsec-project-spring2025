import torch
import torch.nn as nn

class NetworkAnomalyDetector(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NetworkAnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        if num_classes == 2:
            self.fc3 = nn.Linear(32, 1)
        else:
            self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DeeperNetworkAnomalyDetector(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        if num_classes == 2:
            self.fc4 = nn.Linear(32, 1)
        else:
            self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)