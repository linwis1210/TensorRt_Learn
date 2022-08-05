import torch
import torch.nn as nn
import torch.nn.functional as func


class CDNN(nn.Module):
    def __init__(self, InputNum, OutputNum):
        super(CDNN, self).__init__()
        self.fc1 = nn.Linear(InputNum, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, OutputNum)
    def forward(self, x):

        x = func.leaky_relu(self.fc1(x), inplace=True)
        x = func.leaky_relu(self.fc2(x), inplace=True)
        x = func.leaky_relu(self.fc3(x), inplace=True)
        x = func.leaky_relu(self.fc4(x), inplace=True)
        x = func.leaky_relu(self.fc5(x), inplace=True)
        return x


Fcnloss = nn.MSELoss(reduction='mean')