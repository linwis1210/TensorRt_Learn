import torch
import torch.nn as nn
import torch.nn.functional as func

class CDNN(nn.Module):
    def __init__(self, InputNum, OutputNum):
        super(CDNN, self).__init__()
#         self.quant = torch.quantization.QuantStub()
#         self.cnn1 = nn.Conv2d(1, 1, (1,3))

# version1
#         self.fc1 = nn.Linear(InputNum, 151)
#         self.fc2 = nn.Linear(151, 300)
#         self.fc3 = nn.Linear(300, 501)
#         self.fc4 = nn.Linear(501, 301)
#         self.fc5 = nn.Linear(301, 151)
#         self.fc6 = nn.Linear(151, 50)
#         self.fc7 = nn.Linear(50, 300)
#         self.fc8 = nn.Linear(300, OutputNum)
#         self.relu = nn.ReLU(inplace=True)

#version2
        self.fc1 = nn.Linear(InputNum, 81)
        self.fc2 = nn.Linear(81, 120)
        self.fc3 = nn.Linear(120, 170)
        self.fc4 = nn.Linear(170, 150)
        self.fc5 = nn.Linear(150, OutputNum)
#         self.fc6 = nn.Linear(260, OutputNum)
#         self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
#         x = self.quant(x)
#         x = x.view(1,1,-1,9)
#         x = self.cnn1(x)
#         x = x.view(-1, 7)
        x = func.leaky_relu(self.fc1(x), inplace=True)
        x = func.leaky_relu(self.fc2(x), inplace=True)
        x = func.leaky_relu(self.fc3(x), inplace=True)
        x = func.leaky_relu(self.fc4(x), inplace=True)
        x = func.leaky_relu(self.fc5(x), inplace=True)
#         x = func.leaky_relu(self.fc6(x), inplace=True)
#         x = func.leaky_relu(self.fc8(x), inplace=True)
#         x = self.dequant(x)
        return x


Fcnloss = nn.MSELoss(reduction='mean')

class DnnLoss(nn.Module):
    def __init__(self):
        super(DnnLoss, self).__init__()

    def forward(self, t1, t2):
        match_loss = Fcnloss(t1, t2)
        cos_loss = 1 - torch.mean(func.cosine_similarity(t1, t2))
        return match_loss +2*cos_loss
        