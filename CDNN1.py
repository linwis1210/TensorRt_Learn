import torch
import torch.nn as nn
import torch.nn.functional as func

class CDNN(nn.Module):
    def __init__(self):
        super(CDNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=(3, 3), stride=(1, 1), bias=True,padding=1)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=81, kernel_size=(3, 3), stride=(1, 1), bias=True,padding=1)
        self.conv3 = nn.Conv2d(in_channels=81, out_channels=251, kernel_size=(3, 3), stride=(1, 1), bias=True,padding=1)
        self.conv4 = nn.Conv2d(in_channels=251, out_channels=501, kernel_size=(3, 3), stride=(1, 1), bias=True,padding=1)
        
        self.conv51 = nn.Conv2d(in_channels=501, out_channels=501, kernel_size=(1, 1), stride=(1, 1), bias=True,padding=0)
        self.conv52 = nn.Conv2d(in_channels=501, out_channels=501, kernel_size=(3, 3), stride=(1, 1), bias=True,padding=1)

        self.conv61 = nn.Conv2d(in_channels=501, out_channels=501, kernel_size=(1, 1), stride=(1, 1), bias=True,padding=0)
        self.conv62 = nn.Conv2d(in_channels=501, out_channels=501, kernel_size=(3, 3), stride=(1, 1), bias=True,padding=1)
        
        self.conv71 = nn.Conv2d(in_channels=501, out_channels=501, kernel_size=(1, 1), stride=(1, 1), bias=True,padding=0)
        self.conv72 = nn.Conv2d(in_channels=501, out_channels=501, kernel_size=(3, 3), stride=(1, 1), bias=True,padding=1)
        
        self.conv8 = nn.Conv2d(in_channels=501, out_channels=901, kernel_size=(1, 1), stride=(1, 1), bias=True,padding=0)
        
        self.conv91 = nn.Conv2d(in_channels=901, out_channels=901, kernel_size=(1, 1), stride=(1, 1), bias=True,padding=0)
        self.conv92 = nn.Conv2d(in_channels=901, out_channels=901, kernel_size=(1, 1), stride=(1, 1), bias=True,padding=0)
        
        self.conv10 = nn.Conv2d(in_channels=901, out_channels=901, kernel_size=(1, 1), stride=(1, 1), bias=True,padding=0)
        self.dequant = torch.quantization.DeQuantStub()
    def forward(self, x):
#         x = x.unsqueeze(1)
#         x = x.view(1,1,x.shape[0],x.shape[1])
#         x = func.leaky_relu(self.conv1(x), inplace=True)
#         x = func.leaky_relu(self.conv2(x), inplace=True)
#         res1 = x
#         x = func.leaky_relu(self.conv3(x), inplace=True)
#         res2 = x
#         x = func.leaky_relu(self.conv4(x), inplace=True)
#         res = x + res2
#         x = func.leaky_relu(self.conv5(res), inplace=True)
#         res = x + res1
#         x = func.leaky_relu(self.conv6(res), inplace=True)
#         res = x + res1
#         x = self.conv7(res).squeeze()
#         x = x.view(1,1,x.shape[0],x.shape[1])
        x = self.quant(x)
        x = func.leaky_relu(self.conv1(x), inplace=True)
        x = func.leaky_relu(self.conv2(x), inplace=True)
        x = func.leaky_relu(self.conv3(x), inplace=True)
        res = func.leaky_relu(self.conv4(x), inplace=True)
        
        x = func.leaky_relu(self.conv51(res), inplace=True)
        x = func.leaky_relu(self.conv52(x), inplace=True)
        res = x + res
        
        x = func.leaky_relu(self.conv61(res), inplace=True)
        x = func.leaky_relu(self.conv62(x), inplace=True)
        res = x + res

        x = func.leaky_relu(self.conv71(res), inplace=True)
        x = func.leaky_relu(self.conv72(x), inplace=True)
        res = x + res
        
        res = func.leaky_relu(self.conv8(res), inplace=True)
        
        x = func.leaky_relu(self.conv91(res), inplace=True)
        x = func.leaky_relu(self.conv92(x), inplace=True)
        res = x + res
        x = func.leaky_relu(self.conv10(res), inplace=True)
        x = self.dequant(x)
        # x = x.permute(1,2,0)
        return x

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1，padding=(1,1),bias=True):
#         super().__init__()
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=kernel_size // 2,
#             bias=True,
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         out = self.relu(x)
#         return out

class DnnLoss(nn.Module):
    def __init__(self):
        super(DnnLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, t1, t2):
        match_loss = self.mse(t1, t2)
#         cos_loss = 1 - torch.mean(func.cosine_similarity(t1, t2))
        return match_loss
        