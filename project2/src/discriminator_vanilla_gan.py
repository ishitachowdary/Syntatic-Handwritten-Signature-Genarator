import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,128,4,2,1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,4,2,1), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512,4,2,1), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2),
            nn.Flatten(), 
            nn.Linear(512*8*8,1), 
            nn.Sigmoid()
        )
    def forward(self,x): 
        return self.net(x)