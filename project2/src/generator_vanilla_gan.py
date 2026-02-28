import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512*8*8),
            nn.BatchNorm1d(512*8*8), 
            nn.ReLU(True),
            nn.Unflatten(1,(512,8,8)),
            nn.ConvTranspose2d(512,256,4,2,1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            nn.ConvTranspose2d(128,1,4,2,1), 
            nn.Tanh()
        )
    def forward(self,z): 
        return self.net(z)