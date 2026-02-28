import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data_loader_signatures import SignatureDataset
from generator_vanilla_gan import Generator
from discriminator_vanilla_gan import Discriminator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Z_DIM=100; EPOCHS=150; BATCH_SIZE=16; LR=5e-5
USERS=["user_01","user_02","user_03","user_04","user_05"]

criterion = nn.BCELoss()

for user in USERS:
    print(f"Training {user}")
    loader = DataLoader(SignatureDataset('data/cedar',user),batch_size=BATCH_SIZE,shuffle=True)

    G = Generator(Z_DIM).to(DEVICE)
    G.load_state_dict(torch.load('checkpoints/generic/generator.pth',map_location=DEVICE))
    D = Discriminator().to(DEVICE)

    optG = torch.optim.Adam(G.parameters(),LR,(0.5,0.999))
    optD = torch.optim.Adam(D.parameters(),LR,(0.5,0.999))

    fixed_noise = torch.randn(16,Z_DIM,device=DEVICE)

    for epoch in range(1,EPOCHS+1):
        for step, real in enumerate(loader):
            real = real.to(DEVICE); bs = real.size(0)
            z = torch.randn(bs,Z_DIM,device=DEVICE)
            fake = G(z)

            real_lbl = torch.full((bs,1),0.9,device=DEVICE)
            fake_lbl = torch.zeros(bs,1,device=DEVICE)

            optD.zero_grad()
            d_loss = criterion(D(real),real_lbl)+criterion(D(fake.detach()),fake_lbl)
            d_loss.backward()
            if step%2==0: 
                optD.step()

            optG.zero_grad()
            g_loss = criterion(D(fake),real_lbl)
            g_loss.backward(); optG.step()

            if epoch%10==0:
                save_image(G(fixed_noise),f'samples/user_specific/{user}_epoch_{epoch}.png',normalize=True)
            print(user,epoch)

    torch.save(G.state_dict(),f'checkpoints/user_specific/{user}.pth')