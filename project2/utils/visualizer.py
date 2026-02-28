import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_images(tensor, nrow=8):
    grid = make_grid(tensor, nrow=nrow, normalize=True)
    plt.figure(figsize=(10,4))
    plt.imshow(grid.permute(1,2,0))
    plt.axis("off")
    plt.show()
