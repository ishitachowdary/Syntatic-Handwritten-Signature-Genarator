import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

IMG_SIZE = 64

class SignaturePairsDataset(Dataset):
    def __init__(self, root_dir, use_gan=False, gan_dir=None):
        self.pairs = []

        genuine_dir = os.path.join(root_dir, "genuine")
        forgery_dir = os.path.join(root_dir, "forgery")

        users = sorted(os.listdir(genuine_dir))

        for user in users:
            genuine_imgs = [
                os.path.join(genuine_dir, user, f)
                for f in os.listdir(os.path.join(genuine_dir, user))
            ]

            forgery_imgs = []
            if os.path.exists(os.path.join(forgery_dir, user)):
                forgery_imgs = [
                    os.path.join(forgery_dir, user, f)
                    for f in os.listdir(os.path.join(forgery_dir, user))
                ]

            # Genuine–Genuine
            for i in range(len(genuine_imgs) - 1):
                self.pairs.append((genuine_imgs[i], genuine_imgs[i+1], 1))

            # Genuine–Forgery
            for g in genuine_imgs:
                if forgery_imgs:
                    self.pairs.append((g, random.choice(forgery_imgs), 0))

            # GAN augmentation (treated as genuine)
            if use_gan and gan_dir and os.path.exists(gan_dir):
                gan_imgs = [
                    os.path.join(gan_dir, f)
                    for f in os.listdir(gan_dir)
                    if f.endswith(".png")
                ]

                if len(gan_imgs) > 0:
                    for g in genuine_imgs:
                        self.pairs.append((g, random.choice(gan_imgs), 1))
                else:
                    print(f"[WARN] No GAN images found in {gan_dir}. Skipping GAN augmentation.")
                for g in genuine_imgs:
                    self.pairs.append((g, random.choice(gan_imgs), 1))
    def __len__(self):
        return len(self.pairs)

    def read_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 127.5 - 1.0
        return torch.tensor(img).unsqueeze(0)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        return self.read_img(p1), self.read_img(p2), torch.tensor(label, dtype=torch.float32)
