import os, cv2, torch
import numpy as np
from torch.utils.data import Dataset

IMG_SIZE = 64

class SignatureDataset(Dataset):
    def __init__(self, root_dir, user_id=None):
        self.files = []
        base = os.path.join(root_dir, "genuine")
        if user_id:
            base = os.path.join(base, user_id)

        for r,_,fs in os.walk(base):
            for f in fs:
                if f.endswith('.png'):
                    self.files.append(os.path.join(r,f))

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 127.5 - 1.0
        return torch.tensor(img).unsqueeze(0)