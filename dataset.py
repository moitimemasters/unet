import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np


class HandWritingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = []
        self.masks = []
        self.load_image_directory()
        self.load_mask_directory()
    
    def load_image_directory(self):
        self.images = os.listdir(self.image_dir)

    def load_mask_directory(self):  
        self.masks = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        image_path = os.path.join(self.image_dir, self.images[key])
        mask_path = os.path.join(self.mask_dir, self.masks[key])
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.] = 1.

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
                    
        return image, mask
