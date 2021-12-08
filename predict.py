import torch
import torchvision
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
from model import Unet, DEFAULT_FEATURES
from utils import load_checkpoint
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


pred_transform = A.Compose(
        [
            A.Resize(height=750, width=1000),
            A.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
    )


class PredictDataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.images = os.listdir(self.images_dir)        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.images[index])
        image = np.array(Image.open(image_path).convert("RGB"))
        image = pred_transform(image=image)["image"]
        return image, self.images[index]


def main(weights_path, images_dir, prediction_fname):
    results = {}
    model = Unet(3, 1, DEFAULT_FEATURES).to(device="cuda")
    load_checkpoint(torch.load(weights_path), model)
    dataset = PredictDataset(images_dir)
    loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, pin_memory=True)
    loop = tqdm(loader)
    for i, (img, image_name) in enumerate(loop):
        if i >= 3:
            break
        model.eval()
        with torch.no_grad():
            img = img.to(device="cuda")
            out = torch.sigmoid(model(img))
            out = (out > .5).float()
            torchvision.utils.save_image(out, f"saved_images/pred_{i}.jpg")
            torchvision.utils.save_image(img, f"saved_images/real_{i}.jpg")
            results[image_name[0]] = out.squeeze(1).cpu()
    np.savez_compressed(prediction_fname, **results)


if __name__ == "__main__":
    main("model_final.pth", "validate_images", "predictions/")
