from dataclasses import dataclass
import torch


@dataclass
class Config:
    learning_rate = 3e-7
    device = "cuda"
    batch_size = 4
    num_epochs = 3
    num_workers = 8
    image_height = 3000
    image_width = 4000
    pin_memory = True
    load_model = True
    train_img_dir = "train_images/"
    train_mask_dir = "train_masks/"
    validate_img_dir = "validate_images/"
    validate_mask_dir = "validate_masks/"
    


if __name__ == "__main__":
    config = Config()
    print(config.device)
