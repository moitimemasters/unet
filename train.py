import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import Unet, DEFAULT_FEATURES
from config import Config

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

configuration = Config()

loss_functions = None

def train_iteration(model, data, targets, loss_function, scaler, optimizer):
    data = data.to(device=configuration.device)
    targets = targets.float().unsqueeze(1).to(device=configuration.device)
    print(data.shape)
    with torch.cuda.amp.autocast():
        predictions = model(data)
        loss = loss_function(predictions, targets)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss


def train_epoch(loader, model, loss_funtion, optimizer, scaler):
    loop = tqdm(loader)
    for batch_index, (data, targets) in enumerate(loop):
        loss = train_iteration(model, data, targets, loss_funtion, scaler, optimizer)
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=configuration.image_height, width=configuration.image_width),
            A.Rotate(limit=35, p=1.),
            A.HorizontalFlip(p=.5),
            A.VerticalFlip(p=.1),
            A.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
    )
    validate_transform = A.Compose(
        [
            A.Resize(height=configuration.image_height, width=configuration.image_width),
            A.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
    )
    
    model = Unet(in_=3, out=1, features=DEFAULT_FEATURES).to(configuration.device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=configuration.learning_rate)

    train_loader, validate_loader = get_loaders(
        configuration.train_img_dir,
        configuration.train_mask_dir,
        configuration.validate_img_dir,
        configuration.validate_mask_dir,
        configuration.batch_size,
        train_transform,
        validate_transform,
        configuration.num_workers,
        configuration.pin_memory,
    )
    if configuration.load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        check_accuracy(validate_loader, model, device=configuration.device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(configuration.num_epochs):
        print("epoch #%d" % (epoch + 1))
        train_epoch(train_loader, model, loss_function, optimizer, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        check_accuracy(validate_loader, model, device=configuration.device)
        save_predictions_as_imgs(
            validate_loader, model, folder="saved_images/", device=configuration.device
        )


if __name__ == "__main__":
    main()
