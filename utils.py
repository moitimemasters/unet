import torch
import torchvision
from dataset import HandWritingDataset
from torch.utils.data.dataloader import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("~> saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("~> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_img_dir,
    train_mask_dir,
    validate_img_dir,
    validate_mask_dir,
    batch_size,
    train_transform,
    validate_transform,
    num_workers=2,
    pin_memory=True,
):
    train_dataset = HandWritingDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    validate_dataset = HandWritingDataset(
        image_dir=validate_img_dir,
        mask_dir=validate_mask_dir,
        transform=validate_transform,
    )

    validate_loader = DataLoader(
        validate_dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return train_loader, validate_loader


def check_accuracy(loader, model, device="cuda"):
    count = 0
    count_right = 0

    f1_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()
            count_right += (predictions == y).sum()
            count += torch.numel(predictions)
            tp = (y * predictions).sum()
            tn = ((1 - y) * (1 - predictions)).sum()
            fp = ((1 - y) * predictions).sum()
            fn = (y * (1 - predictions)).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score += 2 * (precision * recall) / (precision + recall + 1e-8)

    print(
        f"Got {count_right}/{count} with accuracy = {count_right / count * 100:.2f}"
    )
    print(f"F1 score: {f1_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for index, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > .5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{index}.jpg")

    model.train()

