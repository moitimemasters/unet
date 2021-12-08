from random import shuffle
import os
import numpy as np
from PIL import Image
# 4032 x 3024


def prepare_masks():
    binary = np.load("train_data/binary.npz")
    for file in binary.files:
        image_mask = Image.fromarray(binary[file])
        image = Image.open("train_data/images/" + file)
        size = image_mask.size
        if size[0] < size[1]:
            image_mask=image_mask.rotate(90, expand=True)
            image = image.rotate(90, expand=True)
        image_mask.save("prepared_masks/" + file)
        image.save("prepared_images/" + file)


def train_validate_split(
    output_img_train="train_images/",
    output_img_validate="validate_images/",
    output_mask_train="train_masks/",
    output_mask_validate="validate_masks/",
    p=.75
):
    images_dir = "prepared_images/"
    masks_dir = "prepared_masks/"
    images = os.listdir(images_dir)
    shuffle(images)
    train_part = int(len(images) * p)
    for img in images[:train_part]:
        image = Image.open(images_dir + img)
        image_mask = Image.open(masks_dir + img)
        image.save(output_img_train + img)
        image_mask.save(output_mask_train + img)
    for img in images[train_part:]:
        image = Image.open(images_dir + img)
        image_mask = Image.open(masks_dir + img)
        image.save(output_img_validate + img)
        image_mask.save(output_mask_validate + img)

if __name__ == "__main__":
    # prepare_masks()
    train_validate_split()

