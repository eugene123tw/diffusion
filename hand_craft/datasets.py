import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class SpriteDataset(Dataset):
    def __init__(
        self,
        sprite_paths=[
            "dataset/DungeonCrawl_ProjectUtumnoTileset.png",
            "dataset/animals.png",
            "dataset/monsters.png",
            "dataset/rogues.png",
        ],
        pixel_size=32,
    ):
        super().__init__()
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),            # convert to PIL image
            # transforms.Resize((32, 32)),         # resize to 64x64
            transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
            transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
        ])
        self.sprites = []
        for sprite_path in sprite_paths:
            sprites = crop_spritesheet(sprite_path, pixel_size)
            self.sprites.extend(sprites)

        self.num = len(self.sprites)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # dummy label
        label = torch.tensor(0).to(torch.int64)
        sprite = self.sprites[idx]
        sprite = self.transform(sprite)
        return sprite, label


def crop_spritesheet(
    image_path,
    sprite_size=32,
    show=False,
) -> np.ndarray:
    # Load image using OpenCV
    sheet = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    h, w = sheet.shape[:2]

    sprites = []
    count = 0

    # Loop over grid based on sprite size
    for y in range(0, h, sprite_size):
        for x in range(0, w, sprite_size):
            sprite = sheet[y:y+sprite_size, x:x+sprite_size]
            if not np.any(sprite[:, :, -1]):
                continue  # Skip incomplete edge crops
            sprites.append(sprite[:, :, :-1])  # Exclude alpha channel
            count += 1

    # Plot the sprites if show is True
    if show:
        col = 10
        row = count // col + 1
        fig, axes = plt.subplots(row, col, figsize=(15, 15))
        for i, sprite in enumerate(sprites):
            ax = axes[i // col, i % col]
            ax.imshow(cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA))
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    return np.stack(sprites)

