import torch
import numpy as np
from typing import Tuple
import groundingdino.datasets.transforms as T
from PIL import Image

# Slightly altered image loading function from grounding dino to allow for cropping
def load_image2(image_path: str, x1, x2, y1, y2) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image = image[y1:y2, x1:x2]

    image_source = image_source.crop((x1,y1,x2,y2))
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def miou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        miou = 1.0 if intersection == 0 else 0.0
    else:
        miou = intersection / union

    return miou

def dice_score(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    if total == 0:
        dice = 1.0 
    else:
        dice = 2 * intersection / total

    return dice