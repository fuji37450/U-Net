from torch.utils import data
import torch
import cv2
import os
from PIL import Image
import numpy as np

class SignatureLoader(data.Dataset):
    def __init__(self, imgs_root: str, masks_root: str, scale: float = 1.0):
        super().__init__()
        self.imgs_root = imgs_root
        self.masks_root = masks_root
        self.scale = scale

        self.ids = [filename.split('_')[1].split('.')[0]
                    for filename in os.listdir(imgs_root)]

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        if is_mask:
            pil_img = pil_img.point( lambda p: 1 if p > 127 else 0 )

        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(os.path.join(self.imgs_root, f'X_{name}.jpeg'))
        mask = Image.open(os.path.join(self.masks_root, f'y_{name}.jpeg')).convert('L')
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        return torch.FloatTensor(np.array(img)), torch.LongTensor(np.array(mask))
