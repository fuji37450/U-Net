import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict

from dataloader import SignatureLoader
from model.unet import UNet
from utils import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)
args = parse_args()


def train():

    BATCH_SIZE = 32
    EPOCHS = args.n_epoch

    np.random.seed(0)
    torch.manual_seed(1)

    train_set = SignatureLoader(
        imgs_root=args.imgs_dir, masks_root=args.masks_dir, scale=0.25)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(n_class=1).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    
    best_train_loss = 1e10
    for _ in tqdm(range(1, EPOCHS + 1)):
        epoch_samples = 0
        
        for imgs, true_masks in train_loader:
            model.train()
            metrics = defaultdict(float)

            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            outputs = model(imgs)
            loss = calc_loss(outputs, true_masks, metrics)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_samples += imgs.size(0)

        train_loss = metrics['dice'] / epoch_samples
        if train_loss < best_train_loss:
            print("saving best train model")
            best_train_loss = train_loss
            torch.save(model.state_dict(), f'pths/train_{metrics["dice"]:.2f}.pth')
        print_metrics(metrics, epoch_samples)


if __name__ == '__main__':
    train()
