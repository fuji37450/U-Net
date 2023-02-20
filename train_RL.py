import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict

from dataloader import SignatureLoader
from model.unet import UNet, Refine
from utils import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)
args = parse_args()


def train():

    EPOCHS = args.n_epoch

    np.random.seed(0)
    torch.manual_seed(1)

    train_set = SignatureLoader(
        imgs_root=args.imgs_dir, masks_root=args.masks_dir, scale=0.25)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)

    model = UNet(n_class=1).to(device)
    if args.model_name != '':
        model.load_state_dict(torch.load(f'pths/{args.model_name}.pth'))
    optimizer = optim.Adam(filter(
        lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-5)

    model_refine = Refine().to(device)
    optimizer_refine = optim.Adam(
        filter(lambda p: p.requires_grad, model_refine.parameters()), lr=1e-3)

    best_train_loss = 1e10
    for _ in tqdm(range(1, EPOCHS + 1)):
        epoch_samples = 0

        metrics = defaultdict(float)
        metrics_refine = defaultdict(float)
        for imgs, true_masks in train_loader:
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            # UNet
            model.train()
            outputs = model(imgs)
            l1 = calc_loss(outputs, true_masks, metrics)

            # Refinement Layer
            model_refine.train()
            outputs_refine = model_refine(outputs, imgs)
            l2 = calc_loss(outputs_refine, true_masks, metrics_refine)

            loss = (l1 + l2) / 2

            optimizer.zero_grad()
            optimizer_refine.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_refine.step()

            epoch_samples += imgs.size(0)

        train_loss = metrics_refine['dice'] / epoch_samples
        if train_loss < best_train_loss:
            print(
                f'saving best train model: {1 - train_loss:.2f}.pth')
            best_train_loss = train_loss
            torch.save(model.state_dict(), f'pths/train_BCSD_{1 - train_loss:.2f}.pth')
            torch.save(model_refine.state_dict(),
                       f'pths/train_refine_BCSD_{1 - train_loss:.2f}.pth')
        print_metrics(metrics_refine, epoch_samples)


if __name__ == '__main__':
    train()
