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

    BATCH_SIZE = 32
    EPOCHS = args.n_epoch + 100

    np.random.seed(0)
    torch.manual_seed(1)

    train_set = SignatureLoader(
        imgs_root=args.imgs_dir, masks_root=args.masks_dir, scale=0.25)
    test_set = SignatureLoader(
        imgs_root='data/BCSD/TestSet/X_crop_resize', masks_root='data/BCSD/TestSet/y_crop_resize', scale=0.25)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=2*BATCH_SIZE, shuffle=False)

    model = UNet(n_class=1).to(device)

    model_name = 'rl_train_0.09'
    model.load_state_dict(torch.load(f'pths/{model_name}.pth'))

    model_refine = Refine().to(device)
    optimizer_refine = optim.Adam(filter(lambda p: p.requires_grad, model_refine.parameters()), lr=1e-3)

    best_test_loss = 1e10
    best_train_loss = 1e10    
    for _ in tqdm(range(1, EPOCHS + 1)):
        epoch_samples = 0  
        for imgs, true_masks in train_loader:
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)
            metrics_refine = defaultdict(float)
            
            model.eval()
            model_refine.train()

            outputs = model(imgs)
            outputs_refine = model_refine(outputs, imgs)
            loss_refine = calc_loss(outputs_refine, true_masks, metrics_refine)
            
            optimizer_refine.zero_grad()
            loss_refine.backward()
            optimizer_refine.step()

            epoch_samples += imgs.size(0)
        
        train_loss = metrics_refine['dice'] / epoch_samples
        if train_loss < best_train_loss:
            print("saving best train model")
            best_train_loss = train_loss
            torch.save(model_refine.state_dict(), f'pths/train_refine_{metrics_refine["dice"]:.2f}.pth')
        print_metrics(metrics_refine, epoch_samples)

        epoch_samples = 0
        for imgs, true_masks in test_loader:
            model.eval()
            model_refine.eval()
            metrics_refine = defaultdict(float)

            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            optimizer_refine.zero_grad()

            outputs = model(imgs)
            outputs_refine = model_refine(outputs, imgs)
            loss_refine = calc_loss(outputs_refine, true_masks, metrics_refine)

            epoch_samples += imgs.size(0)

        test_loss = metrics_refine['dice'] / epoch_samples
        if test_loss < best_test_loss:
            print("saving best model")
            best_test_loss = test_loss
            torch.save(model_refine.state_dict(), f'pths/rl_test_refine_{metrics_refine["dice"]:.2f}.pth')
        print_metrics(metrics_refine, epoch_samples, phase='test')


if __name__ == '__main__':
    train()
