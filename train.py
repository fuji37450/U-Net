import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
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
    LEARNING_RATE = 0.001

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    criterion = criterion.to(device)
    t = time.strftime("%m-%d-%H-%M", time.localtime())

    best_test_loss = 1e10
    best_train_loss = 1e10
    for epoch in tqdm(range(1, EPOCHS + 1)):
        epoch_samples = 0
        
        for imgs, true_masks in tqdm(train_loader):
            model.train()
            metrics = defaultdict(float)

            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs = model(imgs)
            loss = calc_loss(outputs, true_masks, metrics)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            epoch_samples += imgs.size(0)

        train_loss = metrics['dice'] / epoch_samples
        if train_loss < best_train_loss:
            print("saving best train model")
            best_train_loss = train_loss
            torch.save(model.state_dict(), f'pths/train_{metrics["dice"]:.2f}.pth')
        print_metrics(metrics, epoch_samples)
        
        epoch_samples = 0
        for imgs, true_masks in tqdm(test_loader):
            model.eval()
            metrics = defaultdict(float)

            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(imgs)   
            loss = calc_loss(outputs, true_masks, metrics)

            # statistics
            epoch_samples += imgs.size(0)

        test_loss = metrics['dice'] / epoch_samples
        if test_loss < best_test_loss:
            print("saving best model")
            best_test_loss = test_loss
            torch.save(model.state_dict(), f'pths/test_{metrics["dice"]:.2f}.pth')
        print_metrics(metrics, epoch_samples, phase='test')
        


if __name__ == '__main__':
    train()
