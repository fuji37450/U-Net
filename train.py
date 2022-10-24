import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
# from tensorboardX import SummaryWriter
import time
from tqdm import tqdm

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

    model = UNet(n_channels=3, n_classes=1).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8, momentum=0.9)
    amp = False
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # writer = SummaryWriter(log_dir='scalar')

    criterion = criterion.to(device)
    iter_n = 0
    t = time.strftime("%m-%d-%H-%M", time.localtime())

    best_test_score = 0
    for epoch in tqdm(range(1, EPOCHS + 1)):
        for i, (imgs, true_masks) in enumerate(tqdm(train_loader)):
            torch.cuda.empty_cache()

            optimizer.zero_grad()

            imgs, true_masks = imgs.to(device), true_masks.to(device)
            masks_pred = model(imgs)

            loss = criterion(masks_pred.reshape(32, 128, 128), true_masks) \
                + dice_loss(F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(
                                0, 3, 1, 2).float(),
                            multiclass=True)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # writer.add_scalar(
            #     f'{args.model_prefix} {t}/train_loss', loss.item(), iter_n)
            # writer.add_scalar(
            #     f'{args.model_prefix} {t}/train_accuracy', accuracy, iter_n)
            # print(f'loss: {loss.item()}, accuracy: {accuracy}')

            # if (i + 1) % 100 == 0:
            test_score = evaluate(model, test_loader, device)
            print(f'test score:{test_score:.6f}')
            if test_score >= best_test_score:
                best_test_score = test_score
                torch.save(model.state_dict(),
                           f'{args.model_prefix}_{test_score:%}.pth')

            iter_n += 1

            # print('Epoch[{}/{}], iter {}, loss:{:.6f}, accuracy:{}'.format(epoch,
            #                                                                EPOCHS, i, loss.item(), accuracy))

    # writer.close()


if __name__ == '__main__':
    train()
