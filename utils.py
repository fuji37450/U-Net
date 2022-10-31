import torch
import argparse
import torch.nn.functional as F
from functools import reduce
import matplotlib.pyplot as plt
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', default='data/BCSD/TrainSet/X_crop_resize', type=str,
                        help='directory of images')
    parser.add_argument('--masks_dir', default='data/BCSD/TrainSet/y_crop_resize', type=str,
                        help='directory of ground truth masks')
    parser.add_argument('--n_epoch', default=100, type=int,
                        help='number of epoch to train')
    args = parser.parse_args()
    return args


def dice_loss(pred, target, smooth=1.):
    pred = pred
    target = target

    intersection = (pred * target).sum(2)
    loss = 1 - ((2. * intersection + smooth) / (pred.sum(2) + target.sum(2) + smooth))
    
    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    
    pred = pred.reshape(target.size()[0], 128, 128)
    bce = F.binary_cross_entropy_with_logits(pred, target.float())

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase='Train'):
    outputs = []
    for k in metrics.keys():
        outputs.append('{}: {:4f}'.format(k, metrics[k] / epoch_samples))

    print('{}: {}'.format(phase, ', '.join(outputs)))


def plot_side_by_side(img_arrays, filedir):
    os.mkdir(filedir)
    nrow, ncol = 1, (len(img_arrays) + 2)
    
    for i in range(len(img_arrays[0])):
        _, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))
        plt.setp(plots, xticks=[], yticks=[]) 
        x, y, pred_y, pred_refine_y = img_arrays[0][i], img_arrays[1][i], img_arrays[2][i], img_arrays[3][i]
        
        x = x.swapaxes(0, 1)
        x = x.swapaxes(1, 2)
        pred_y = pred_y.swapaxes(0, 1)
        pred_y = pred_y.swapaxes(1, 2)
        pred_refine_y = pred_refine_y.swapaxes(0, 1)
        pred_refine_y = pred_refine_y.swapaxes(1, 2)
        _, pred_yo = cv2.threshold((pred_y * 255).astype('uint8'), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, pred_refine_yo = cv2.threshold((pred_refine_y * 255).astype('uint8'), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        plots[0].imshow(x)
        plots[1].imshow(y, cmap='gray')
        plots[2].imshow(pred_y, cmap='gray')
        plots[3].imshow(pred_yo, cmap='gray')
        plots[4].imshow(pred_refine_y, cmap='gray')
        plots[5].imshow(pred_refine_yo, cmap='gray')
        plt.savefig(f'{filedir}{i}')
        plt.close()