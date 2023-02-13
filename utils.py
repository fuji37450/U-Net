import torch
import argparse
import torch.nn.functional as F
from functools import reduce
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', default='data/BCSD/TrainSet/X_crop_resize', type=str,
                        help='directory of images')
    parser.add_argument('--masks_dir', default='data/BCSD/TrainSet/y_crop_resize', type=str,
                        help='directory of ground truth masks')
    parser.add_argument('--result_dir', default='',
                        type=str, help='directory of segmented image')
    parser.add_argument('--n_epoch', default=100, type=int,
                        help='number of epoch to train')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='number of epoch to train')
    parser.add_argument('--model_name', default='',
                        type=str, help='pretrain model name')
    parser.add_argument('--refine', default=False, type=bool,
                        help='To use refine layer or not, default False')
    parser.add_argument('--model_refine_name', default='',
                        type=str, help='pretrain refine model name')
    args = parser.parse_args()
    return args


def dice_coef(pred, target, smooth=1.):
    intersection = (pred * target).sum(2)
    return ((2. * intersection + smooth) / (pred.sum(2) + target.sum(2) + smooth)).mean()


def dice_loss(pred, target):
    loss = 1 - dice_coef(pred, target)

    return loss


l1loss = torch.nn.L1Loss()


def calc_loss(pred, target, metrics, bce_weight=0.5):

    bce = F.binary_cross_entropy_with_logits(
        pred.reshape(target.size()), target.float())

    # pred = torch.sigmoid(pred)
    pred = threshold_preds(pred, target.size()).to('cuda')
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


def swap_func(x):
    '''
    x: np array
    '''
    return x.swapaxes(0, 1).swapaxes(1, 2)


def get_mask_from_pred(x):
    return cv2.threshold((x * 255).astype('uint8'), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


def get_masked_image(img, mask):
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    background = np.full(img.shape, 255, dtype=np.uint8)
    mask = cv2.bitwise_not(mask)
    masked_background = cv2.bitwise_and(background, background, mask=mask)
    result = cv2.bitwise_or(
        (masked_img * 255).astype('uint8'), masked_background)
    return np.asarray(result)


def plot_side_by_side(img_arrays, img_names, filedir):
    nrow, ncol = 1, len(img_arrays)

    for i in range(len(img_arrays[0])):
        _, plots = plt.subplots(nrow, ncol, sharex='all',
                                sharey='all', figsize=(ncol * 4, nrow * 4))
        plt.setp(plots, xticks=[], yticks=[])

        for col in range(ncol):
            if col == 0:
                plots[col].imshow(img_arrays[col][i])
            else:
                plots[col].imshow(img_arrays[col][i], cmap='gray')

        plt.savefig(f'{filedir}/{img_names[i]}.jpg')
        plt.close()


def threshold_preds(preds, shape):
    preds = np.asarray(
        list(map(swap_func, torch.sigmoid(preds).data.cpu().numpy())))
    # preds = np.asarray(list(map(swap_func, preds.data.cpu().numpy())))
    masks = np.asarray(list(map(get_mask_from_pred, preds)))
    preds = torch.from_numpy(masks).reshape(shape) / 255
    return preds
