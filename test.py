import torch
from dataloader import SignatureLoader
from model.unet import UNet, Refine
from utils import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

BATCH_SIZE = 32
model = UNet(n_class=1).to(device)
model_name = 'rl_train_0.06'
model.load_state_dict(torch.load(f'pths/{model_name}.pth'))
model.eval()

model_refine = Refine().to(device)
model_refine_name = 'rl_test_refine_5.91'
model_refine.load_state_dict(torch.load(f'pths/{model_refine_name}.pth'))
model_refine.eval()

if model_refine_name.split('_')[1] == 'train':
    set = SignatureLoader(
            imgs_root='data/BCSD/TrainSet/X_crop_resize', masks_root='data/BCSD/TrainSet/y_crop_resize', scale=0.25)
    loader = torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE, shuffle=False)
else:
    set = SignatureLoader(
        imgs_root='data/BCSD/TestSet/X_crop_resize', masks_root='data/BCSD/TestSet/y_crop_resize', scale=0.25)
    loader = torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE, shuffle=False)

for i, (inputs, labels) in enumerate(loader):
    # Get the first batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Predict
    pred = model(inputs)
    pred_refine = model_refine(pred, inputs)

    # The loss functions include the sigmoid function.
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    pred_refine = torch.sigmoid(pred_refine)
    pred_refine = pred_refine.data.cpu().numpy()

    plot_side_by_side([inputs.cpu().numpy(), labels.cpu().numpy(), pred, pred_refine], filedir=f'imgs/{model_refine_name}_{i}/')