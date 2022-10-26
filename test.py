import torch
from dataloader import SignatureLoader
from model.unet import UNet
from utils import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

BATCH_SIZE = 32
model = UNet(n_class=1).to(device)
model_name = 'train_0.11'
model.load_state_dict(torch.load(f'pths/{model_name}.pth'))
model.eval()   # Set model to the evaluation mode

if model_name.split('_')[0] == 'train':
    set = SignatureLoader(
            imgs_root='data/BCSD/TrainSet/X_crop_resize', masks_root='data/BCSD/TrainSet/y_crop_resize', scale=0.25)
    loader = torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE, shuffle=False)
else:
    set = SignatureLoader(
        imgs_root='data/BCSD/TestSet/X_crop_resize', masks_root='data/BCSD/TestSet/y_crop_resize', scale=0.25)
    loader = torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE, shuffle=False)

# Get the first batch
inputs, labels = next(iter(loader))
inputs = inputs.to(device)
labels = labels.to(device)

# Predict
pred = model(inputs)
# The loss functions include the sigmoid function.
pred = torch.sigmoid(pred)
pred = pred.data.cpu().numpy()

plot_side_by_side([inputs.cpu().numpy(), labels.cpu().numpy(), pred], filedir=f'imgs/{model_name}/')