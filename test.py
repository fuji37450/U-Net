import torch
from dataloader import SignatureLoader
from model.unet import UNet, Refine
from utils import *
import numpy as np

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

args = parse_args()

BATCH_SIZE = 32
model = UNet(n_class=1).to(device)
model.load_state_dict(torch.load(f'pths/{args.model_name}.pth'))
model.eval()

if args.refine:
    model_refine = Refine().to(device)
    model_refine.load_state_dict(torch.load(
        f'pths/{args.model_refine_name}.pth'))
    model_refine.eval()

set = SignatureLoader(imgs_root=args.imgs_dir,
                      masks_root=args.masks_dir, scale=0.25)
loader = torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE, shuffle=False)

for i, (inputs, labels) in enumerate(loader):
    inputs = inputs.to(device)
    labels = labels.to(device).cpu().numpy()

    preds = model(inputs)

    if args.refine:
        pred_refines = model_refine(preds, inputs)

    preds = torch.sigmoid(preds).data.cpu().numpy()

    inputs = np.asarray(list(map(swap_func, inputs.cpu().numpy())))
    preds = list(map(swap_func, preds))
    masks = np.asarray(list(map(get_mask_from_pred, preds)))

    plot_data = [inputs, labels, masks]

    if args.refine:
        pred_refines = torch.sigmoid(pred_refines).data.cpu().numpy()
        pred_refines = np.asarray(list(map(swap_func, pred_refines)))
        mask_refines = np.asarray(list(map(get_mask_from_pred, pred_refines)))
        plot_data.append(mask_refines)

    filename = args.model_name
    if args.refine:
        filename += args.model_refine_name[6:]

    plot_side_by_side(plot_data, filedir=f'imgs/{filename}_{i}/')
