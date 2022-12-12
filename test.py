import torch
from dataloader import SignatureLoader
from model.unet import UNet, Refine
from utils import *

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
    labels = labels.to(device)

    pred = model(inputs)

    if args.refine:
        pred_refine = model_refine(pred, inputs)
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()

    plot_data = [inputs.cpu().numpy(), labels.cpu().numpy(), pred]

    if args.refine:
        pred_refine = torch.sigmoid(pred_refine)
        pred_refine = pred_refine.data.cpu().numpy()
        plot_data.append(pred_refine)

    filename = args.model_name
    if args.refine:
        filename += args.model_refine_name[6:]
    plot_side_by_side(plot_data, filedir=f'imgs/{filename}_{i}/')
