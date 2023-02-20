import torch
from dataloader import SignatureLoader
from model.unet import UNet, Refine
from utils import *
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

args = parse_args()

model = UNet(n_class=1).to(device)
model.load_state_dict(torch.load(f'pths/{args.model_name}.pth'))
model.eval()

if args.refine:
    model_refine = Refine().to(device)
    model_refine.load_state_dict(torch.load(
        f'pths/{args.model_refine_name}.pth'))
    model_refine.eval()

set = SignatureLoader(imgs_root=args.imgs_dir,
                      masks_root=args.masks_dir, mode='valid')
loader = torch.utils.data.DataLoader(
    set, batch_size=args.batch_size, shuffle=False)

dice_score = 0
# filename = args.model_name
# if args.refine:
#     filename += args.model_refine_name[6:]
# os.mkdir(f'imgs/{filename}')

for i, (inputs, labels, names) in tqdm(enumerate(loader)):
    inputs = inputs.to(device)
    labels = labels.to(device)

    preds = model(inputs)
    if args.refine:
        preds = model_refine(preds, inputs)
    # preds = torch.sigmoid(preds)
    # preds[preds > 0.5] = 1
    # preds[preds < 0.5] = 0
    # print(preds)
    preds = threshold_preds(preds, labels.size()).to(device)
    # print(preds)
    dice_score += dice_coef(preds, labels).data.cpu().numpy()

    # print(dice_loss(pred_refines, labels).data.cpu().numpy())
    # dice_score += dice_loss(pred_refines, labels).data.cpu().numpy()

    # preds = torch.sigmoid(preds).data.cpu().numpy()
    # labels = labels.cpu().numpy()

    # inputs = np.asarray(list(map(swap_func, inputs.cpu().numpy())))
    # preds = list(map(swap_func, preds))
    # masks = np.asarray(list(map(get_mask_from_pred, preds)))

    # plot_data = [inputs, labels, masks]

    # if args.refine:
    #     pred_refines = torch.sigmoid(pred_refines).data.cpu().numpy()
    #     pred_refines = np.asarray(list(map(swap_func, pred_refines)))
    #     mask_refines = np.asarray(list(map(get_mask_from_pred, pred_refines)))
    #     plot_data.append(mask_refines)

    # plot_side_by_side(plot_data, names, filedir=f'imgs/{filename}')

print(f'Average Dice Coefficient:{dice_score / len(loader)}')
