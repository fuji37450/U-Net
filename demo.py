import torch
from dataloader import SignatureLoader
from model.unet import UNet, Refine
from utils import *
import numpy as np
from PIL import Image

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

set = SignatureLoader(imgs_root=args.imgs_dir, mode='test')
loader = torch.utils.data.DataLoader(
    set, batch_size=args.batch_size, shuffle=False)


if __name__ == '__main__':
    result_dir = args.result_dir
    os.mkdir(result_dir)

    for i, (inputs, names) in enumerate(loader):
        inputs = inputs.to(device)
        model.eval()
        preds = model(inputs)

        if args.refine:
            preds = model_refine(preds, inputs)

        inputs = np.asarray(list(map(swap_func, inputs.cpu().numpy())))
        preds = np.asarray(
            list(map(swap_func, torch.sigmoid(preds).data.cpu().numpy())))
        masks = np.asarray(list(map(get_mask_from_pred, preds)))

        maskeds = np.asarray(list(map(get_masked_image, inputs, masks)))

        for (i, masked) in enumerate(maskeds):
            masked = Image.fromarray(masked)
            masked.save(f'{result_dir}/{names[i]}.png')
