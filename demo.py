import argparse, os, glob
from lib.RDN import Blend as AR
from lib.model import Net as SR
from lib.utils import rgb2y
from SR_modes import run_SR
import torch
from skimage.measure import compare_ssim, compare_psnr
import numpy as np
import cv2

parser = argparse.ArgumentParser(description="Official Pytorch Code for K. Ko et. al., Light Field Super-Resolution via Adaptive Feature Remixing, IEEE Trans. Image Process., vol. 30, pp. 4114-4128, Apr. 2021", usage='use "%(prog)s --help" for more information', formatter_class=argparse.RawTextHelpFormatter)

list_of_modes = ["SR", "AR", "SAR", "ASR"]
mode = {"SR": "Spatial Super-Resolution",
"AR": "Angular Super-Resolution",
"SAR": "Joint Super-Resolution (SR->AR)",
"ASR": "Joint Super-Resolution (AR->SR)"
}
parser.add_argument('--mode', choices=list_of_modes,
                    help='Select test mode.\n' + '\n'.join(['(%d) %s: %s' % (i+1, key, mode[key]) for i, key in enumerate(mode)]))

parser.add_argument('--path', required=True, help='Path for the pretrained model')

parser.add_argument('--data_path', default="./data/HCI1", help='''Path for directory of test dataset.
Default is for HCI1''')

parser.add_argument('--device', default=0, help='GPU device number')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

device = torch.device("cuda:0")
SR_model = SR().to(device)
AR_model = AR().to(device)

### Load pretrained parameters ###
if os.path.exists(args.path):
    print("=> Load the pretrained model from", args.path)
    models = torch.load(args.path)
    SR_model = models['state_dictSR']
    AR_model = models['state_dictAR']
else:
    print("Error: please check path for the pretrained model")

### Load test dataset ###
print("==> Test the performance of %s for %s dataset"%(mode[args.mode], os.path.basename(args.data_path)))

dirs = glob.glob(os.path.join(args.data_path, '*'))
PSNR = []
SSIM = []
for path in dirs:
    print("===> Processing scene of %s"%os.path.basename(path))
    path = os.path.join(path, '%d.png')
    images = [np.float32(rgb2y(cv2.imread(path%i)[..., ::-1])) for i in range(9*9)]
    if args.mode == "SR":
        outputs, targets = run_SR(SR_model, images, factor=2)
        for i, target in enumerate(targets):
            PSNR.append(compare_psnr(outputs[i], target))
            SSIM.append(compare_ssim(outputs[i], target))

print("PSNR : {}, SSIM : {}".format(np.mean(PSNR), np.mean(SSIM)))
