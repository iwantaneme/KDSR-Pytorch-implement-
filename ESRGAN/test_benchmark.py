import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model.ESRGAN import ESRGAN as Generator

# parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
# parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
# parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
# opt = parser.parse_args()

UPSCALE_FACTOR = config.upscale_factor
MODEL_NAME = config.model_path
results = {'img': {'psnr': [], 'ssim': []}}

model = Generator(3, 3, scale_factor=UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()

checkpointG = torch.load('epochs/' + MODEL_NAME, map_location=config.device)
netG_state_dict = model.state_dict()
newG_state_dict = {k: v for k, v in checkpointG["state_dict"].items() if k in netG_state_dict}
netG_state_dict.update(newG_state_dict)
model.load_state_dict(netG_state_dict)

test_set = TestDatasetFromFolder(config.hr_dir, upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'RF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

for image_name, lr_image, hr_restore_img, hr_image in test_bar:
    image_name = image_name[0]
    with torch.no_grad():
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()
    with torch.no_grad():
        sr_image = model(lr_image)
    mse = ((hr_image - sr_image) ** 2).data.mean()
    psnr = 10 * log10(1 / mse)
    ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

    test_images = torch.stack(
        [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
         display_transform()(sr_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=3, padding=5)
    utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                     image_name.split('.')[-1], padding=0)
    utils.save_image(hr_restore_img, out_path + image_name.split('.')[0] + '_bicubic.' +
                     image_name.split('.')[-1], padding=0)
    utils.save_image(hr_image, out_path + image_name.split('.')[0] + '_ESRGAN.' +
                     image_name.split('.')[-1], padding=0)


    # save psnr\ssim
    results['img']['psnr'].append(psnr)
    results['img']['ssim'].append(ssim)

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')
