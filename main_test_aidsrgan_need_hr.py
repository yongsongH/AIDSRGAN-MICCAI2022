import os.path
import logging
import re

import numpy as np
from collections import OrderedDict

import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
import torch
import pytorch_ssim
import utilss
from utilss import utils_logger
from utilss import utils_image as util
from utilss import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from models import enhance_model_gan as net

'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Original by Kai Zhang (12/Dec./2019)

New Version by Yongsong HUANG (03/Aug./2022) 
https://hyongsong.work/

'''

"""
# --------------------------------------------

MICCAI 2022 Workshop  [AID-SRGAN]
 
Rethinking Degradation: Radiograph Super-Resolution via AID-SRGAN

# --------------------------------------------
|--model_zoo                            # model_zoo
   |--aidsrgan_390000_x2                # model_name      
|--testset                              # testsets
   |--MURA_mini_X2                      # testset_name
|--results                              # results
   |--                                  # result_name = testset_name + '_' + model_name
   
# --------------------------------------------
"""


def main(json_path='options/train_msrresnet_gan_pr.json'):

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'aidsrgan_390000_x2'
    testset_name = 'MURA_mini_X2'
    hr_testset_name = 'MURA_Test_HR'  # hr path
    need_degradation = False              # default: True
    x8 = False                           # default: False, x8 to boost performance, default: False
    sf = 2                                # scale factor
    show_img = False                     # default: False




    task_current = 'sr'       # 'sr' for super-resolution
    n_channels = 3            # fixed
    model_pool = 'model_zoo'  # fixed
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    noise_level_img = 0       # fixed: 0, noise level for LR image
    result_name = testset_name + '_' + model_name
    border = sf if task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM
    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    H_path = os.path.join(testsets, hr_testset_name)  # H_path, for High-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_msrresnet import D_add_deloss_denosing_RCABlock_msrresnet0 as net
    # from models.network_msrresnet import En_MSRResNet0 as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=16, upscale=sf)

    # print(model)
    model.load_state_dict(torch.load(model_path), strict=False)

    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False

    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    logger.info('model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        # degradation process, bicubic downsampling
        if need_degradation:
            img_L = util.modcrop(img_L, sf)
            img_L = util.imresize_np(img_L, 1/sf)
            # img_L = util.uint2single(util.single2uint(img_L))
            # np.random.seed(seed=0)  # for reproducibility
            # img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        util.imshow(util.single2uint(img_L), title='LR image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        if not x8:
            img_E,_ = model(img_L)
        else:
            img_E = utilss.utils_model.test_mode(model, img_L, mode=3, sf=sf)



        img_E = util.tensor2uint(img_E)

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------

            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()
            img_H = util.modcrop(img_H, sf)

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None

            if np.ndim(img_H) == 3:  # RGB image
                img_E_y = util.rgb2ycbcr(img_E, only_y=True)
                img_H_y = util.rgb2ycbcr(img_H, only_y=True)
                psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=border)
                ssim_y = util.calculate_ssim(img_E_y, img_H_y, border=border)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+'.png'))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - x{} --PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr, ave_ssim))
        if np.ndim(img_H) == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info('Average PSNR/SSIM( Y ) - {} - x{} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr_y, ave_ssim_y))

if __name__ == '__main__':

    main()

