import numpy as np
import cv2
import os
import random
import skimage
from skimage import util

'''
Python  3.6
PyTorch 1.1.0
Windows 10 or Linux

Yongsong HUANG (03/Aug./2022) https://hyongsong.work/

'''

"""
# --------------------------------------------

MICCAI 2022 Workshop  [AID-SRGAN]

Rethinking Degradation: Radiograph Super-Resolution via AID-SRGAN

# --------------------------------------------

In this study, we propose a practical degradation model for radiographs.


This is the final stage of degradation: image compression. We recommend to compress the image twice.

# --------------------------------------------
"""

def read_path(file_pathname,output_pathname):
        #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename)

        #save figure
        cv2.imwrite(output_pathname+"/"+filename, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

#注意*处如果包含家目录（home）不能写成~符号代替
#必须要写成"/home"的格式，否则会报错说找不到对应的目录
#读取的目录

read_path("new_dataset/dataset/new test medical image_real_hr+noise_v2/JPG_De",
          "new_dataset/dataset/new test medical image_real_hr+noise_v2/JPG_De_2")