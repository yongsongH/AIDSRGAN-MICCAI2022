# -*- coding: utf-8 -*-
import cv2 as cv
import os

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


This is the second stage of degradation: downsampling.

# --------------------------------------------
"""

def read_path(file_pathname,output_pathname):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv.imread(file_pathname+'/'+filename)
        x,y = img.shape[0:2]
        scale = 2
        x = int(x/scale)
        y = int(y/scale)

        newimg = cv.resize(img, (x,y), interpolation=cv.INTER_CUBIC )
        cv.imwrite(output_pathname+"/"+filename,newimg)

#注意*处如果包含家目录（home）不能写成~符号代替
#必须要写成"/home"的格式，否则会报错说找不到对应的目录
#读取的目录
read_path("testsets/heatmap/test","testsets/heatmap/test")