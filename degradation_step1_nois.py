# coding: utf-8
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


This is the earliest part of the degradation: statistical noise.
   
# --------------------------------------------
"""


def gaussianblur(img):
    k_num1 = random.randint(1, 5)  # 高斯核必须为正数和奇数
    k_num2 = random.randint(1, 5)  # 高斯核必须为正数和奇数

    if k_num1 % 2 == 0:
        k_num1 = k_num1 + 1

    if k_num2 % 2 == 0:
        k_num2 = k_num2 + 1

    s_X = random.uniform(0, 1.1)
    s_Y = random.uniform(0, 1.1)
    image_GB = cv2.GaussianBlur(img, ksize=(k_num1, k_num2), sigmaX=s_X, sigmaY=s_Y)

    return image_GB


def motion_blur(image, degree=6, angle=45):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def sp_noise2(img):
    h = img.shape[0]
    w = img.shape[1]
    img1 = img.copy()
    sp = h * w  # 计算图像像素点个数

    snr = random.uniform(0.9, 0.99)  # 范围越小 黑点越密集 咋回事

    NP = int(sp * (1 - snr))  # 计算图像椒盐噪声点个数
    for i in range(NP):
        randx = np.random.randint(1, h - 1)  # 生成一个 1 至 h-1 之间的随机整数
        randy = np.random.randint(1, w - 1)  # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
            img1[randx, randy] = 0
        else:
            img1[randx, randy] = 255

    return img1


def sp_noise(img): # Poisson

    image_GB = util.random_noise(img, mode='poisson')
    noise_gs_img = image_GB * 255

    return noise_gs_img



def read_path(file_pathname,output_pathname):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename)
        ####change to gray
      #（下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图）
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # image_np=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # 确定0-9
        randnum = random.randint(0, 9)
        # randnum = randnum % 10

        if randnum == 0:
            img_out = img
        elif randnum == 1:
            img_out = gaussianblur(img)
        elif randnum == 2:
            img_out = motion_blur(img)
        elif randnum == 3:
            img_out = sp_noise(img)
        elif randnum == 4:
            img_tem = gaussianblur(img)
            img_out = motion_blur(img_tem)
        elif randnum == 5:
            img_tem = gaussianblur(img)
            img_out = sp_noise(img_tem)
        elif randnum == 6:
            img_tem = sp_noise(img)
            img_out = motion_blur(img_tem)
        elif randnum == 7:
            img_tem1 = gaussianblur(img)
            img_tem2 = sp_noise(img_tem1)
            img_out = motion_blur(img_tem2)
        elif randnum == 8:
            img_tem1 = sp_noise(img)
            img_tem2 = gaussianblur(img_tem1)
            img_out = motion_blur(img_tem2)
        elif randnum == 9:
            img_tem1 = motion_blur(img)
            img_tem2 = gaussianblur(img_tem1)
            img_out = sp_noise(img_tem2)

        cv2.imwrite(output_pathname+"/"+filename,img_out)

#注意*处如果包含家目录（home）不能写成~符号代替
#必须要写成"/home"的格式，否则会报错说找不到对应的目录
#读取的目录 原路径-》目标路径

read_path("F:/HYS/1213/medical dataset/X4/test medical image X4/new test medical image_hr","F:/HYS/1213/medical dataset/X4/test medical image X4/new test medical image_real_hr+noise_v2")