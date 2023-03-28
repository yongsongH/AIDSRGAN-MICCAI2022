# [AIDSRGAN](https://doi.org/10.1007/978-3-031-21014-3_5)
 

Pytorch code for the paper  
["Rethinking Degradation: Radiograph Super-Resolution via AID-SRGAN"](https://doi.org/10.1007/978-3-031-21014-3_5) | [MICCAI 2022 Workshop](https://sites.google.com/view/mlmi2022/)

[Paper :sunny:](https://arxiv.org/abs/2208.03008) | [Poster :snowflake:](https://hyongsong.work) |  [Dataset :cyclone:](https://doi.org/10.6084/m9.figshare.20418036.v3 ) | [Model（Includes PSRGAN: ***kdsrgan_medical_x2_120000_G eg.***）:zap:](https://doi.org/10.6084/m9.figshare.20418384.v3)

## About this repo:

Special thanks for their excellent work:

1) [PSRGAN](https://github.com/yongsongH/Infrared_Image_SR_PSRGAN)
2) [SPSR](https://github.com/Maclory/SPSR)
3) [KAIR](https://github.com/cszn/KAIR)

## Introduction

- We propose a practical degradation model for radiographs, which considers
most possible degradation factors, such as statistical noise, motion blur,
compression, and each of them has variant parameters. This model aims to
represent complex nonlinear degeneracies, unlike current models that focus
more on downsampling. In addition, the degradation model is applied to
synthesize data to train the proposed SR model.

<p align="center">
  <img src="testsets/fig1.png" width="800"/>
</p>

- We propose a medical attention denoising SRGAN model (AID-SRGAN).
An attention mechanism is introduced into the denoising module to make
it more robust to complicated degradation. Moreover, we propose a two-
stage training approach to train the proposed AID-SRGAN, i.e., we first
separately train the denoising module and SR module to obtain a relatively
good denoising network and SR network, respectively. We then jointly train
the denoising and SR modules in an end-to-end manner to further improve
the performance. Finally, it is a flexible framework and easy to follow.

<p align="center">
  <img src="testsets/fig3.R1.png" width="800"/>
</p>

## Using the code:

- Clone this repository:
```bash
git clone https://github.com/yongsongH/AIDSRGAN-MICCAI2022
```

The code is stable using **Python 3.7, Pytorch 0.4.1**

To install all the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Links for downloading the Datasets:

1) MURA SR Dataset - <a href="https://doi.org/10.6084/m9.figshare.20418036.v3"> Link (training)</a> 
2) MURA Test Dataset - <a href="https://doi.org/10.6084/m9.figshare.20418036.v3"> mini and plus (test) </a> 

## Using the Code for our dataset

### Dataset Preparation

:warning: In order to running the code, please carefully check the following information.

#### Training
:one: First, we need three types of datasets. These include real low-resolution images, only downsampled low-resolution images, and high-resolution images. For example (**X4 and X2 are Upsampling factors.**):

 Real low-resolution images  | Only downsampled low-resolution images  | High-resolution images
 ---- | ----- | ------  
 MURA_LR_X2  | MURA_LHR_X2 | MURA_SR_GT
 MURA_LR_X4  | MURA_LHR_X4 | MURA_SR_GT


:two: X2, Like:

```bash

options folder-----
      train_enhance_msrresnet_denosing_add_deloss.json----
      
        "dataroot_H": "trainsets/MURA_SR_GT"              // path of H training dataset
        "dataroot_L": "trainsets/MURA_LR_X2"              // path of L training dataset
        "dataroot_LHR": "trainsets/MURA_LHR_X2"           // path of L_HR training dataset
          .......
```

#### Test
:one: Like training, we need to be careful to choose the right dataset when we test. For example (**X4 and X2 are Upsampling factors.**):

 Real low-resolution images  | Only downsampled low-resolution images  | High-resolution images
 ---- | ----- | ------  
 MURA_mini_X2  | None | MURA_Test_HR
 MURA_mini_X4  | None | MURA_Test_HR
 MURA_plus_X2  | None | MURA_Test_HR
 MURA_plus_X4  | None | MURA_Test_HR


:two: X2, Like:

```bash

main_test_aidsrgan_need_hr.py -----
      
        model_name = 'aidsrgan_390000_x2'
        testset_name = 'MURA_mini_X2'
        hr_testset_name = 'MURA_Test_HR'      # hr path
        need_degradation = False              # default: True
        x8 = False                            # default: False, x8 to boost performance, default: False
        sf = 2                                # scale factor
        show_img = False                      # default: False
          .......
```

- When tested, their GT images will be the same.

:scream: **Please be careful!**

:dizzy_face: **The upsampling factor must correspond to the pre-trained model and dataset. 
Otherwise there will be unexpected errors.** 

### Training Command:

After the ***.json*** file is confirmed, please run the training code. Like:

```bash 
  run main_train_aidsrgan+.py
```


### Testing Command:

After the ***parameters*** are confirmed, please run the test code. Like:

```bash 
  run main_test_aidsrgan_need_hr.py
```


## Citation:

```bash

　# [AIDSRGAN](https://doi.org/10.1007/978-3-031-21014-3_5)
 

Pytorch code for the paper  
["Rethinking Degradation: Radiograph Super-Resolution via AID-SRGAN"](https://doi.org/10.1007/978-3-031-21014-3_5) | [MICCAI 2022 Workshop](https://sites.google.com/view/mlmi2022/)

[Paper :sunny:](https://arxiv.org/abs/2208.03008) | [Poster :snowflake:](https://hyongsong.work) |  [Dataset :cyclone:](https://doi.org/10.6084/m9.figshare.20418036.v3 ) | [Model（Includes PSRGAN: ***kdsrgan_medical_x2_120000_G eg.***）:zap:](https://doi.org/10.6084/m9.figshare.20418384.v3)

## About this repo:

Special thanks for their excellent work:

1) [PSRGAN](https://github.com/yongsongH/Infrared_Image_SR_PSRGAN)
2) [SPSR](https://github.com/Maclory/SPSR)
3) [KAIR](https://github.com/cszn/KAIR)

## Introduction

- We propose a practical degradation model for radiographs, which considers
most possible degradation factors, such as statistical noise, motion blur,
compression, and each of them has variant parameters. This model aims to
represent complex nonlinear degeneracies, unlike current models that focus
more on downsampling. In addition, the degradation model is applied to
synthesize data to train the proposed SR model.

<p align="center">
  <img src="testsets/fig1.png" width="800"/>
</p>

- We propose a medical attention denoising SRGAN model (AID-SRGAN).
An attention mechanism is introduced into the denoising module to make
it more robust to complicated degradation. Moreover, we propose a two-
stage training approach to train the proposed AID-SRGAN, i.e., we first
separately train the denoising module and SR module to obtain a relatively
good denoising network and SR network, respectively. We then jointly train
the denoising and SR modules in an end-to-end manner to further improve
the performance. Finally, it is a flexible framework and easy to follow.

<p align="center">
  <img src="testsets/fig3.R1.png" width="800"/>
</p>

## Using the code:

- Clone this repository:
```bash
git clone https://github.com/yongsongH/AIDSRGAN-MICCAI2022
```

The code is stable using **Python 3.7, Pytorch 0.4.1**

To install all the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Links for downloading the Datasets:

1) MURA SR Dataset - <a href="https://doi.org/10.6084/m9.figshare.20418036.v3"> Link (training)</a> 
2) MURA Test Dataset - <a href="https://doi.org/10.6084/m9.figshare.20418036.v3"> mini and plus (test) </a> 

## Using the Code for our dataset

### Dataset Preparation

:warning: In order to running the code, please carefully check the following information.

#### Training
:one: First, we need three types of datasets. These include real low-resolution images, only downsampled low-resolution images, and high-resolution images. For example (**X4 and X2 are Upsampling factors.**):

 Real low-resolution images  | Only downsampled low-resolution images  | High-resolution images
 ---- | ----- | ------  
 MURA_LR_X2  | MURA_LHR_X2 | MURA_SR_GT
 MURA_LR_X4  | MURA_LHR_X4 | MURA_SR_GT


:two: X2, Like:

```bash

options folder-----
      train_enhance_msrresnet_denosing_add_deloss.json----
      
        "dataroot_H": "trainsets/MURA_SR_GT"              // path of H training dataset
        "dataroot_L": "trainsets/MURA_LR_X2"              // path of L training dataset
        "dataroot_LHR": "trainsets/MURA_LHR_X2"           // path of L_HR training dataset
          .......
```

#### Test
:one: Like training, we need to be careful to choose the right dataset when we test. For example (**X4 and X2 are Upsampling factors.**):

 Real low-resolution images  | Only downsampled low-resolution images  | High-resolution images
 ---- | ----- | ------  
 MURA_mini_X2  | None | MURA_Test_HR
 MURA_mini_X4  | None | MURA_Test_HR
 MURA_plus_X2  | None | MURA_Test_HR
 MURA_plus_X4  | None | MURA_Test_HR


:two: X2, Like:

```bash

main_test_aidsrgan_need_hr.py -----
      
        model_name = 'aidsrgan_390000_x2'
        testset_name = 'MURA_mini_X2'
        hr_testset_name = 'MURA_Test_HR'      # hr path
        need_degradation = False              # default: True
        x8 = False                            # default: False, x8 to boost performance, default: False
        sf = 2                                # scale factor
        show_img = False                      # default: False
          .......
```

- When tested, their GT images will be the same.

:scream: **Please be careful!**

:dizzy_face: **The upsampling factor must correspond to the pre-trained model and dataset. 
Otherwise there will be unexpected errors.** 

### Training Command:

After the ***.json*** file is confirmed, please run the training code. Like:

```bash 
  run main_train_aidsrgan+.py
```


### Testing Command:

After the ***parameters*** are confirmed, please run the test code. Like:

```bash 
  run main_test_aidsrgan_need_hr.py
```


## Citation:

```bash

　@inproceedings{huang2022rethinking,
  title={Rethinking Degradation: Radiograph Super-Resolution via AID-SRGAN},
  author={Huang, Yongsong and Wang, Qingzhong and Omachi, Shinichiro},
  booktitle={Machine Learning in Medical Imaging: 13th International Workshop, MLMI 2022, Held in Conjunction with MICCAI 2022, Singapore, September 18, 2022, Proceedings},
  pages={43--52},
  year={2022},
  organization={Springer}
}


```
## Contact

If you meet any problems, please describe them and [contact](https://hyongsong.work/) me. 

:no_good_man: **Impolite or anonymous emails are not welcome. There may be some difficulties for me to respond to the email without self-introduce. Thank you for understanding.**
