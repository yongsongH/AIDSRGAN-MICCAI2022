import torch
import torchvision
import math
import torch.nn as nn
import models.basicblock as B
import functools
import torch.nn.functional as F
import torch.nn.init as init
from models.KD_block import KDB
from models.network_dncnn import DnCNN


"""
# --------------------------------------------
# modified SRResNet
#   -- MSRResNet0 (v0.0)
#   -- MSRResNet1 (v0.1)
# --------------------------------------------
References:
@inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={European Concerence on Computer Vision (ECCV)},
  pages={0--0},
  year={2018}
}
@inproceedings{ledig2017photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan and others},
  booktitle={IEEE concerence on computer vision and pattern recognition},
  pages={4681--4690},
  year={2017}
}
# --------------------------------------------
"""
class D_add_deloss_denosing_no_deloss_semsrresnet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, nb=16, upscale=4, act_mode='R',
                 upsample_mode='upconv'):  # nc setting 32|64|128
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(D_add_deloss_denosing_no_deloss_semsrresnet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        """

        add dncnn

        """
        # m_head = B.conv(in_nc, nc, mode='C')
        bias = True
        m_dncnn_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_dncnn_body = [B.RCAGroup(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_dncnn_tail = B.conv(nc, nc, mode='C', bias=bias)

        self.dncnn_head = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_tail)

        """
        add de_loss

        """
        m_dncnn_de_loss_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.dncnn_de_loss = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_de_loss_tail)

        # m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        ############ enhance ############

        # m_hance = [B.RRDB(nc) for _ in range(1)]
        # D_hance = [B.DResidualBlock(nc,nc)] #分离卷积
        # m_hance = [B.RCABlock(nc) for _ in range(1)]

        #################################

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.KD = KDB(nc)
        #####################################
        # self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        # m_hance 增强型模块
        self.head = self.dncnn_head
        self.body = B.ShortcutBlock(B.sequential(*m_body))
        self.tail = B.sequential(*m_uper, m_tail)
        self.copress = nn.Conv2d(2 * nc, nc, kernel_size=1)

    def forward(self, x):
        x_head = self.head(x)
        x_kd = self.KD(x_head)
        x_body = self.body(x_head)
        x_body = self.copress(torch.cat([x_kd, x_body], dim=1))
        result = self.tail(x_body)
        # dncnn de_loss
        # deloss = self.dncnn_de_loss(x) - x

        return result

class D_add_deloss_denosing_RCABlock_Dropout_msrresnet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, nb=16, upscale=4, act_mode='R',
                 upsample_mode='upconv', p=0.1):  # nc setting 32|64|128
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(D_add_deloss_denosing_RCABlock_Dropout_msrresnet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        """

        Add dncnn

        """
        # m_head = B.conv(in_nc, nc, mode='C')
        # dropout_layer = B.conv(nc, nc, mode='CRD')

        bias = True
        # m_dncnn_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        # dropout_layer = B.conv(in_nc, nc, mode='CRD', bias=bias)
        m_dncnn_head = B.conv(in_nc, nc, mode='CRD' , bias=bias)
        m_dncnn_body = [B.RCABlock(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_dncnn_tail = B.conv(nc, nc, mode='C', bias=bias)

        self.dncnn_head = B.sequential( m_dncnn_head, *m_dncnn_body, m_dncnn_tail)


        """
        add de_loss

        """
        m_dncnn_de_loss_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.dncnn_de_loss = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_de_loss_tail)

        # m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        ############ enhance ############

        # m_hance = [B.RRDB(nc) for _ in range(1)]
        # D_hance = [B.DResidualBlock(nc,nc)] #分离卷积
        # m_hance = [B.RCABlock(nc) for _ in range(1)]

        #################################

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.KD = KDB(nc)
        #####################################
        # self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        # m_hance 增强型模块
        self.drop_layer = nn.Dropout(p=p)
        self.head = self.dncnn_head
        self.body = B.ShortcutBlock(B.sequential(*m_body))
        self.tail = B.sequential(*m_uper, m_tail)
        self.copress = nn.Conv2d(2 * nc, nc, kernel_size=1)

    def forward(self, x):
        x_head = self.head(x)
        # x_dropout = self.drop_layer(x_head)
        x_kd = self.KD(x_head)
        x_body = self.body(x_head)
        x_body = self.copress(torch.cat([x_kd, x_body], dim=1))
        result = self.tail(x_body)
        # dncnn de_loss
        deloss = self.dncnn_de_loss(x) - x

        return result, deloss


class D_add_deloss_denosing_RCABlock_msrresnet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, nb=16, upscale=4, act_mode='R',
                 upsample_mode='upconv'):  # nc setting 32|64|128
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(D_add_deloss_denosing_RCABlock_msrresnet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        """

        Add dncnn
        
        """
        # m_head = B.conv(in_nc, nc, mode='C')
        bias = True
        m_dncnn_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_dncnn_body = [B.RCABlock(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_dncnn_tail = B.conv(nc, nc, mode='C', bias=bias)

        self.dncnn_head = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_tail)

        """
        add de_loss

        """
        m_dncnn_de_loss_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.dncnn_de_loss = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_de_loss_tail)

        # m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        ############ enhance ############

        # m_hance = [B.RRDB(nc) for _ in range(1)]
        # D_hance = [B.DResidualBlock(nc,nc)] #分离卷积
        # m_hance = [B.RCABlock(nc) for _ in range(1)]

        #################################

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.KD = KDB(nc)
        #####################################
        # self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        # m_hance 增强型模块
        self.head = self.dncnn_head
        self.body = B.ShortcutBlock(B.sequential(*m_body))
        self.tail = B.sequential(*m_uper, m_tail)
        self.copress = nn.Conv2d(2 * nc, nc, kernel_size=1)

    def forward(self, x):
        x_head = self.head(x)
        x_kd = self.KD(x_head)
        x_body = self.body(x_head)
        x_body = self.copress(torch.cat([x_kd, x_body], dim=1))
        result = self.tail(x_body)
        # dncnn de_loss
        deloss = self.dncnn_de_loss(x) - x

        return result, deloss


class D_add_deloss_denosing_RCABlock_msrresnet0_test(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, nb=16, upscale=4, act_mode='R',
                 upsample_mode='upconv'):  # nc setting 32|64|128
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(D_add_deloss_denosing_RCABlock_msrresnet0_test, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        """

        Add dncnn

        """
        # m_head = B.conv(in_nc, nc, mode='C')
        bias = True
        m_dncnn_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_dncnn_body = [B.RCABlock(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_dncnn_tail = B.conv(nc, nc, mode='C', bias=bias)

        self.dncnn_head = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_tail)

        """
        add de_loss

        """
        m_dncnn_de_loss_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.dncnn_de_loss = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_de_loss_tail)

        # m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        ############ enhance ############

        # m_hance = [B.RRDB(nc) for _ in range(1)]
        # D_hance = [B.DResidualBlock(nc,nc)] #分离卷积
        # m_hance = [B.RCABlock(nc) for _ in range(1)]

        #################################

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.KD = KDB(nc)
        #####################################
        # self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        # m_hance 增强型模块
        self.head = self.dncnn_head
        self.body = B.ShortcutBlock(B.sequential(*m_body))
        self.tail = B.sequential(*m_uper, m_tail)
        self.copress = nn.Conv2d(2 * nc, nc, kernel_size=1)

    def forward(self, x):
        x_head = self.head(x)
        x_kd = self.KD(x_head)
        x_body = self.body(x_head)
        x_body = self.copress(torch.cat([x_kd, x_body], dim=1))
        result = self.tail(x_body)
        # dncnn de_loss
        deloss = self.dncnn_de_loss(x) - x

        return result

class D_add_deloss_denosing_semsrresnet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, nb=16, upscale=4, act_mode='R',
                 upsample_mode='upconv'):  # nc setting 32|64|128
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(D_add_deloss_denosing_semsrresnet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        """

        add dncnn

        """
        # m_head = B.conv(in_nc, nc, mode='C')
        bias = True
        m_dncnn_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_dncnn_body = [B.RCAGroup(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_dncnn_tail = B.conv(nc, nc, mode='C', bias=bias)

        self.dncnn_head = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_tail)

        """
        add de_loss

        """
        m_dncnn_de_loss_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.dncnn_de_loss = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_de_loss_tail)

        # m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        ############ enhance ############

        # m_hance = [B.RRDB(nc) for _ in range(1)]
        # D_hance = [B.DResidualBlock(nc,nc)] #分离卷积
        # m_hance = [B.RCABlock(nc) for _ in range(1)]

        #################################

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.KD = KDB(nc)
        #####################################
        # self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        # m_hance 增强型模块
        self.head = self.dncnn_head
        self.body = B.ShortcutBlock(B.sequential(*m_body))
        self.tail = B.sequential(*m_uper, m_tail)
        self.copress = nn.Conv2d(2 * nc, nc, kernel_size=1)

    def forward(self, x):
        x_head = self.head(x)
        x_kd = self.KD(x_head)
        x_body = self.body(x_head)
        x_body = self.copress(torch.cat([x_kd, x_body], dim=1))
        result = self.tail(x_body)
        # dncnn de_loss
        deloss = self.dncnn_de_loss(x) - x

        return result, deloss

class D_add_deloss_denosing_msrresnet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, nb=16, upscale=4, act_mode='R',
                 upsample_mode='upconv'):  # nc setting 32|64|128
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(D_add_deloss_denosing_msrresnet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        """

        add dncnn

        """
        # m_head = B.conv(in_nc, nc, mode='C')
        bias = True
        m_dncnn_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_dncnn_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_dncnn_tail = B.conv(nc, nc, mode='C', bias=bias)

        self.dncnn_head = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_tail)

        """
        add de_loss
        
        """
        m_dncnn_de_loss_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.dncnn_de_loss = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_de_loss_tail)

        # m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        ############ enhance ############

        # m_hance = [B.RRDB(nc) for _ in range(1)]
        # D_hance = [B.DResidualBlock(nc,nc)] #分离卷积
        # m_hance = [B.RCABlock(nc) for _ in range(1)]

        #################################

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.KD = KDB(nc)
        #####################################
        # self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        # m_hance 增强型模块
        self.head = self.dncnn_head
        self.body = B.ShortcutBlock(B.sequential(*m_body))
        self.tail = B.sequential(*m_uper, m_tail)
        self.copress = nn.Conv2d(2 * nc, nc, kernel_size=1)

    def forward(self, x):
        x_head = self.head(x)
        x_kd = self.KD(x_head)
        x_body = self.body(x_head)
        x_body = self.copress(torch.cat([x_kd, x_body], dim=1))
        result = self.tail(x_body)
        # dncnn de_loss
        deloss = self.dncnn_de_loss(x)-x


        return result,deloss

#######################################
class D_2denosing_msrresnet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, nb=16, upscale=4, act_mode='R',
                 upsample_mode='upconv'):  # nc setting 32|64|128
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(D_2denosing_msrresnet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        """

        add dncnn

        """
        # m_head = B.conv(in_nc, nc, mode='C')
        bias = True
        m_dncnn_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_dncnn_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_dncnn_tail = B.conv(nc, nc, mode='C', bias=bias)

        self.dncnn_head = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_tail)

        # m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        ############ enhance ############

        # m_hance = [B.RRDB(nc) for _ in range(1)]
        # D_hance = [B.DResidualBlock(nc,nc)] #分离卷积
        # m_hance = [B.RCABlock(nc) for _ in range(1)]

        #################################

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in range(n_upscale)]

        # H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        # H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        # m_tail = B.sequential(H_conv0, H_conv1)
        bias = True
        m_tail_dncnn_head = B.conv(nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_tail_dncnn_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail_dncnn_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        m_tail = B.sequential(m_tail_dncnn_head, *m_tail_dncnn_body, m_tail_dncnn_tail)

        self.KD = KDB(nc)
        #####################################
        # self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        # m_hance 增强型模块
        self.head = self.dncnn_head
        self.body = B.ShortcutBlock(B.sequential(*m_body))
        self.tail = B.sequential(*m_uper, m_tail)
        self.copress = nn.Conv2d(2 * nc, nc, kernel_size=1)

    def forward(self, x):
        x_head = self.head(x)
        x_kd = self.KD(x_head)
        x_body = self.body(x_head)
        x_body = self.copress(torch.cat([x_kd, x_body], dim=1))
        result = self.tail(x_body)
        return result
#######################################
class D_denosing_msrresnet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, nb=16, upscale=4, act_mode='R', upsample_mode='upconv'):#nc setting 32|64|128
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(D_denosing_msrresnet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        """
        
        add dncnn
        
        """
        # m_head = B.conv(in_nc, nc, mode='C')
        bias = True
        m_dncnn_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_dncnn_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_dncnn_tail = B.conv(nc, nc, mode='C', bias=bias)

        self.dncnn_head = B.sequential(m_dncnn_head, *m_dncnn_body, m_dncnn_tail)


        # m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        ############ enhance ############

        # m_hance = [B.RRDB(nc) for _ in range(1)]
        # D_hance = [B.DResidualBlock(nc,nc)] #分离卷积
        # m_hance = [B.RCABlock(nc) for _ in range(1)]

        #################################


        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]




        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)


        self.KD = KDB(nc)
        #####################################
        # self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        # m_hance 增强型模块
        self.head = self.dncnn_head
        self.body = B.ShortcutBlock(B.sequential(*m_body))
        self.tail = B.sequential(*m_uper, m_tail)
        self.copress=nn.Conv2d(2*nc,nc,kernel_size=1)



    def forward(self, x):
        x_head  = self.head(x)
        x_kd    = self.KD(x_head)
        x_body  = self.body(x_head)
        x_body  = self.copress(torch.cat([x_kd,x_body],dim=1))
        result  = self.tail(x_body)
        return result


#######################################
class D_msrresnet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=128, nb=16, upscale=4, act_mode='R', upsample_mode='upconv'):#nc setting 32|64|128
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(D_msrresnet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')

        # m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        ############ enhance ############

        # m_hance = [B.RRDB(nc) for _ in range(1)]
        # D_hance = [B.DResidualBlock(nc,nc)] #分离卷积
        # m_hance = [B.RCABlock(nc) for _ in range(1)]

        #################################


        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]




        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)


        self.KD = KDB(nc)
        #####################################
        # self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        # m_hance 增强型模块
        self.head = m_head
        self.body = B.ShortcutBlock(B.sequential(*m_body))
        self.tail = B.sequential(*m_uper, m_tail)
        self.copress=nn.Conv2d(2*nc,nc,kernel_size=1)



    def forward(self, x):
        x_head  = self.head(x)
        x_kd    = self.KD(x_head)
        x_body  = self.body(x_head)
        x_body  = self.copress(torch.cat([x_kd,x_body],dim=1))
        result  = self.tail(x_body)
        return result

#######################################
class En_MSRResNet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=4, act_mode='R', upsample_mode='upconv'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(En_MSRResNet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')

        m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        ############ enhance ############

        # m_hance = [B.RRDB(nc) for _ in range(10)]
        m_hance = [B.RCABlock(nc) for _ in range(1)]

        #################################


        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), B.sequential(*m_hance), *m_uper, m_tail)
        # m_hance 增强型模块

    def forward(self, x):
        x = self.model(x)
        return x

# --------------------------------------------
# modified SRResNet v0.0
# https://github.com/xinntao/ESRGAN
# --------------------------------------------
class MSRResNet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=4, act_mode='R', upsample_mode='upconv'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(MSRResNet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')

        m_body = [B.ResBlock(nc, nc, mode='C'+'B'+act_mode+'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'+'B'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x

###################### Ori MSRResNet0###########################
class MSRResNet_0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=4, act_mode='R', upsample_mode='upconv'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(MSRResNet_0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')

        m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


#########################################################
# --------------------------------------------
# modified SRResNet v0.1
# https://github.com/xinntao/ESRGAN
# --------------------------------------------
class MSRResNet1(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=4, act_mode='R', upsample_mode='upconv'):
        super(MSRResNet1, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nc, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nc=nc)
        self.recon_trunk = make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nc, nc * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nc, nc * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nc, nc * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nc, nc * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nc, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nc=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

