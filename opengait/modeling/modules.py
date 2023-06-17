import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange
from utils import clones, is_list_or_tuple


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, s, c, h, w]
            Out x: [n, s, ...]
        """
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1, c, h, w), *args, **kwargs)
        input_size = x.size()
        output_size = [n, s] + [*input_size[1:]]
        return x.view(*output_size)
        

class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, seq_dim=1, **kwargs):
        """
            In  seqs: [n, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **kwargs)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(seq_dim, curr_start, curr_seqL)
            # save the memory
            # splited_narrowed_seq = torch.split(narrowed_seq, 256, dim=1)
            # ret = []
            # for seq_to_pooling in splited_narrowed_seq:
            #     ret.append(self.pooling_func(seq_to_pooling, keepdim=True, **kwargs)
            #                [0] if self.is_tuple_result else self.pooling_func(seq_to_pooling, **kwargs))
            rets.append(self.pooling_func(narrowed_seq, **kwargs))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out


class SeparateBNNecks(nn.Module):
    """
        GaitSet: Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.parallel_BN1d:
            p, n, c = x.size()
            x = x.transpose(0, 1).contiguous().view(n, -1)  # [n, p*c]
            x = self.bn1d(x)
            x = x.view(n, p, c).permute(1, 0, 2).contiguous()
        else:
            x = torch.cat([bn(_.squeeze(0)).unsqueeze(0)
                           for _, bn in zip(x.split(1, 0), self.bn1d)], 0)  # [p, n, c]
        if self.norm:
            feature = F.normalize(x, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            feature = x
            logits = feature.matmul(self.fc_bin)
        return feature, logits


class SeparateLNNecks(nn.Module):
    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_LN=False):
        super(SeparateLNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_LN:
            self.LN = nn.LayerNorm(in_channels)
        else:
            self.LN = clones(nn.LayerNorm(in_channels), parts_num)

        self.parallel_LN = parallel_LN

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.parallel_LN:
            x = self.LN(x)
        else:
            x = torch.cat([bn(_.squeeze(0)).unsqueeze(0)
                           for _, bn in zip(x.split(1, 0), self.LN)], 0)  # [p, n, c]
        if self.norm:
            feature = F.normalize(x, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            feature = x
            logits = feature.matmul(self.fc_bin)
        return feature, logits


class FocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False


class PatchPyramidMapping:

    def __init__(self, u=4, v=4, p=6.5, eps=1.0e-6):
        self.u = list(range(1, u+1))
        self.v = list(range(1, v+1))
        self.W = list()
        for _u in self.u:
            for _v in self.v:
                self.W.append((2 ** (_u-1)) * (2 ** (_v-1)))
        self.W = sorted(self.W)

        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c, h, w = x.size()
        features = []
        for _W in self.W:
            if _W <= h:
                z = x.view(n, c, _W, -1)
                z = self.gem(z).squeeze(-1)
                features.append(z)
            else:
                z = x.view(n, c, h, -1)
                vertical_strips = math.ceil(_W/h)
                z = torch.chunk(z, vertical_strips, -1)
                for _z in z:
                    _ = self.gem(_z.reshape(n, c, -1))
                    features.append(_)
        features = torch.cat(features, -1)
        return features


class PartPooling(nn.Module):
    def __init__(self, parts_num):
        super(PartPooling, self).__init__()
        self.parts_num = parts_num

    def forward(self, inputs):
        """
        inputs: [n, c, h, w]
        retrun: [n, parts_num, h*w]
        """
        n, c, h, w = inputs.shape

        assert c % self.parts_num == 0, 'The output channels should be an integer multiple of the parts_num'

        k = c // self.parts_num

        inputs = inputs.view(n, c, -1)
        outs = F.max_pool2d(inputs, kernel_size=(k, 1), stride=(k, 1))
        return outs


class PartSelfAttention(nn.Module):
    def __init__(self, dim, heads=1):
        super(PartSelfAttention, self).__init__()
        self.dim = dim
        self.heads = heads

        self.to_qvk = nn.Linear(dim, heads * dim * 3, bias=False)
        self.scale_factor = dim ** -0.5  # 1/np.sqrt(dim)

    def forward(self, inputs):
        """
        imputs: [b, p, h*w]
        return: [b, p, heads*h*w]
        """
        assert inputs.dim() == 3, '3D tensor must be provided'
        qkv = self.to_qvk(inputs)  # [b, p, dim*3*h ]

        q, k, v = tuple(rearrange(qkv, 'b p (d k h) -> k b h p d ', k=3, h=self.heads))  # [b, h, p, dim] * 3

        dot_prod = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale_factor
        attention = F.softmax(dot_prod, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h i d -> b i (h d)')

        return out


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
               
               
class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)