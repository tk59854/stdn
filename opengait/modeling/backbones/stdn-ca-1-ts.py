import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class DPFE(nn.Module):  
    def __init__(self, spatial_conv, dim, height, kernel_size=3, padding=1, last=False, squeeze=4):
        super(DPFE,  self).__init__()

        self.last = last
        self.spatial_conv = spatial_conv
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(dim, dim//squeeze, kernel_size=kernel_size, padding=padding, bias=False), 
            # nn.BatchNorm1d(dim//squeeze),
            nn.LeakyReLU(inplace=True), 
            nn.Conv1d(dim//squeeze, dim, kernel_size=kernel_size, padding=padding, bias=False),   
        )
    
    def forward(self, x):
        # x[n, s, c, p, w]
        n, s, c, p, w = x.shape

        identity = x

        spatial_f = self.spatial_conv(x.view(-1, c, p, w)).view(n, s, -1, p, w)

        temporal_f = x.max(-1)[0].permute(0, 3, 2, 1).contiguous()  # [n, p, c, s]
        temporal_f = self.temporal_conv(temporal_f.view(-1, c, s)).view(n, p, c, s) + temporal_f
        temporal_f = temporal_f.permute(0, 3, 2, 1).contiguous()  # [n, s, c, p]

        temporal_weight = torch.sigmoid(temporal_f).unsqueeze(-1)
        weighted_f = spatial_f * temporal_weight + identity
        
        if not self.last:
            return weighted_f
        else:
            return weighted_f, temporal_f


class ESA(nn.Module):
    def __init__(self, reso_h, dim, kernel_size=3, padding=1, last=False, **kwargs):
        super().__init__()

        if not last:
            starts_ratiao = torch.Tensor([0, 1, 1, 1, 1])
        else:
            starts_ratiao = torch.Tensor([0, 1, 1, 1, 1, 1, 1, 1, 1]) 
        starts = [int((reso_h/starts_ratiao.sum())*_ratiao) for _ratiao in starts_ratiao.cumsum(axis=0)]
        self.part_l = [starts[i+1] - starts[i] for i in range(len(starts)-1)]

        global_conv = nn.Sequential(
            nn.Identity())
        self.global_pconv = DPFE(global_conv, dim, reso_h, last=last)

        part_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, bias=False, padding=padding, **kwargs), 
            nn.LeakyReLU(inplace=True),)
        self.part_pconv = nn.ModuleList(
            [DPFE(part_conv, dim, _l, last=last) for _l in self.part_l])

        self.last = last

    def forward(self, x):
        # x [n, s, c, h, w]
        
        x_split = torch.split(x, self.part_l, dim=-2)
        
        if not self.last:
            global_f = self.global_pconv(x)
            part_f = torch.cat([_p_conv(_f) for _p_conv, _f in zip(self.part_pconv, x_split)], -2)  # [n, s, c, h, w]
            return global_f + part_f
        else: 
            global_f, global_temporal_f = self.global_pconv(x)

            part_outs = [_p_conv(_f) for _p_conv, _f in zip(self.part_pconv, x_split)]
            part_f, part_temporal_f = [], []
            for _outs in part_outs:
                part_f.append(_outs[0])
                part_temporal_f.append(_outs[1])
            part_f = torch.cat(part_f, -2)
            part_temporal_f = torch.cat(part_temporal_f, -1)

            spatial_f = part_f + global_f  # [n, s, c, h, w]
            temporal_f = part_temporal_f + global_temporal_f  # [n, s, c, h]
            
            return spatial_f, temporal_f


class LinkLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, is_stem=False):
        super().__init__()
        self.is_stem = is_stem
        if is_stem:
            self.link_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.LeakyReLU(inplace=True),
            )
            self.temporal_select = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        elif stride==2:
            self.link_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2), stride=stride),
            )
        else:
            self.link_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.LeakyReLU(inplace=True),
            )
 
    def forward(self, x):
        if self.is_stem:
            n, s, c, h, w = x.shape
            x = self.link_layer(x.view(-1, c, h, w))
            _, c, h, w = x.shape
            x = x.view(n, s, c, h, w)
            x = self.temporal_select(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous().view(n, s//3, c, h, w)
        else:
            n, s, c, h, w = x.shape
            x = self.link_layer(x.view(-1, c, h, w))
            _, c, h, w = x.shape
            x = x.view(n, s, c, h, w)
        return x


class STDN1TS(nn.Module):
    def __init__(self, c_in=1, depths=[1, 1, 1], dims=[64, 128, 256], downsample_stride=[1, 2, 1]):
        super().__init__()

        self.height = 64
        self.num_stages = len(depths)

        self.stages = nn.ModuleList() 

        dims.insert(0, c_in)
        for i in range(self.num_stages):
            self.height = self.height // downsample_stride[i]
            
            blocks = []
            for j in range(depths[i]):
                last = ((i == self.num_stages - 1) and (j == depths[i]-1))
                blocks.append(ESA(reso_h=self.height, dim=dims[i+1], kernel_size=3, padding=1, last=last))

            self.stages.append(nn.Sequential(
                LinkLayer(dims[i], dims[i+1], stride=downsample_stride[i], is_stem=(i==0)),
                *blocks
            ))

    def forward_features(self, x):
        for i in range(self.num_stages):
            x = self.stages[i](x)
        return x

    def forward(self, x):
        out = self.forward_features(x)
        return out


if __name__ == '__main__':
    import time

    model = STDN1TS(depths=[1, 1, 1], dims=[64, 128, 256], downsample_stride=[1, 2, 1])
    x = torch.randn((1, 30, 1, 64, 44))
    
    print(model)
    t = time.time()
    spacial, temporal = model(x)
    print(time.time() - t)
    print(spacial.shape)
    print(temporal.shape)