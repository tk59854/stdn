import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.ticker as mtick
from utils.vis_tensor import show_tensor_3d


def test_imgs(dir_path, model=None):
    imgs = os.listdir(dir_path)
    imgs_path = [os.path.join(dir_path, _img) for _img in imgs]
    
    view = dir_path.split('/')[-1]
    condi = dir_path.split('/')[-2]
    
    sils = []
    i=0
    for _img_path in sorted(imgs_path):
        img = cv2.imread(str(_img_path), cv2.IMREAD_GRAYSCALE)
        img = img[..., 10:-10].astype('uint8')/255.
        # if view == '090':
        #     plt.imsave('{}_{}.png'.format(condi, i), img, cmap='gray')
        #     i += 1
        sils.append(img)
    sils = torch.Tensor(np.array(sils)).unsqueeze(0)
    if len(sils.size()) == 4:
        sils = sils.unsqueeze(2)
        
    inputs = sils.cuda()  # [1, s, 1, h, w]
    n, s, c, h, w = inputs.shape
    
    print(torch.mean(inputs.view(n, s, c, -1), dim=-1).shape)
    
    inputs = ((inputs - torch.mean(inputs.view(n, s, c, -1), dim=-1).unsqueeze(-1).unsqueeze(-1)).view(n, s, c, h, w) ** 2) / (h * w)
    
    for i, _img in enumerate(inputs.squeeze(0)):
        print(_img.shape)
        _img = _img.clone().detach().cpu().squeeze(0).type(torch.float32).numpy()
        plt.imsave('var_{}.png'.format(i), _img, cmap='gray')
    
    # print(inputs.shape)
    
    # inputs = [[sils], None, None, None, None]
    
    # print(model.Backbone)
    
    out = model.Backbone.stages[0](inputs)
    
    print(out.shape)
    
    show_tensor_3d(out.permute(0, 2, 1, 3, 4), num_seqs=5, num_channel=1, save_name='bg_{}'.format(view), channel_avg=True)
    
    # print(model)
    
    
    
    # # att vis 
    
    # spatial_f, temporal_f = model.Backbone(inputs)
    # # # # spatial_f = model.Backbone(sils)
    # # print(spatial_f.shape) # ([1, 5, 256, 32, 22])
    # # print(temporal_f.shape) # ([1, 5, 256, 32])
    
    # # temporal_data = torch.mean(temporal_f[0, 0, :, :], 0, keepdim=True) # (1, 32)
    
    # temporal_data = torch.mean(temporal_f[0, :, :, :], 1, keepdim=True) # (5, 1, 32)
    # temporal_data = torch.mean(temporal_data, 0, keepdim=True).squeeze(0)  # (1, 32)
    
    
    
    # temporal_data = temporal_data.unsqueeze(0)  # (1, 1, 32)
    
    # temporal_data = F.interpolate(temporal_data, (64), mode='linear') # (1, 1, 64)
    # temporal_data = temporal_data.squeeze(0).squeeze(0).unsqueeze(1).repeat(1, 44)  # (64, 44)

    # temporal_weight = F.sigmoid(temporal_data)
    
    # # temporal_weight = temporal_weight.squeeze(0).squeeze(0).detach().cpu().numpy()
    
    # temporal_weight = temporal_weight.detach().cpu().numpy()  # (64, 44)
    
    # gei = torch.mean(sils, 1).squeeze(0).squeeze(0)
    
    # weighted_gei = (gei) * (temporal_weight)
    
    # plt.imsave('{}_weighted_gei_{}.png'.format(condi, view), weighted_gei, cmap='magma')
    # # print(temporal_weight)
    
    
    