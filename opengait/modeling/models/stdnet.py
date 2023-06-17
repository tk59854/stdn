import torch
import torch.nn as nn

import math

import math

from ..base_model import BaseModel
from ..modules import SeparateFCs, PackSequenceWrapper


class STDNet(BaseModel): 
    def __init__(self, *args, **kargs):
        super(STDNet, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):  

        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.msg_mgr.log_info('---- BACKBONE ----')
        self.msg_mgr.log_info(self.Backbone)
        self.TP = PackSequenceWrapper(torch.max)
        self.head1 = SeparateFCs(**model_cfg['SeparateFCs_1'])
        self.BN = nn.BatchNorm1d(model_cfg['SeparateFCs_1']['out_channels'] * model_cfg['SeparateFCs_1']['parts_num'])
        self.head2 = SeparateFCs(**model_cfg['SeparateFCs_2'])


    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2)
        del ipts

        seqL = None if not self.training else seqL

        s = sils.shape[1]
        if s < 3:
            repeat = math.ceil(3/s)
            sils = sils.repeat(1, repeat, 1, 1, 1)
        
        x_s, x_t = self.Backbone(sils)  # [n, s, c, h, w]
        n, s, c, h, w = x_s.shape

        # Temporal Pooling
        spacial_f = self.TP(x_s, seqL, dim=1)[0]  # [n, c, h, w]
        temporal_f = self.TP(x_t, seqL, dim=1)[0]  # [n, c, h]

        # HPM
        spacial_f = spacial_f.max(-1)[0] + spacial_f.mean(-1)  # [n, c, h]

        feat = torch.cat((spacial_f, temporal_f), -1)
        feat = feat.permute(2, 0, 1).contiguous()  # [p, n, c]

        embed = self.head1(feat)  # [p, n, c]
        p, n, c = embed.shape

        bnft = self.BN(embed.permute(1, 0, 2).contiguous().view(n, p*c)) # [n, p, c]
        bnft = bnft.view(n, p, c).permute(1, 0, 2).contiguous()  # [p, n, c]
        logits = self.head2(bnft)

        embed = embed.permute(1, 0, 2).contiguous()  # [n, p, c]
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p, c]
        embed = embed

        num_visi = 1 if not self.training else 40
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {'image/x_s': x_s.view(-1, c, h, w)[:num_visi, 0:1, :, :],
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval