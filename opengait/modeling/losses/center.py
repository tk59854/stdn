import torch
from torch import nn

from modeling.losses.base import BaseLoss, gather_and_scale_wrapper


class CenterLoss(BaseLoss):  # BaseLoss
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        num_part: seperate centerloss for each part
        aplha: [0, 1], scale the gradient

    """

    def __init__(self, num_classes=76, feat_dim=512, num_part=5, alpha=0.6, loss_term_weight=5.e-4):
        super(CenterLoss, self).__init__(loss_term_weight)  # 
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.num_part = num_part
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_part, num_classes, feat_dim)).float()
        self.loss_term_weight = loss_term_weight

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        """
            embeddings: [n, p, c]
            labels: [n]
        """
        assert embeddings.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        device = embeddings.device

        batch_size = embeddings.size(0)
        embeddings = embeddings.permute(
            1, 0, 2).contiguous()  # [n, p, c] -> [p, n, c]
        embeddings = embeddings.float()

        xx = torch.pow(embeddings, 2).sum(dim=-1, keepdim=True)  # [p, n, 1]
        cc = torch.pow(self.centers, 2).sum(dim=-1).unsqueeze(1) # [p, 1, N]
        distmat = xx + cc - 2*(torch.bmm(embeddings, self.centers.permute(0, 2, 1).contiguous()))  # [p, n, N]


        classes = torch.arange(self.num_classes).long().to(device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes)).unsqueeze(0)

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / (batch_size * self.num_part * 2 * self.alpha)

        self.info.update({'loss': loss.detach().clone()})

        return loss, self.info


if __name__ == '__main__':

    torch.manual_seed(1)

    center_loss = CenterLoss()
    features = torch.rand(4, 5, 512)
    targets = torch.Tensor([0, 1, 2, 3]).long()

    loss = center_loss(features, targets)
    print(loss)
