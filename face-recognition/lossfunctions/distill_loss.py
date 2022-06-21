import torch.nn as nn
import torch
import torch.nn.functional as F
from lossfunctions.RKD import distill_cri

class DistillLoss(nn.Module):
    def __init__(self, args, device):
        super(DistillLoss, self).__init__()
        self.distill_type = args.distill_type
        self.distill_func = args.distill_func
        self.device = device

        # Distillation Function
        if self.distill_func == 'l1':
            self.criterion = F.l1_loss
        elif self.distill_func == 'l1_smooth':
            self.criterion = F.smooth_l1_loss
        elif self.distill_func == 'l2':
            self.criterion = F.mse_loss
        elif self.distill_func == 'cosine':
            self.criterion = self.cosine_loss
        elif self.distill_func == 'both':
            self.criterion = self.both_loss
        elif self.distill_func == 'both_smooth':
            self.criterion = self.both_loss_smooth
        elif self.distill_func == 'triple':
            self.criterion = self.triple_loss
        else:
            raise('Select proper distillation function')

        # mode
        self.args = args


    def cosine_loss(self, l , h):
        l = l.view(l.size(0), -1)
        h = h.view(h.size(0), -1)
        return torch.mean(1.0 - F.cosine_similarity(l, h))


    def both_loss(self, l , h):
        l1_loss = F.l1_loss(l, h)
        corr_loss = self.cosine_loss(l, h)
        return corr_loss * self.alpha + l1_loss * (1-self.alpha)


    def both_loss_smooth(self, l , h):
        l1_loss = F.smooth_l1_loss(l, h)
        corr_loss = self.corr_loss(l, h)
        return corr_loss * self.alpha + l1_loss * (1-self.alpha)


    def forward(self, low_feature, high_feature):
        loss_sum = 0.
        
        # Get the feature or attention value in the forward pass
        for l, h in zip(low_feature, high_feature):
            if self.distill_type == 'self-attention':
                l = torch.max(l, dim=1)[0]
                h = torch.max(h, dim=1)[0]

            d_loss = self.criterion(l, h)
            loss_sum += d_loss
        
        return loss_sum