import torch
import torch.nn as nn

class InpaintingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mask, output, gt):
        loss_hole = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_valid = self.l1(mask * output, mask * gt)
        return loss_hole*6.0 + loss_valid*1.0
    
class MarginalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, gt, beta, lamb, offset=1e-5):
        diff = torch.abs(output-gt)
        l1 = self.l1(diff)
        l2 = (diff /(np.min((output,gt),axis=0)+offset))**lamb
        l1 = l1/sum(l1)
        l2 = l2/sum(l2)
        return l1*beta+l2*(1-beta)