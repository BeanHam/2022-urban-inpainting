import torch
import torch.nn as nn

class inpaintingtLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mask, output, gt):
        
        ## l1 loss on hole
        l1_loss_hole = self.l1((1-mask)*output, (1-mask)*gt)
        l1_loss_valid = self.l1(mask*output, mask*gt)
        
        ## relative loss
        diff = torch.abs(torch.exp(output)-torch.exp(gt))
        relative_loss = torch.mean(diff/torch.exp(gt))
        
        return {'l1_loss_hole':l1_loss_hole, 
                'l1_loss_valid':l1_loss_valid,
                'relative_loss':relative_loss}