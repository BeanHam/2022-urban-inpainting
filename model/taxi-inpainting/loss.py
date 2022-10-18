import torch
import torch.nn as nn

## seed
seed = 816
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class inpaintingtLoss(nn.Module):
    def __init__(self, mape_loss):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mape_loss = mape_loss

    def forward(self, mask, output, gt):
        
        if self.mape_loss:
            diff = torch.abs(output-gt)
            mape = diff/(torch.abs(gt) + torch.abs(output))
            mape[torch.isnan(mape)]=0
            mape[torch.isinf(mape)]=0
            l1_loss_hole = torch.mean((1-mask)*diff*(1+mape))
            l1_loss_valid = torch.mean(mask*diff*(1+mape))
        else:
            ## l1 loss on hole
            hole_output = output[torch.where(mask==0)]
            hole_gt = gt[torch.where(mask==0)]
            valid_output = output[torch.where(mask==1)]
            valid_gt = gt[torch.where(mask==1)]
            l1_loss_hole = self.l1((1-mask)*output, (1-mask)*gt)
            l1_loss_valid = self.l1((1-mask)*output, (1-mask)*gt)
            
            #l1_loss_hole = self.l1((1-mask)*output, (1-mask)*gt)
            #l1_loss_valid = self.l1(mask*output, mask*gt)
        
        return {'l1_loss_hole':l1_loss_hole, 
                'l1_loss_valid':l1_loss_valid}