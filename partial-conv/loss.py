import torch
import torch.nn as nn

class inpaintingtLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mask, output, gt, epsilon):
        
        ## split
        output_current_hour = output[:,:,-1,:,:]
        gt_current_hour = gt[:,:,-1,:,:]
        mask_current_hour = mask[:,:,-1,:,:]
        
        output_prev_hours = output[:,:,:-1,:,:]
        gt_prev_hours = gt[:,:,:-1,:,:]
        mask_prev_hours = mask[:,:,:-1,:,:]
        
        ## l1 loss on current hour
        output_hole = output_current_hour[torch.where(mask_current_hour==0)]
        output_valid = output_current_hour[torch.where(mask_current_hour==1)]
        gt_hole = gt_current_hour[torch.where(mask_current_hour==0)]
        gt_valid = gt_current_hour[torch.where(mask_current_hour==1)]
        l1_loss_current_hole = self.l1(output_hole, gt_hole)
        l1_loss_current_valid = self.l1(output_valid, gt_valid)
        
        ## l1 loss on previous hours
        l1_loss_prev_hours = self.l1(output_prev_hours, gt_prev_hours)
        
        ## relative loss
        diff = torch.abs(output-gt)
        mini = torch.min(output,gt)+epsilon
        relative_loss = torch.mean(diff/mini)
        
        ## when chunk size is 1: no prev_hour_loss
        if gt.size(2) == 1: l1_loss_prev_hours = 0
        
        return {'l1_loss_current_hole':l1_loss_current_hole, 
                'l1_loss_current_valid':l1_loss_current_valid,
                'l1_loss_prev_hours':l1_loss_prev_hours,
                'relative_loss':relative_loss}