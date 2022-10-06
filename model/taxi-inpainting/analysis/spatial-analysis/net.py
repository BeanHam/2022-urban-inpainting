import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import Parameter

######################################################
############# Partial Convolution
######################################################

## seed
seed = 816
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(init_type='gaussian'):
    
    def init_fun(m):
        
        ## seed
        seed = 816
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'uniform':
                nn.init.uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun

class PartialConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        
        ## seed
        seed = 816
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        super().__init__()
        self.input_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.input_conv.apply(weights_init('kaiming'))
        self.mask_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)
            
        ## output and update masks
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, chunk_size=1, bn=True, sample='none-3', activ='relu',conv_bias=True):
        super().__init__()
        if sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, (1,7,7), (1,2,2), (0,3,3), bias=conv_bias)
        elif sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, (1,5,5), (1,2,2), (0,2,2), bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, (1,3,3), (1,2,2), (0,1,1), bias=conv_bias)
        elif sample == 'down-3-3d':
            self.conv = PartialConv(in_ch, out_ch, (chunk_size, 3,3), (2,2,2), (2*((chunk_size-1)//4)+1,1,1), bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, (1,3,3), (1,1,1), (0,1,1), bias=conv_bias)
        if bn:
            self.bn = nn.BatchNorm3d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class PConvUNet(nn.Module):
    def __init__(self, chunk_size, layer_size=6, input_channels=1, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.chunk_size = chunk_size
        
        ## encoders
        self.enc_1 = PCBActiv(input_channels, 64, sample='down-3',conv_bias=True)
        self.enc_2 = PCBActiv(64, 128, sample='down-3',conv_bias=True)
        self.enc_3 = PCBActiv(128, 256, sample='down-3',conv_bias=True)
        self.enc_4 = PCBActiv(256, 512, sample='down-3',conv_bias=True)
        self.enc_5 = PCBActiv(512, 512, chunk_size, sample='down-3-3d',conv_bias=True)
        self.enc_6 = PCBActiv(512, 512, chunk_size, sample='down-3-3d',conv_bias=True)
        
        ## decoders        
        self.dec_6 = PCBActiv(512 + 512, 512, activ='relu',conv_bias=True)
        self.dec_5 = PCBActiv(512 + 512, 512, activ='relu',conv_bias=True)
        self.dec_4 = PCBActiv(512 + 256, 256, activ='relu',conv_bias=True)
        self.dec_3 = PCBActiv(256 + 128, 128, activ='relu',conv_bias=True)
        self.dec_2 = PCBActiv(128 + 64, 64, activ='relu',conv_bias=True)
        self.dec_1 = PCBActiv(64 + input_channels, input_channels, bn=False, activ='relu', conv_bias=True)

    def forward(self, input, input_mask):
        ## encoder
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the mask output of enc_N
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask
        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key
        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]
        
        ## concatenation
        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            enc_h_out = h_dict[enc_h_key]
            dec_l_key = 'dec_{:d}'.format(i)
            h = F.interpolate(h, size=(enc_h_out.size(2),enc_h_out.size(3),enc_h_out.size(4)), mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, size=(enc_h_out.size(2),enc_h_out.size(3),enc_h_out.size(4)), mode=self.upsampling_mode)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
        return h, h_mask
    
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()