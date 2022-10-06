import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import torch
import cv2
import itertools

from PIL import Image
from glob import glob
from torch.utils import data
from tqdm import tqdm    
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from cv2 import PSNR
from net import *
from utils import *
from loss import *
from torch.optim.lr_scheduler import StepLR

## seed
seed = 816
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

############################################################
################## data preparation
############################################################

class taxi_data(torch.utils.data.Dataset):
    
    """
    Prepare taxi data
    """
    
    def __init__(self, 
                 imgs, 
                 masks,
                 img_size, 
                 chunk_size):
        super(taxi_data, self).__init__()
        self.imgs = imgs
        self.masks = masks
        self.img_size = img_size
        self.chunk_size = chunk_size

    def __getitem__(self, index):
        
        ## chunk images & masks
        chunk_imgs = torch.from_numpy(self.imgs[index-self.chunk_size+1:index+1]).float()
        chunk_masks = torch.from_numpy(self.masks[index-self.chunk_size+1:index+1]).float()
        
        ## reshape
        chunk_imgs = chunk_imgs.reshape(1,self.chunk_size,self.img_size,self.img_size)
        chunk_masks = chunk_masks.reshape(1,self.chunk_size,self.img_size,self.img_size)
        
        return chunk_masks, chunk_imgs

    def __len__(self):
        return len(self.imgs)


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples, chunk_size):
        self.num_samples = num_samples
        self.chunk_size = chunk_size

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        order = np.array(np.random.permutation(self.num_samples))
        order = order[order>=self.chunk_size-1]
        self.num_samples_small = len(order)
        i = 0
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples_small:
                np.random.seed()
                order = np.array(np.random.permutation(self.num_samples))
                order = order[order>=self.chunk_size-1]
                self.num_samples_small = len(order)
                i = 0

############################################################
############# Load Data ####################################
############################################################

def load_data(parameters):
    
    """
    Load data for training, validation & test, given specific chunk size and mask type
    
    Arg:
        - parameters: dictionary containing different parameter values
    
    Output:
        - training set
        - validation set
        - test set with biased masking
        - test set with random masking
    
    """
    
    ## parameters
    img_root = parameters['img_root']
    mask_root = parameters['mask_root']
    chunk_size = parameters['chunk_size']
    mask_type = parameters['mask_type']
    image_size = parameters['image_size']
    batch_size = parameters['batch_size']
    
    ## load images
    train_imgs = np.load(img_root+'/train.npy')
    val_imgs = np.load(img_root+'/val.npy')
    test_imgs = np.load(img_root+'/test.npy')
    
    ## load masks
    train_masks = np.load(mask_root+f'/train_{mask_type}_mask.npy')
    val_masks = np.load(mask_root+f'/val_{mask_type}_mask.npy')
    test_masks_biased = np.load(mask_root+'/test_biased_mask.npy')
    test_masks_random = np.load(mask_root+'/test_random_mask.npy')
    
    ## prep datasets
    dataset_train = taxi_data(train_imgs, train_masks, image_size, chunk_size)
    dataset_train = iter(data.DataLoader(dataset_train, 
                                         batch_size=batch_size, 
                                         sampler=InfiniteSampler(len(dataset_train), chunk_size)))
    dataset_val = taxi_data(val_imgs, val_masks, image_size, chunk_size)
    dataset_test_biased = taxi_data(test_imgs, test_masks_biased, image_size, chunk_size)
    dataset_test_random = taxi_data(test_imgs, test_masks_random, image_size, chunk_size)
    
    return dataset_train, dataset_val, dataset_test_biased, dataset_test_random

############################################################
################## model training & evaluation
############################################################                

def train(dataset_train, 
          dataset_val, 
          dataset_test_biased, 
          dataset_test_random, 
          parameters):
    
    """
    Function to train partial convolution inpainting model.
    
    Arg: 
        - dataset_train: training set
        - dataset_val: validation set
        - dataset_test_biased: test set with biased masking
        - dataset_test_ranom: test set with random masking
        - parameters: all parameters
        - all_results: results storage
        
    """
    ## seed
    seed = 816
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ## laod weight values
    valid_loss_weight = parameters['valid_loss_weight']
    hole_loss_weight = parameters['hole_loss_weight']
    mape_losses = parameters['mape_losses']
    chunk_size = parameters['chunk_size']
    mask_type = parameters['mask_type']
    learning_rate = parameters['learning_rate']
    learning_rate_step = parameters['learning_rate_step']
    eval_iter = parameters['eval_iter']
    batch_size = parameters['batch_size']
    device = parameters['device']
    max_iter = parameters['max_iter']
    
    ## different mape losses
    for mape_loss in mape_losses:
        
        ## mape loss progress
        if mape_loss == 0:
            print('   --- No Mape Loss...')
        else:
            print(f'   --- With Mape Loss')
        
        ## initialization
        parameters['mape_loss'] = mape_loss
        model = PConvUNet(chunk_size=chunk_size).to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        criterion = inpaintingtLoss(mape_loss).to(device)
        scheduler = StepLR(optimizer, learning_rate_step, 0.9)
        val_model_loss = []
        
        ## training
        for ite in tqdm(range(max_iter)):
            
            ## training mode
            model.train()
                
            ## training
            mask, gt = [x.to(device) for x in next(dataset_train)]
            output, _ = model(gt, mask)
            loss_dict = criterion(mask, output, gt)
            loss = hole_loss_weight*(loss_dict['l1_loss_hole'])+\
                   valid_loss_weight*(loss_dict['l1_loss_valid'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            ## validation
            if (ite + 1) % eval_iter == 0 or (ite + 1) == max_iter:
                ## evaluation mode
                model.eval()
                val_model_loss.append(loss_eval(model, criterion, dataset_val, parameters))
            
        ## test results
        test_metrics_biased = quantitative_eval(model, dataset_test_biased, parameters)
        test_metrics_random = quantitative_eval(model, dataset_test_random, parameters)
        biased_psnr = np.array(test_metrics_biased['psnr_output'])
        random_psnr = np.array(test_metrics_random['psnr_output'])
        
        ## model configuration
        model_results = [mask_type, chunk_size, mape_loss,
                         np.mean(test_metrics_biased['hole_l1_output']), 
                         np.mean(test_metrics_biased['hole_mse_output']), 
                         np.mean(test_metrics_biased['ssim_output_5']), 
                         np.mean(biased_psnr[~np.isinf(biased_psnr)]), 
                         np.mean(test_metrics_random['hole_l1_output']), 
                         np.mean(test_metrics_random['hole_mse_output']), 
                         np.mean(test_metrics_random['ssim_output_5']), 
                         np.mean(random_psnr[~np.isinf(random_psnr)])]
            
        ## save configuration
        np.save(f'results/chunk_size_{chunk_size}/{mask_type}_{mape_loss}.npy', model_results)
        torch.save(model.state_dict(), f'model_states/chunk_size_{chunk_size}/{mask_type}_{mape_loss}')
        np.save(f'logs/chunk_size_{chunk_size}/{mask_type}_{mape_loss}.npy', val_model_loss)


def loss_eval(model, criterion, dataset, parameters):
    
    ## parameters
    batch_size = parameters['batch_size']
    chunk_size = parameters['chunk_size']
    device = parameters['device']
    model_loss_total = 0
    valid_loss_weight = parameters['valid_loss_weight']
    hole_loss_weight = parameters['hole_loss_weight']
    
    ## iterate through
    all_gt = []
    all_output = []
    all_mask = []
    outer_indices = [k for k in range(chunk_size-1, len(dataset), chunk_size)]
    for i in range(0, len(outer_indices), batch_size):
        inner_indices = range(i, min(len(outer_indices), i+batch_size))
        mask, gt= zip(*[dataset[outer_indices[j]] for j in inner_indices])
        mask = torch.stack(mask).to(device)
        gt = torch.stack(gt).to(device)
        with torch.no_grad():
            output, _ = model(gt, mask)
        all_gt.append(gt)
        all_output.append(output)
        all_mask.append(mask)
    all_gt = torch.cat(all_gt)
    all_output = torch.cat(all_output)
    all_mask = torch.cat(all_mask)
    loss_dict = criterion(all_mask, all_output, all_gt)
    model_loss = hole_loss_weight*loss_dict['l1_loss_hole']+\
                 valid_loss_weight*loss_dict['l1_loss_valid']
    return model_loss.item()

def quantitative_eval(model, dataset, parameters):
    
    ## parameters
    device = parameters['device']
    batch_size = parameters['batch_size']
    chunk_size = parameters['chunk_size']
    all_time_max = parameters['all_time_max']
    
    ## metrics
    valid_l1_output = []
    hole_l1_output = []
    hole_mse_output = []
    ssim_output_5 = []
    psnr_output = []
    
    ## loss
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    ## iterate through
    outer_indices = [k for k in range(chunk_size-1, len(dataset), chunk_size)]
    for i in range(0, len(outer_indices), batch_size):
        inner_indices = range(i, min(len(outer_indices), i+batch_size))
        mask, gt= zip(*[dataset[outer_indices[j]] for j in inner_indices])
        mask = torch.stack(mask).to(device)
        gt = torch.stack(gt).to(device)
        with torch.no_grad():
            output, _ = model(gt, mask)
        output_comp = mask * gt + (1 - mask) * output
        
        ## scale back
        gt = (gt*all_time_max).squeeze_(1).cpu().detach().numpy()
        output_comp = (output_comp*all_time_max).squeeze_(1).cpu().detach().numpy()
        output = (output*all_time_max).squeeze_(1).cpu().detach().numpy()
        mask = mask.squeeze_(1).cpu().detach().numpy()        
        
        for j in range(len(output)):
            for k in range(chunk_size):
                ## single image & output
                gt_single = gt[j][k]
                output_single = output[j][k]
                output_comp_single = output_comp[j][k]
                mask_single = mask[j][k]
                
                ## calculate metrics
                ## overall image
                ssim_output_5.append(structural_similarity(output_comp_single, gt_single, win_size=5, data_range=all_time_max))
                psnr_output.append(peak_signal_noise_ratio(output_comp_single, gt_single, data_range=all_time_max))
                
                ## hole regions
                output_single_hole = output_single[np.where(mask_single == 0)]
                gt_single_hole = gt_single[np.where(mask_single == 0)]
                hole_l1_output.append(np.mean(np.abs(output_single_hole - gt_single_hole)))
                hole_mse_output.append(mean_squared_error(output_single_hole, gt_single_hole))
                
    return {'valid_l1_output':valid_l1_output,
            'hole_l1_output':hole_l1_output,
            'hole_mse_output':hole_mse_output,
            'ssim_output_5':ssim_output_5,
            'psnr_output':psnr_output}
