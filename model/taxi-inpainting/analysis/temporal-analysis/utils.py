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

############################################################
################## data preparation
############################################################

class taxi_data(torch.utils.data.Dataset):
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
################## model training & evaluation
############################################################                

def train_l1(dataset_train, dataset_val, dataset_test_biased, dataset_test_random, parameters, all_results):
    
    ## weight values values
    valid_loss_weight = parameters['valid_loss_weight']
    prev_hours_loss_weight = parameters['prev_hours_loss_weight']
    hole_loss_weights = parameters['hole_loss_weights']
    parameters['relative_loss_weight'] = 0
    
    ## parameter values
    max_iter = parameters['max_iter']
    learning_rate_m = parameters['learning_rate_m']
    learning_rate_step = parameters['learning_rate_step']
    eval_iter = parameters['eval_iter']
    batch_size = parameters['batch_size']
    chunk_size = parameters['chunk_size']
    device = parameters['device']
    all_time_max = parameters['all_time_max']
    
    ## initialize best model dict
    best_model = {
        'relative_loss_weight': 0,
        'best_val_perf': np.inf
    }
    
    counter = 1
    for hole_loss_weight in hole_loss_weights:
    
        parameters['hole_loss_weight'] = hole_loss_weight
        print(f'[{counter}/{len(hole_loss_weights)}]...')
        
        ## initialization
        model = PConvUNet(chunk_size=chunk_size).to(device)
        optimizer_m = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate_m)
        criterion_m = inpaintingtLoss().to(device)
        scheduler_m = StepLR(optimizer_m, learning_rate_step, 0.9)
        val_model_loss = []
        
        ## training iterations
        for i in tqdm(range(max_iter)):
            
            ## training mode
            model.train()
            
            ## training
            mask, gt = [x.to(device) for x in next(dataset_train)]
            output, _ = model(gt, mask)
            loss_dict = criterion_m(mask, output, gt)
            loss = hole_loss_weight*(loss_dict['l1_loss_hole'])+\
                   valid_loss_weight*(loss_dict['l1_loss_valid'])
            optimizer_m.zero_grad()
            loss.backward()
            optimizer_m.step()
            scheduler_m.step()
            
            ## validation
            if (i + 1) % eval_iter == 0 or (i + 1) == max_iter:
                ## evaluation mode
                model.eval()
                val_model_loss.append(loss_eval_all(model, criterion_m, dataset_val, parameters))
        
        ## validation            
        final_val_loss = np.mean(quantitative_eval(model, dataset_val, parameters)['hole_l1_output'])
        
        ## check with best performing model so far
        if final_val_loss < best_model['best_val_perf']:
            best_model['hole_loss_weight'] = hole_loss_weight
            best_model['best_val_perf'] = final_val_loss
            best_model['val_model_loss'] = val_model_loss
            best_model['model_state'] = model.state_dict()
        counter += 1
        
    ## evaluate on test set
    model = PConvUNet(chunk_size=chunk_size).to(device)
    model.load_state_dict(best_model['model_state'])    
    test_metrics_biased = quantitative_eval(model, dataset_test_biased, parameters)
    test_metrics_random = quantitative_eval(model, dataset_test_random, parameters)
    best_model['test_metrics_biased'] = test_metrics_biased
    best_model['test_metrics_random'] = test_metrics_random
    
    return best_model


############################################################
################## training procedure with relative loss
############################################################   

def train_relative(dataset_train, dataset_val, dataset_test_biased, dataset_test_random, parameters, all_results):
    
    
    ## weight values values
    valid_loss_weight = parameters['valid_loss_weight']
    prev_hours_loss_weight = parameters['prev_hours_loss_weight']
    hole_loss_weights = parameters['hole_loss_weights']
    relative_loss_weights = parameters['relative_loss_weights']
    
    ## parameter values
    max_iter = parameters['max_iter']
    learning_rate_m = parameters['learning_rate_m']
    learning_rate_step = parameters['learning_rate_step']
    eval_iter = parameters['eval_iter']
    batch_size = parameters['batch_size']
    chunk_size = parameters['chunk_size']
    device = parameters['device']
    all_time_max = parameters['all_time_max']
    
    ## initialize best model dict
    best_model = {
        'best_val_perf': np.inf
    }
    
    counter = 1
    
    for hole_loss_weight, relative_loss_weight in itertools.product(hole_loss_weights, relative_loss_weights):
        
        print(f'[{counter}/{len(hole_loss_weights)*len(relative_loss_weights)}]...')
        parameters['hole_loss_weight'] = hole_loss_weight
        parameters['relative_loss_weight'] = relative_loss_weight
        
        ## initialization
        model = PConvUNet(chunk_size=chunk_size).to(device)
        optimizer_m = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate_m)
        criterion_m = inpaintingtLoss().to(device)
        scheduler_m = StepLR(optimizer_m, learning_rate_step, 0.9)
        val_model_loss = []
        
        ## training iterations
        for i in tqdm(range(max_iter)):
        
            ## training mode
            model.train()
            
            ## training
            mask, gt = [x.to(device) for x in next(dataset_train)]
            output, _ = model(gt, mask)
            loss_dict = criterion_m(mask, output, gt)
            loss = hole_loss_weight*(loss_dict['l1_loss_hole'])+\
                   valid_loss_weight*(loss_dict['l1_loss_valid'])+\
                   relative_loss_weight*loss_dict['relative_loss']            
            optimizer_m.zero_grad()
            loss.backward()
            optimizer_m.step()
            scheduler_m.step()
            
            ## validation
            if (i + 1) % eval_iter == 0 or (i + 1) == max_iter:
                ## evaluation mode
                model.eval()
                val_model_loss.append(loss_eval_all(model, criterion_m, dataset_val, parameters))
        
        ## validation            
        final_val_loss = np.mean(quantitative_eval(model, dataset_val, parameters)['hole_l1_output'])
        
        ## check with best performing model so far
        if final_val_loss < best_model['best_val_perf']:
            best_model['hole_loss_weight'] = hole_loss_weight
            best_model['relative_loss_weight'] = relative_loss_weight
            best_model['best_val_perf'] = final_val_loss
            best_model['val_model_loss'] = val_model_loss
            best_model['model_state'] = model.state_dict()
        counter += 1
        
    ## check the best model on test dataset
    model = PConvUNet(chunk_size=chunk_size).to(device)
    model.load_state_dict(best_model['model_state'])
    test_metrics_biased = quantitative_eval(model, dataset_test_biased, parameters)
    test_metrics_random = quantitative_eval(model, dataset_test_random, parameters)
    best_model['test_metrics_biased'] = test_metrics_biased
    best_model['test_metrics_random'] = test_metrics_random
    
    return best_model

def loss_eval_all(model, criterion_m, dataset, parameters):
    
    ## parameters
    batch_size = parameters['batch_size']
    chunk_size = parameters['chunk_size']
    hole_loss_weight = parameters['hole_loss_weight']
    valid_loss_weight = parameters['valid_loss_weight']
    prev_hour_loss_weight = parameters['prev_hours_loss_weight']
    relative_loss_weight = parameters['relative_loss_weight']
    device = parameters['device']
    model_loss_total = 0
    
    ## iterate through
    #permutation = torch.randperm(len(dataset))
    #permutation = permutation[permutation>=chunk_size-1]
    #for i in range(0, len(permutation), batch_size):
    #    indices = permutation[i:i+batch_size]
    for i in range(chunk_size-1, len(dataset), batch_size):
        indices = range(i, min(len(dataset), i+batch_size))        
        mask, gt= zip(*[dataset[i] for i in indices])
        mask = torch.stack(mask).to(device)
        gt = torch.stack(gt).to(device)
        with torch.no_grad():
            output, _ = model(gt, mask)
        loss_dict = criterion_m(mask, output, gt)
        model_loss_total += hole_loss_weight*(loss_dict['l1_loss_hole'])+\
                            valid_loss_weight*(loss_dict['l1_loss_valid'])+\
                            relative_loss_weight*(loss_dict['relative_loss'])
    
    return (model_loss_total/len(dataset)).item()

def partial_mse(output_img, gt_img, mask):
    non_zero_gt_imags = gt_img[gt_img*(1-mask)!=0]
    non_zero_output_imags = output_img[gt_img*(1-mask)!=0]
    return np.mean((non_zero_gt_imags-non_zero_output_imags)**2)

def quantitative_eval(model, dataset, parameters):
    
    ## parameters
    device = parameters['device']
    batch_size = parameters['batch_size']
    chunk_size = parameters['chunk_size']
    all_time_max = parameters['all_time_max']
    
    ## metrics
    hole_l1_output = []
    hole_mse_output = []
    ssim_output_5 = []
    psnr_output = []

    ## iterate through
    for i in range(chunk_size-1, len(dataset), batch_size):
        indices = range(i, min(len(dataset), i+batch_size))
        mask, gt = zip(*[dataset[i] for i in indices])
        mask = torch.stack(mask).to(device)
        gt = torch.stack(gt).to(device)
        with torch.no_grad():
            output, _ = model(gt, mask)
        output_comp = mask * gt + (1 - mask) * output
        
        ## scale back
        gt = (gt*all_time_max).cpu().detach().numpy()
        output_comp = (output_comp*all_time_max).cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()       
        
        ## take the last layer
        gt = gt[:,:,-1,:,:]
        output_comp = output_comp[:,:,-1,:,:]
        mask = mask[:,:,-1,:,:]
        
        ## calculate ssim & partial mse
        for j in range(len(output)):
            ## single image & output
            gt_single = gt[j][0]
            output_comp_single = output_comp[j][0]
            mask_single = mask[j][0]
            
            ## calculate metrics
            ## overall image
            ssim_output_5.append(structural_similarity(output_comp_single, gt_single, win_size=5, data_range=all_time_max))
            psnr_output.append(peak_signal_noise_ratio(output_comp_single, gt_single, data_range=all_time_max))
            
            ## hole regions
            output_comp_single_hole = output_comp_single[np.where(mask_single == 0)]
            gt_single_hole = gt_single[np.where(mask_single == 0)]
            hole_l1_output.append(np.mean(np.abs(output_comp_single_hole - gt_single_hole)))
            hole_mse_output.append(mean_squared_error(output_comp_single_hole, gt_single_hole))

    return {'hole_l1_output':hole_l1_output,
            'hole_mse_output':hole_mse_output,
            'ssim_output_5':ssim_output_5,
            'psnr_output':psnr_output}

def process_results(results, chunk_size):
    
    biased_tabular_view = []
    random_tabular_view = []
    
    for result in results:
        
        ## biase test set evaluation
        biased_psnr = np.array(result['test_metrics_biased']['psnr_output'])
        biased_tabular_view.append([
            result['mask_type'], 
            result['loss_type'],
            result['hole_loss_weight'],
            result['relative_loss_weight'],
            np.mean(result['test_metrics_biased']['hole_l1_output']),
            np.mean(result['test_metrics_biased']['hole_mse_output']),
            np.mean(result['test_metrics_biased']['ssim_output_5']),
            np.mean(biased_psnr[~np.isinf(biased_psnr)])
        ])
        
        ## random test set evaluation
        random_psnr = np.array(result['test_metrics_random']['psnr_output'])
        random_tabular_view.append([
            result['mask_type'], 
            result['loss_type'],
            result['hole_loss_weight'],        
            result['relative_loss_weight'],
            np.mean(result['test_metrics_random']['hole_l1_output']),
            np.mean(result['test_metrics_random']['hole_mse_output']),
            np.mean(result['test_metrics_random']['ssim_output_5']),
            np.mean(random_psnr[~np.isinf(random_psnr)])
        ])    
        ## save results
        torch.save(result['model_state'], 'model_states/'+result['mask_type']+'_'+result['loss_type']+'_'+str(chunk_size))
        np.save('logs/'+result['mask_type']+'_'+result['loss_type']+'_model'+'_'+str(chunk_size), result['val_model_loss'])
        
    ## make tabular view
    biased_tabular_view = pd.DataFrame(biased_tabular_view, 
                                       columns=['mask_type', 'loss', 'hole_loss_weight',
                                                'relative_loss_weight', 'hole_l1_output', 
                                                'hole_mse_output', 'ssim_5', 'psnr'])
    random_tabular_view = pd.DataFrame(random_tabular_view, 
                                       columns=['mask_type', 'loss', 'hole_loss_weight',
                                                'relative_loss_weight', 'hole_l1_output', 
                                                'hole_mse_output', 'ssim_5', 'psnr'])
    
    ## save tables
    biased_tabular_view.to_csv(f'results/biased_{chunk_size}.csv', index=False)
    random_tabular_view.to_csv(f'results/random_{chunk_size}.csv', index=False)
    