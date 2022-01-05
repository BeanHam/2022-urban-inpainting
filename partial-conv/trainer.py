import torch
import torch.nn as nn
import itertools
import warnings
import json
from utils import *
warnings.filterwarnings("ignore")


def main():
    
    ################################
    ## load parameters
    ################################
    parameter_path = 'parameters.json'
    with open(parameter_path) as json_file:
        parameters = json.load(json_file)
        
    img_root = parameters['img_root']
    mask_root = parameters['mask_root']
    mask_types = parameters['mask_types']
    loss_types = parameters['loss_types']
    image_size = parameters['image_size']
    chunk_sizes = parameters['chunk_sizes']
    batch_size = parameters['batch_size']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parameters['device'] = device
    
    ################################
    ## training
    ################################
    
    ## prameter-tuning
    for chunk_size in chunk_sizes:
        
        parameters['chunk_size'] = chunk_size
        parameters['max_iter'] = 20000 + 5000*(chunk_size-1)
        
        ## storage 
        all_results = []
        print(f'================================================')
        print(f'Chunk Size: {chunk_size}')
        for mask_type, loss_type in itertools.product(mask_types, loss_types):
            
            #############################################
            ########## load parameters & data 
            #############################################
            print(f'Mask Type: {mask_type}. Loss Type: {loss_type}.')
            parameters['mask_type'] = mask_type
            parameters['loss_type'] = loss_type
            
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
            dataset_val = taxi_data(val_imgs, val_masks, image_size, chunk_size)
            dataset_test_biased = taxi_data(test_imgs, test_masks_biased, image_size, chunk_size)
            dataset_test_random = taxi_data(test_imgs, test_masks_random, image_size, chunk_size)
            iterator_train = iter(data.DataLoader(dataset_train, batch_size=batch_size, 
                                                  sampler=InfiniteSampler(len(dataset_train), chunk_size)))
            
            #############################################
            ########## training
            #############################################        
            if loss_type == 'l1': 
                result = train_l1(iterator_train, dataset_val, dataset_test_biased, dataset_test_random, parameters,all_results)
                result['mask_type'] = mask_type
                result['loss_type'] = loss_type
                all_results.append(result)
            else: 
                result = train_relative(iterator_train, dataset_val, dataset_test_biased, dataset_test_random,parameters,all_results)
                result['mask_type'] = mask_type
                result['loss_type'] = loss_type        
                all_results.append(result)
              
        ## save results
        process_results(all_results, chunk_size)
    
if __name__ == '__main__':
    main()    