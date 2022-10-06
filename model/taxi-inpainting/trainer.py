import torch
import torch.nn as nn
import itertools
import warnings
import json
from utils import *
warnings.filterwarnings("ignore")

def main():
    
    ## seed
    seed = 816
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ## load parameters
    parameter_path = 'config.json'
    with open(parameter_path) as json_file:
        parameters = json.load(json_file)
    
    ## get parameter values
    mask_types = parameters['mask_types']
    chunk_sizes = parameters['chunk_sizes']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parameters['device'] = device
    
    ## training
    for chunk_size, mask_type in itertools.product(chunk_sizes, mask_types):
        
        print(f'==============================')
        print(f'Chunk Size: {chunk_size}')
        print(f'--- Mask Type: {mask_type}')
        parameters['chunk_size'] = chunk_size
        parameters['mask_type'] = mask_type
            
        ## load data
        dataset_train, dataset_val, dataset_test_biased, dataset_test_random = load_data(parameters)
            
        ## training
        train(dataset_train, 
              dataset_val, 
              dataset_test_biased, 
              dataset_test_random, 
              parameters)
    
if __name__ == '__main__':
    main()    
