import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
from termcolor import colored
from net import *
from utils import *
warnings.filterwarnings('ignore')

def main():
    
    ## load data
    img_root = "../../../../data_min_max"
    mask_root = "../../../../data_min_max"
    test_imgs = np.load(img_root+'/test.npy')
    test_masks_random = np.load(mask_root+'/test_random_mask.npy')
    image_size = 64
    batch_size = 16

    ## images
    for chunk_size in [1,2,3,5,7,10]:
        for mask_type in ["random", "biased"]:
            print('=====================')
            print(f'Chunk Size: {chunk_size}. Mask: {mask_type}')
            
            ## parameters
            dataset_test = taxi_data(test_imgs, test_masks_random, image_size, chunk_size)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            
            ## load model
            model = PConvUNet(chunk_size=chunk_size).to(device)
            model.load_state_dict(torch.load(f'../../model_states/chunk_size_{chunk_size}/{mask_type}_0'))
            model.eval()
            
            ## spatial error
            gt_total = torch.zeros((64,64))
            spatial_error = torch.zeros((64,64))
            all_time_max = 1428
        
            outer_indices = [k for k in range(chunk_size-1, len(dataset_test), chunk_size)]
            for i in range(0, len(outer_indices), batch_size):
                ## get data
                inner_indices = range(i, min(len(outer_indices), i+batch_size))
                mask, gt= zip(*[dataset_test[outer_indices[j]] for j in inner_indices])
                mask = torch.stack(mask).to(device)
                gt = torch.stack(gt).to(device)
                image = mask*gt
            
                ## modeling
                with torch.no_grad():
                    output, _ = model(gt, mask)
                output_comp = mask * gt + (1 - mask) * output
                
                ## scale back
                gt = (gt*all_time_max).cpu()
                output_comp = (output_comp*all_time_max).cpu()
                
                ## spatial error
                spatial_error = spatial_error + torch.sum(torch.sum(torch.sum(output_comp-gt, dim=0),dim=0),dim=0)
            
            ## spatial error
            spatial_error = spatial_error/len(dataset_test)
        
            ## spatial visualization
            plt.figure(figsize=(8,8))
            plt.axis('off')
            plt.title(f'Spatial Error Distribution - T={chunk_size}, Mask={mask_type}', fontsize=18,y=1.05)
            plt.imshow(spatial_error, cmap='bwr', vmax=4, vmin=-4)
            over = torch.sum(spatial_error[spatial_error > 0]).item()
            under = torch.sum(spatial_error[spatial_error < 0]).item()
            overestimation_test = colored('Overestimation', 'red')
            underestimation_test = colored('Underestimation', 'red')
            plt.text(0,45,f'Total Overestimation Value: {round(over, 2)}',size=16, color="red")
            plt.text(0,50,f'Total Underestimation Value: {round(under,2)}',size=16, color="blue")
            plt.text(0,55,f'Total Absolute Error Value: {round(np.abs(under)+over,2)}',size=16)
            plt.colorbar(fraction=0.03)
            plt.savefig(f'T-{chunk_size}-Mask-{mask_type}.png', bbox_inches='tight')
            pass

if __name__ == '__main__':
    main()    
