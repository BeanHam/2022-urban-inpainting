import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
from net import *
from utils import *
warnings.filterwarnings('ignore')

def main():
    
    ## load data
    img_root = "../../../../data_min_max"
    mask_root = "../../../../data_min_max"
    image_size = 64
    batch_size = 16
    test_imgs = np.load(img_root+'/test.npy')
    test_masks_random = np.load(mask_root+'/test_random_mask.npy')
    
    ## images
    for chunk_size in [1,2,3,5,7,10,15]:
        for mape_loss in [0, 1]:
            
            print('======================')
            print(f'Chunk Size: {chunk_size}. Mape: {mape_loss}')
            
            ## parameters
            dataset_test = taxi_data(test_imgs, test_masks_random, image_size, chunk_size)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            
            ## load model
            model = PConvUNet(chunk_size=chunk_size).to(device)
            model.load_state_dict(torch.load(f'../../model_states/chunk_size_{chunk_size}/biased_{mape_loss}'))
            model.eval()
            
            ## spatial error
            gt_total = torch.zeros((64,64))
            sequential_error = []
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
                gt = (gt*all_time_max).cpu().squeeze_(1)
                output_comp = (output_comp*all_time_max).cpu().squeeze_(1)
                
                # sequential error
                for m in range(len(gt)):
                    for n in range(chunk_size):
                        sequential_error += [torch.mean(output_comp[m][n] - gt[m][n]).numpy().tolist()]
            
            ## 24-hour period
            sequential_error_24period = np.zeros(24)
            for i in range(int(len(sequential_error)/24)):
                sequential_error_24period += np.array(sequential_error[i*24:(i+1)*24])
            sequential_error_24period = sequential_error_24period/int((len(sequential_error))/24)
        
            ## sequential plots
            plt.figure(figsize=(8,5))
            plt.title(f'Temporal Error Distribution, T={chunk_size}. MAPE={mape_loss}', size=18,y=1.05)
            plt.scatter(np.array(range(24)), sequential_error_24period)
            plt.plot(sequential_error_24period)
            plt.hlines(xmin=0,xmax=24,y=0,color='red')
            plt.xlabel('Hour of the Day', fontsize=16)
            plt.ylabel('Avg. L1 Error (Hole)', fontsize=16)
            plt.yticks(size=14)
            plt.yticks(size=14)
            plt.savefig(f'T-{chunk_size}-MAPE-{mape_loss}.png', bbox_inches='tight')
            pass

if __name__ == '__main__':
    main()    
