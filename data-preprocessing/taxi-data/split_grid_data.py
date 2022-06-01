import numpy as np
import pandas as pd
import warnings
import random
import cv2
import os
import itertools
from PIL import Image
from skimage import measure
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


def hotspot_detection(image, kernel_size=5, quantile=0.9, scaling_factor=500):
    
    blurred = cv2.GaussianBlur(image, (kernel_size,kernel_size), 0)
    thresh = cv2.threshold(blurred, np.quantile(blurred.ravel(), quantile), 255, cv2.THRESH_BINARY)[1]
    
    ## how many groups/clusters in the image
    labels = measure.label(thresh, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    coord_groups = {}
    # loop over the unique components
    for label in np.unique(labels):
        if label == 0:
            continue 
        x_coords, y_coords = np.where(labels == label)
        coord_groups[label] = [(x_coords[i], y_coords[i]) for i in range(len(x_coords))]
    
    ## length of each group/cluster
    length = np.array([len(coord_groups[i]) for i in coord_groups.keys()])
    
    ## probability of selecting an coordinate from each group/cluster
    unique_labels = np.delete(np.unique(labels),0)
    probs = np.exp(length/scaling_factor)/sum(np.exp(length/scaling_factor))
    
    ## select a group
    selected_group = np.random.choice(unique_labels, 1, p=probs)[0]
    
    ## return selected coordiantes
    return random.choice(coord_groups[selected_group])

def random_walk(canvas, ini_x, ini_y, length):
    action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    x = ini_x
    y = ini_y
    img_size = canvas.shape[-1]
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=img_size - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=img_size - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 0
    return canvas

        
def main():
    
    np.random.seed(1)
    root = "D:/nyc_taxi/grid_data/"
    
    ## get train data
    train_years = list(range(2012,2016))
    train_data = []
    train_label = []
    for year in train_years:
        year_data = np.load(root+str(year)+'_grid_data.npy')
        train_data.append(year_data)
        train_label.append(np.concatenate(np.repeat(np.array(range(24)).reshape(1,-1), len(year_data)/24, axis=0)))
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    
    ## get val data
    val_year = 2016
    val_data = np.load(root+str(val_year)+'_grid_data.npy')
    val_label = np.concatenate(np.repeat(np.array(range(24)).reshape(1,-1), len(val_data)/24, axis=0))
    
    ## get test data
    test_data = val_data[31*24:]
    test_label = val_label[31*24:]
    val_data = val_data[:31*24]
    val_label = val_label[:31*24]
    
    ## check
    assert len(train_data) == len(train_label)
    assert len(val_data) == len(val_label)
    assert len(test_data) == len(test_label)
    
    ## get shape
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    ## max_min normalization
    all_time_max = np.max([np.max(train_data), np.max(val_data), np.max(test_data)])
    all_time_min = np.min([np.min(train_data), np.min(val_data), np.min(test_data)])
    train_data = (train_data - all_time_min)/all_time_max
    val_data = (val_data - all_time_min)/all_time_max
    test_data = (test_data - all_time_min)/all_time_max
    print(f'All Time Max: {all_time_max}')
    print(f'All Time Min: {all_time_min}')
    
    ## generate masks
    image_size = 64
    splits = ['train', 'val', 'test']
    mask_types = ['biased', 'random']
    masks = {}
    
    for split, mask_type in itertools.product(splits, mask_types):
        print(split+"_"+mask_type)
        
        if split == 'train': data = train_data
        elif split == 'val':data = val_data
        else: data = test_data
        
        mask = []
        for img in data:
            canvas = np.ones((image_size, image_size)).astype("i")
            ## biased masking initialization
            if mask_type == 'biased':
                try:
                    ini_x, ini_y = hotspot_detection(img)
                except:
                    ini_x = random.randint(0, image_size - 1)
                    ini_y = random.randint(0, image_size - 1)
            ## random initialization
            else:
                ini_x = random.randint(0, image_size - 1)
                ini_y = random.randint(0, image_size - 1)
            mask.append(random_walk(canvas, ini_x, ini_y, image_size ** 2))
        mask = np.stack(mask)
        masks[split+'_'+mask_type] = mask
    
    ## save data
    np.save(f"D:/nyc_taxi/data_3d/train.npy", train_data)
    np.save(f"D:/nyc_taxi/data_3d/test.npy", test_data)
    np.save(f"D:/nyc_taxi/data_3d/val.npy", val_data)

    ## save masks
    for key in masks.keys():
        np.save(f"D:/nyc_taxi/data_3d/{key}_mask.npy", masks[key])
        
if __name__ == "__main__":
    main()