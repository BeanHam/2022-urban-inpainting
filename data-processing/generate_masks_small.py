import argparse
import numpy as np
import random
from PIL import Image
from skimage import measure
import cv2
import os 

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_dir', type=str, default='D:/nyc_taxi/')
    parser.add_argument('--save_dir', type=str, default='D:/nyc_taxi/masks')
    args = parser.parse_args()
    
    ## initialize all splits & types
    image_splits = ['train_img', 'val_img', 'test_img']
    mask_types = ['biased', 'random']
    
    ## iterate through train, test, & validation split
    ## & iterate through random and biased mask type
    for mask_type in mask_types:
        for split in image_splits:
            
            print(f'Processing {mask_type}, {split}...')
            
            final_save_dir = f'{args.save_dir}/{mask_type}/{split}'
            if os.path.exists(final_save_dir):
                next
            else:
                os.makedirs(final_save_dir)
                split_img_dir = f'{args.image_dir}/{split}/'
                img_files = os.listdir(split_img_dir)
                
                for img_file in img_files:
                    img = np.load(f'{split_img_dir}/{img_file}')
                    canvas = np.ones((args.image_size, args.image_size)).astype("i")
                    ## biased masking initialization
                    if mask_type == 'biased':
                        try:
                            ini_x, ini_y = hotspot_detection(img)
                        except:
                            ini_x = random.randint(0, args.image_size - 1)
                            ini_y = random.randint(0, args.image_size - 1)
                    ## random initialization
                    else:
                        ini_x = random.randint(0, args.image_size - 1)
                        ini_y = random.randint(0, args.image_size - 1)
                    mask = random_walk(canvas, ini_x, ini_y, args.image_size*args.image_size/2)
                    np.save(f'{final_save_dir}/{img_file}', mask)
                    
                    