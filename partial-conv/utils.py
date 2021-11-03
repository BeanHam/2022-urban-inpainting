import torch
import torch.nn as nn
import random
import torch
from PIL import Image
from glob import glob
from torch.utils import data
import numpy as np
    
class taxi_data(torch.utils.data.Dataset):
    def __init__(self, 
                 img_root, 
                 mask_root, 
                 img_transform, 
                 mask_transform,
                 split='train'):
        super(taxi_data, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        if split == 'train': self.paths = glob('{:s}/train/**/*.png'.format(img_root),recursive=True)
        else: self.paths = glob('{:s}/{:s}/*'.format(img_root, split))
        self.mask_paths = glob('{:s}/*.png'.format(mask_root))
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        #gt_img = self.img_transform(gt_img.convert('RGB'))
        gt_img = self.img_transform(gt_img)
        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask)
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0
                
def evaluate(model, criterion, dataset, device, batch_size):
    
    indices= np.random.randint(0, high=len(dataset), size=batch_size)
    _, mask, gt = zip(*[dataset[i] for i in indices])
    mask = torch.stack(mask).to(device)
    gt = torch.stack(gt).to(device)
    output, _ = model(gt, mask)
    loss = criterion(mask, output, gt)
    del mask, gt, output
        
    return (loss/batch_size).item()

def unorm(tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225])
    return inv_normalize(tensor)