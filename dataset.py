import os
#import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import PIL
#from tensorlayer.prepro import flip_axis, elastic_transform, rotation, shift, zoom, brightness
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, RandomResizedCrop, ToPILImage, ToTensor

from mri_utils import Downsample

def load_mask():
    from scipy.io import loadmat
    mask = loadmat('../mask/Gaussian1D/GaussianDistribution1DMask_30.mat')
    #mask = loadmat('mask/Gaussian1D/GaussianDistribution1DMask_20.mat')
    mask = mask['maskRS1']
    return mask

class MRIDataset(data.Dataset):
    def __init__(self, root, image_set, transform=False):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.ids = [i.strip() for i in open(root + image_set + '.txt').readlines()]

        self.mask = load_mask()
        self.DAGAN = 'SegChallenge' in root and image_set == 'test'

    def __getitem__(self, index):
        if self.DAGAN:
            x = np.load(os.path.join(self.root, self.image_set, self.ids[index]))
        else:
            x = cv2.imread(os.path.join(self.root, self.image_set, self.ids[index]), cv2.IMREAD_GRAYSCALE)

        # data augmentation
        if self.transform:
            #x = x[..., np.newaxis]
            #x = flip_axis(x, axis=1, is_random=True)
            ##x = elastic_transform(x, alpha=255 * 3, sigma=255 * 0.10, is_random=True) TODO
            #x = rotation(x, rg=10, is_random=True, fill_mode='constant')
            #x = shift(x, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
            #x = zoom(x, zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
            ##x = brightness(x, gamma=0.05, is_random=True)
            #x = x[..., 0]

            #ori = x.copy()

            transformations = Compose(
                [ToPILImage(),
                 RandomRotation(degrees=10, resample=PIL.Image.BICUBIC),
                 #RandomAffine(degrees=10, translate=(-25, 25), scale=(0.90, 1.10), resample=PIL.Image.BILINEAR),
                 #RandomHorizontalFlip(),
                 RandomResizedCrop(size=256, scale=(0.90, 1.0), ratio=(0.95, 1.05), interpolation=PIL.Image.BICUBIC),
                 #CenterCrop(size=(256, 256)),
                 ToTensor(),
                ])
            x = x[..., np.newaxis]
            x = transformations(x).float().numpy() * 255
            x = x[0]

        image, _, _ = Downsample(x, self.mask)
        #image = x
        #image = np.round(np.clip(image, 0, 255)) XXX this is not wrong (written before 2019.08.13), but DAGAN has some data problem 
        #image = np.clip(image, 0, 255)
        #print(np.max(image), np.min(image))
        #cv2.imwrite('aa.bmp', ori.astype(np.uint8))
        #cv2.imwrite('a.bmp', image.astype(np.uint8))
        #xxx

        #return x, image
        x = x / 255.
        image = image / 255.
        return torch.from_numpy(x).float().unsqueeze(0), torch.from_numpy(image).float().unsqueeze(0)

    def __len__(self):
        return len(self.ids)
