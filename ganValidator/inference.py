#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inputs: Image chips
Outputs: Score Dictionary

@author: avasquez
"""

import os
import sys
import gc
import time
import utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms

##clear bins
gc.collect()
torch.cuda.empty_cache()

D_HIDDEN = 64
IMG_CHANNEL = 3
Z_DIM = 100
G_HID = 64
X_DIM = 64
D_HID = 64

class ImageFolderWithPaths(dset.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
##discriminator network
class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.main = nn.Sequential(
            ##layer 1
            nn.Conv2d(IMG_CHANNEL, D_HID, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            ##layer 2
            nn.Conv2d(D_HID, D_HID * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(D_HID * 2),
            nn.LeakyReLU(0.2, inplace = True),
            ##layer 3
            nn.Conv2d(D_HID * 2, D_HID * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(D_HID * 4),
            nn.LeakyReLU(0.2, inplace = True),
            ##layer 4
            nn.Conv2d(D_HID * 4, D_HID * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(D_HID * 8),
            nn.LeakyReLU(0.2, inplace = True),
            ##output layer
            nn.Conv2d(D_HID * 8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid())
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

if __name__ == "__main__":
    
    mnt = '/mnt/opsdata/neurocondor/datasets/avasquez/data/'
    # basepath = mnt + 'ML/objectDetection/NeuroTracker/data/demo/MWIRN-01_47_45/'
    basepath = mnt + 'Neuro/'
    disWeights = basepath + 'chips/output/dNet_5.pth'
    dataPath = basepath + 'chips/chipsNSemiTest/'
    
    CUDA = False
    # BATCH_SIZE = len(os.listdir(dataPath + 'imgs/'))
    BATCH_SIZE = len(os.listdir(dataPath + 'imgs/'))
    CUDA = CUDA and torch.cuda.is_available()
    print('Pytorch Version: {}'.format(torch.__version__))
    
    if CUDA:
        print('CUDA version: {}'.format(torch.version.cuda))
    cudnn.benchmark = True
    device = torch.device("cuda:0" if CUDA else "cpu")
    
    dataset = ImageFolderWithPaths(root=dataPath,
                             transform=transforms.Compose([
                                 transforms.Resize(X_DIM),
                                 transforms.CenterCrop(X_DIM),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5)),
                             ]))
     
    ##define data loaded
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                                             shuffle=True, num_workers=2)
    
    dataiter = iter(dataloader)
    inputs, label, filename = dataiter.next()
    
    ##run net on data
    discriminator = DNet()
    print(discriminator)
    discriminator.load_state_dict(torch.load(disWeights))
    ##disable gradient computation
    
    ##start clock
    tic = time.time()
    output = discriminator(inputs)
    ##stop clock
    toc = time.time()
    ##time to process one chip
    cpt = round((toc - tic), 4) / BATCH_SIZE
    
    outputArray = output.detach().cpu().numpy()
    
    cnt=0
    for i in outputArray:
        if i >= .75:
            cnt+=1
    print(cnt)
    # detDict = utils.getScores(filename, label, outputArray)

    gc.collect()
    ##Clock time
    print('\n----Time----\n')
    print('Inference time per chip (s): ', round(cpt, 4))