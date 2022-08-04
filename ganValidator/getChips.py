#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate chips if needed
Input: Annotations and JPEGImages
Output: Chips

@author: avasquez
"""
import os
import time
import gc
import utils
from utils import WriteChips
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    ##start clock
    t0 = time.time()
    
    ##class of chip from annots must be in this list
    classList = ['Bird', 'FixedWing', 'GeneralAviation', 'Helicopter', 'Hex', 'Rotor', 'Unknown', 'Payload', 'AllChips']
    
    mnt = '/mnt/opsdata/neurocondor/datasets/avasquez/'
    basepath = mnt + 'data/Camera/IR/processed/'
    directories = [i for i in os.listdir(basepath) if os.path.isdir(basepath + i)]
    
    ##Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=str, default = basepath,
                        help="common path")
    parser.add_argument("--writepath", type=str, default = mnt + 'data/Camera/IR/chips/', 
                        help="where to write chips")
    args = parser.parse_args()
    
    ##clear out write path
    utils.clearFolder(args.writepath)
    
    for Class in classList:
        utils.clearFolder(args.writepath + Class + '/')
    
    fileCnt = 0 ##cnt files processed
    totalChipCnt = 0
    for directory in tqdm(directories, desc='Total Directory Status', colour='green'):
        annotPath = args.basepath + directory + '/Annotations/'
        largeImagePath = args.basepath + directory + '/JPEGImages/'
        
        desc = 'Directory Status ' + directory
        for file in tqdm(os.listdir(annotPath), desc = desc, colour='blue',mininterval=.05, smoothing=0.9):
            ##make sure that the file is a file
            if not os.path.isdir(annotPath + file):
                pre, _ = os.path.splitext(file)
                annotation = annotPath + pre + '.xml'
                largeImage = largeImagePath + pre + '.jpg'
                try:
                    chipCnt = WriteChips(file, annotPath, largeImagePath, args.writepath, classList, totalChipCnt).getChip()
                    totalChipCnt += chipCnt 
                    fileCnt += 1
                except:
                    print('\nError in WriteChips: ', file)
            
    t1 = time.time()
    tf = round((t1 - t0), 5)
    print('----------------Stats---------------------')
    print('Files Processed: ', fileCnt)
    print('Chips Processed: ', totalChipCnt)
    print('Execution Time: ', tf)
    gc.collect()
