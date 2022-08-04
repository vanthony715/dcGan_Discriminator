#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: avasquez
"""
import os
import shutil
import cv2
# import torch
# import torchvision.utils as vutils
import xml.etree.ElementTree as ET

def clearFolder(Path):
    if os.path.isdir(Path):
        print('Removing File: ', Path)
        shutil.rmtree(Path)
    print('Creating File: ', Path)
    os.mkdir(Path)
    
# def saveNet(OutPath, Network, Epoch, NetName):
#         torch.save(Network.state_dict(), 
#                    os.path.join(OutPath, NetName + '_{}.pth'. format(Epoch)))
        
# def saveImage(OutPath, Sample, Epoch, Name):
#     SavePath = OutPath + Name + str(Epoch) + '.png'
#     vutils.save_image(Sample, SavePath, normalize = True)

class WriteChips:
    def __init__(self, file, annotpath, imagepath, writepath, classList, totalChipCnt):
        self.totalChipCnt = totalChipCnt 
        self.file = file
        self.annotpath = annotpath
        self.imagepath = imagepath
        self.writepath = writepath
        self.classList = classList

    ##gets chip coords from XML file
    def getChip(self):
        tree = ET.parse(self.annotpath + self.file)
        root = tree.getroot()
        pre, _ =  os.path.splitext(self.file)
        bbDict = {'name':[], 'xmin':[], 'ymin':[], 'xmax':[], 'ymax':[], 'imageChip':[]}
        # print(AnnotPath + File)
    
        for elem in root.iter('name'): 
            if elem.text in self.classList:
                bbDict['name'].append(elem.text)
                
        for elem in root.iter('xmin'):
            bbDict['xmin'].append(elem.text)
            
        for elem in root.iter('ymin'): 
            bbDict['ymin'].append(elem.text)
            
        for elem in root.iter('xmax'):
            bbDict['xmax'].append(elem.text)
            
        for elem in root.iter('ymax'): 
            bbDict['ymax'].append(elem.text)
        
        ##read image to extract chip
        image = cv2.imread(self.imagepath + pre + '.jpg')
        
        ##calc chips from large image
        chipCnt = 0
        for i in range(len(bbDict['name'])):
            name = bbDict['name'][i]
            xmin = bbDict['xmin'][i]
            ymin = bbDict['ymin'][i]
            xmax = bbDict['xmax'][i]
            ymax = bbDict['ymax'][i]
            w = int(xmax) - int(xmin)
            h = int(ymax) - int(ymin)
            chip = image[int(ymin) : int(ymin) + h, int(xmin) : int(xmin) + w]
        
            ##write chip to class directory
            cv2.imwrite(self.writepath + name + '/' + str(self.totalChipCnt).zfill(6) + '-' + name + '.jpg', chip)
            
            ##write chip to allChips directory
            cv2.imwrite(self.writepath + 'AllChips/' + str(self.totalChipCnt).zfill(6) + '-' + name + '.jpg', chip)
            
            chipCnt += 1
        return chipCnt
    
def resizeChip(ImageChip, scaleWidth, scaleHeight):
    try:
        dim = (scaleWidth, scaleHeight)
        resizedChip = cv2.resize(ImageChip, dim)
    except:
        pass
    return resizedChip

# def writeChip(Pre, resizedChip, chipWritePath):
#     try:
#         # cv2.imwrite(chipWritePath + str(cnt).zfill(6) + '-' + Class + '.jpg', resizedChip)
#         filename = Pre + '.jpg'
#         cv2.imwrite(chipWritePath + filename, resizedChip)
#     except:
#         print('Error in WriteChip')
#     return 0