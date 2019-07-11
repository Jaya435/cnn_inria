#!/usr/bin/env python
# coding: utf-8

import torch
import itertools as it
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
from torch.autograd import Variable
import matplotlib.patches as mpatches
import argparse
import os
import re

parser = argparse.ArgumentParser(description='Predict the class of each pixel for an image and save the result. Images taken from train folder and include mask')
parser.add_argument('-model',type=str,default='model_inria.pt',help='A saved pytorch model')
parser.add_argument('-inpfile',type=str,default='/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/train/images/kitsap31.tif',help='Path and filename of image to be classified')
parser.add_argument('-mask',type=str,default='/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/train/gt/kitsap31.tif',help='Path to mask in train folder')

class SegBlockEncoder(nn.Module):
    def __init__(self,in_channel,out_channel, kernel=4,stride=2,pad=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channel,out_channel,kernel,stride=stride,
                padding=pad,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        y=self.model(x)
        return y
class SegBlockDecoder(nn.Module):
    def __init__(self,in_channel,out_channel, kernel=4,stride=2,pad=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channel,out_channel,kernel,stride=stride,padding=pad,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        y=self.model(x)
        return y

class Net(nn.Module):
    def __init__(self,cr=2):
        super(Net,self).__init__()
        self.cr = cr
        self.encoder = nn.Sequential(
            SegBlockEncoder(in_channel=3,out_channel=self.cr),
            SegBlockEncoder(in_channel=self.cr,out_channel=self.cr*2),
            SegBlockEncoder(in_channel=self.cr*2,out_channel=self.cr*4),
            SegBlockEncoder(in_channel=self.cr*4,out_channel=self.cr*8),
            SegBlockEncoder(in_channel=self.cr*8,out_channel=self.cr*16)
            )
        
        self.decoder = nn.Sequential(
            SegBlockDecoder(in_channel=self.cr*16, out_channel=self.cr*8),
            SegBlockDecoder(in_channel=self.cr*8, out_channel=self.cr*4),
            SegBlockDecoder(in_channel=self.cr*4, out_channel=self.cr*2),
            SegBlockDecoder(in_channel=self.cr*2, out_channel=self.cr),
            SegBlockDecoder(in_channel=self.cr, out_channel=2), 
            )

        self.output = nn.Softmax(dim =1)
        
    def forward(self,x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        y = self.output(x2)
        return y

#net = Net()
#net = nn.DataParallel(net)
#net.load_state_dict(torch.load('model_inria_100.pt',map_location=lambda storage, loc: storage))

def accuracy(out, labels):
  outputs = np.argmax(out, axis=1)
  return np.sum(outputs==labels)/float(labels.size)

#loader = transforms.Compose([ transforms.ToTensor()])

trans = transforms.ToPILImage()
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    RGB_image = trans(image)
    image = Variable(image, requires_grad=True)
    if torch.cuda.is_available():                                                                     
        image = image.cuda()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image,RGB_image  #assumes that you're using GPU

#image,RGB_image = image_loader('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/test/images/sfo7.tif')
def image_plotter(image,RGB_image,mask,fname):
    output = net(image)
    variable = Variable(output)
    if torch.cuda.is_available():
        variable = variable.cuda()
    num = variable.data[0]
    num = num.permute(1,2,0)
    b = num.numpy()
    # building_images = b[:,:,0]
    # no_building = b[:,:,1]
    # fig=plt.figure(figsize=(5, 5), dpi= 300, facecolor='w', edgecolor='k')
    # ax1 = plt.subplot(1,2,1)
    # ax2 = plt.subplot(1,2,2)
    # ax1.imshow(building_images)
    # ax2.imshow(no_building)
    final_prediction=b[:,:,0]
    labels = (final_prediction > 0.5).astype(np.int)
    fig=plt.figure(figsize=(15, 10), dpi= 300, facecolor='w', edgecolor='k')
    ax1 = plt.subplot(1,3,1)
    ax1.set_title('Original RGB_image of {}'.format(fname))
    ax2 = plt.subplot(1,3,3)
    ax2.set_title('Prediction of buildings in {}'.format(fname))
    ax3=plt.subplot(1,3,2)
    ax3.set_title('Mask of buildings in {}'.format(fname))
    im = ax2.imshow(labels, interpolation='none',cmap="binary")
    patches = [(mpatches.Patch(color='black', label="No Buildings")),(mpatches.Patch(color='white', label="Buildings"))]
    ax2.legend(handles=patches, bbox_to_anchor=(1.45, 1), loc="upper right", borderaxespad=0.,facecolor="plum" )
    ax1.imshow(RGB_image)
    ax3.imshow(mask,cmap="binary_r")
    #ax3.legend(handles=patches, bbox_to_anchor=(1.45,1),loc="upper right",borderaxespad=0.,facecolor="plum")
    plt.savefig('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/predict/Predicted_{}.png'.format(fname),bbox_inches='tight',dpi=300)

if __name__ == '__main__':
    args=parser.parse_args()
    path = args.inpfile
    base = os.path.basename(path) 
    fname = os.path.splitext(base)[0]
    mask = Image.open(args.mask)
    model = args.model
    match = re.search('arch(\d+)',model)
    net_size = match.group(1)
    net_size = int(net_size)
    net = Net(net_size)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(model,map_location=lambda storage, loc: storage))
    loader = transforms.Compose([ transforms.ToTensor()])
    image,RGB_image = image_loader(args.inpfile)
    image_plotter(image,RGB_image,mask,fname)
