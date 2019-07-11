#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch
import itertools as it
from PIL import Image
import os
import random
import glob
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from torch.utils import data as D
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import time
import argparse
from torchsummary import summary
#print ('Cuda Available: {}'.format(torch.cuda.is_available()))
#print ('Current Device: {}'.format(torch.cuda.current_device()))
#print ('Device Count: {}'.format(torch.cuda.device_count()))
#print ('Device Name: {}'.format(torch.cuda.get_device_name(0)))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ('Device: {}'.format(device))
# parser = argparse.ArgumentParser()
# parser.add_argument('--depth', help='inital depth of convolution', type=int,default=32)
# parser.add_argument('--batch_size',help='select batch size', type=int,default=1)
# parser.add_argument('--lr',help='learning rate for optimizer',type=float,default=0.001)
# parser.add_argument('--epochs',help='Number of epochs',type=int,default=100)
param_grid = {
    'max_depth':[2,4,8,16,32,64,128,256],
    'learning rate':[1,0.1,0.01,0.001,0.0001],
    'batch_size':[2,4,8,16,32,64,128,256]
}
combinations = it.product(*(param_grid[param] for param in param_grid))
#print(len(list(combinations)))


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
    def __init__(self):
        super().__init__()
        self.cr = 32
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
            SegBlockDecoder(in_channel=self.cr*2,out_channel=self.cr),
            SegBlockDecoder(in_channel=self.cr, out_channel= 2)
            )
        
        self.output = nn.Softmax(dim = 1)

    def forward(self,x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        y = self.output(x2)
        #y_f = y[:,0,:,:]+y[:,1,:,:]
        return y

def multi_class_cross_entropy_loss_torch(predictions, labels):
    """
    Calculate multi-class cross entropy loss for every pixel in an image, for every image in a batch.

    In the implementation,
    - the first sum is over all classes,
    - the second sum is over all rows of the image,
    - the third sum is over all columns of the image
    - the last mean is over the batch of images.
    
    :param predictions: Output prediction of the neural network.
    :param labels: Correct labels.
    :return: Computed multi-class cross entropy loss.
    """

    loss = -torch.mean(torch.sum(torch.sum(torch.sum(labels * torch.log(predictions), dim=1), dim=1), dim=1))
    return loss


def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

class BuildingsDataset(Dataset):
    """INRIA buildings dataset"""
    
    def __init__(self,images_dir,gt_dir,train=True):
        """
        Args:
        images_dir = path to the satellite images
        gt_dir = path to the binary mask
        """
        self.image_paths = images_dir
        self.target_paths = gt_dir
        self.train=train
        
    def transform(self, image, mask):
        # Resize
#         resize = transforms.Resize(size=(5000, 5000))
#         image = resize(image)
#         mask = resize(mask)
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(128, 128))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = torch.cat([(mask==0).float(),(mask==1).float()],dim=0)
        return image, mask
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)

image_paths = glob.glob('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/train/images/*.tif')
target_paths = glob.glob('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/train/gt/*.tif')
valid_paths = glob.glob('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/test/images/*.tif')
ind_10=len(image_paths)//10
train = BuildingsDataset(image_paths[ind_10:], target_paths[ind_10:])
test = BuildingsDataset(image_paths[:ind_10], target_paths[:ind_10],train=False)

# image_paths = glob.glob('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/inria_test/images/*.tif')
# target_paths = glob.glob('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/inria_test/gt/*.tif')
# print('Size of training data: {}'.format(len(image_paths)))

# test = BuildingsDataset(image_paths, target_paths)
# train = BuildingsDataset(image_paths, target_paths,train=False)

x = random.randint(0,len(train)-1)
#print (x)
image_pair = train[x]

for_plot = (image_pair[1].permute(1,2,0))
for_plot_pos = for_plot[:,:,0]
for_plot_neg = for_plot[:,:,1]

fig=plt.figure(figsize=(15, 15), dpi= 300, facecolor='w', edgecolor='k')
ax1 = plt.subplot(1, 3, 1)
ax1.imshow((image_pair[0].permute(1,2,0)))
ax2 = plt.subplot(1,3,2)
ax2.imshow(for_plot_pos,cmap='binary')
ax3 = plt.subplot(1,3,3)
ax3.imshow(for_plot_neg,cmap='binary')
#plt.savefig('example_dataloader.png',bbox_inches='tight')
plt.close()
def fit(num_epochs, model, opt):
    '''function to train the model'''
    train_loss, valid_loss, t_taken, num_epoch = [], [], [],[]

    for epoch in range(num_epochs):
        '''Repeat for a given number of epochs'''# run the model for 200 epochs
        t0 = time.time()
    
        model.train()
        print ('fit function begun')
        for data in train_loader:
            inputs,labels = data[0].to(device),data[1].to(device)
            pred = model(inputs)
            loss = multi_class_cross_entropy_loss_torch(pred, labels)
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_loss.append(loss.item())
            t_taken.append(time.time())
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.3f}, Time: {:.3f}'.format(
                epoch+1, num_epochs, loss.item(),time.time()-t0))
        
    print('Finished Training')
    return (train_loss)



net = Net() # resets model
net_size = 64
lr,net.cr,batch_size,num_epochs= 0.001,net_size,128,100

optimizer = optim.Adam(net.parameters(), lr)

train_loader = DataLoader(train,batch_size,num_workers=0,shuffle=True)

test_loader = DataLoader(test,batch_size,shuffle=False,num_workers=0)
if torch.cuda.device_count() > 1:
    print("Let us use",torch.cuda.device_count(),"GPUS!")
    net = nn.DataParallel(net)
    net.to(device)
    train_loss = fit(num_epochs, net,optimizer)
else:
    net.to(device)
    print(next(net.parameters()).is_cuda)
    train_loss = fit(num_epochs, net,optimizer)
plt.plot(train_loss)


plt.savefig('lr{}net_size{}batch{}epochs{}.png'.
            format(lr,net_size,batch_size,num_epochs))

#plt.show()

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images,labels = data[0].to(device),data[1].to(device)
        outputs = net(images)
        labels = labels.long()
        outputs = outputs.long()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


