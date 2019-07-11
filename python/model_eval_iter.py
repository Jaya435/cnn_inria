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
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',help='select batch size', type=int,default=128)
parser.add_argument('--lr',help='learning rate for optimizer',type=float,default=0.001)
parser.add_argument('--num_epochs',help='Number of epochs',type=int,default=100) 
parser.add_argument('--arch_size', help='inital depth of convolution', type=int,default=64)

#@click.command()
#@click.argument('batch_size', default=128, help='Batch size of image for input')
#@click.argument('learning_rate', default=0.01, help='Learning rate for optimiser')
#@click.argument('num_epochs', default=100, help='Number of epochs to train model')
#@click.argument('arch_size', default=64, help='Size of first convolutional feature')

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
        self.cr = args.arch_size
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

# number of epochs to train the model
# track change in validation loss
def train_eval(train_loader, valid_loader, n_epochs, model, optimizer):
    '''function to train the model'''
    start = time.time()
    valid_loss_min = np.Inf 

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        net.train()
        for info in train_loader:
            
            data,target = info[0].to(device),info[1].to(device)
            # move tensors to GPU if CUDA is available
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = multi_class_cross_entropy_loss_torch(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
        train_run_loss.append(loss.item())

        ######################    
        # validate the model #
        ######################
        net.eval()
        for info in valid_loader:
            data,target = info[0].to(device),info[1].to(device)
            # move tensors to GPU if CUDA is available
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss =  multi_class_cross_entropy_loss_torch(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
        valid_run_loss.append(loss.item())

        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), '/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/saved_models/model_inria_batch{}_lr{}_arch{}_epochs{}.pt'.format(
                args.batch_size,args.lr,args.arch_size,args.num_epochs))
            valid_loss_min = valid_loss
    finish = time.time()
    print ('Time Taken for {} epochs: {}'.format(n_epochs,finish-start))
    return(train_run_loss,valid_run_loss)

   
def train_valid_test_split(image_paths, target_paths):
    dataset = BuildingsDataset(image_paths, target_paths)
    validation_split = .2
    test_split = .1
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    valid_split = int(np.floor(validation_split * dataset_size))
    test_split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[valid_split:], indices[test_split:valid_split], indices[:test_split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset,args.batch_size,num_workers=0,sampler=train_sampler)

    valid_loader = DataLoader(dataset,args.batch_size,num_workers=0,sampler=valid_sampler)

    test_loader = DataLoader(dataset, args.batch_size, num_workers=0, sampler=test_sampler)
    return (train_loader, valid_loader, test_loader)

def model_eval(test_loader,net):
    net.eval()
    with torch.no_grad():
        count = 0
        correct = 0
        total = 0
        start = time.time()                                                                  
        for (images, labels) in test_loader:
            images = Variable(images)      
            labels = Variable(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images) 
            _, predicted = torch.max(outputs.data, 0)
            labels = labels.long()
            total += labels.nelement()
            correct += (predicted == labels).sum().item()
        print('Correct: {}, Total: {}'.format(correct,total))
        stop = time.time()   
        result = ('Accuracy: {:.3f} %, Time: {:.2f}s, Batch_Size: {}, Learning Rate: {}, Initial Architecture: {}'.format(((correct / total)*100),(stop-start),args.batch_size, args.lr, args.arch_size))
        print(result)
        f = open('results_120.txt','a')
        f.write(result + '\n')
        f.close()
if __name__ == "__main__":

    args = parser.parse_args()
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    
    image_paths = glob.glob('/exports/eddie/scratch/s1217815/AerialImageDataset/train/images/*.tif') 
    target_paths = glob.glob('/exports/eddie/scratch/s1217815/AerialImageDataset/train/gt/*.tif') 
    test_paths = glob.glob('/exports/eddie/scratch/s1217815/AerialImageDataset/test/images/*.tif') 
      
    train_run_loss,valid_run_loss = [],[]

    net = Net() # resets model

    optimizer = optim.Adam(net.parameters(), args.lr)
    
    train_loader, valid_loader, test_loader = train_valid_test_split(image_paths, target_paths)
    print('Data Load Successful')
    if torch.cuda.device_count() > 1:
        print("Let us use",torch.cuda.device_count(),"GPUS!")
        net = nn.DataParallel(net)
        net.to(device)
        train_run_loss,valid_run_loss = train_eval(train_loader, valid_loader, args.num_epochs, net,optimizer)
        plt.plot(train_run_loss,label='Training Loss')
        plt.plot(valid_run_loss,label='Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('lr{}net_size{}batch{}epochs{}.png'.
                    format(args.lr,args.arch_size,args.batch_size,args.num_epochs),bbox_inches='tight')
    else:
        net.to(device)
        print(next(net.parameters()).is_cuda)
        train_run_loss,valid_run_loss = train_eval(train_loader,valid_loader,args.num_epochs, net,optimizer)
    #load best model
    net = Net()
    net = nn.DataParallel(net)
    net.to(device)
    net.load_state_dict(torch.load('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/saved_models/model_inria_batch{}_lr{}_arch{}_epochs{}.pt'.format(args.batch_size,args.lr,args.arch_size,args.num_epochs)))
    model_eval(test_loader,net)
    
