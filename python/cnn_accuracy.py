#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import re
import model_eval_iter as main
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time
from torch.autograd import Variable
import os

class SegBlockEncoder(nn.Module):
    def __init__(self,in_channel,out_channel, kernel=4,stride=2,pad=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channel,out_channel,kernel,stride=stride,
                padding=pad,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            #nn.Tanh()
            #nn.PReLU()
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
            #nn.Tanh()
            #nn.PReLU()
        )

    def forward(self, x):
        y=self.model(x)
        return y

class Net(nn.Module):
    def __init__(self,cr=2):
        self.cr = cr
        super(Net,self).__init__()
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

def train_valid_test_split(image_paths, target_paths,batch_size):
    dataset = main.BuildingsDataset(image_paths, target_paths)
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
    #print(len(train_indices), len(val_indices), len(test_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset,batch_size,num_workers=0,sampler=train_sampler)

    valid_loader = DataLoader(dataset,batch_size,num_workers=0,sampler=valid_sampler)

    test_loader = DataLoader(dataset, batch_size, num_workers=0, sampler=test_sampler)
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
        print('Accuracy: {:.3f} %, Time: {:.2f}s'.format(100 * correct / total,stop-start))
    return ((100*correct)/total) 

def model_accuracy(models):
    accList = []
    for model in models:
        print ('Path to model {}'.format(model))
        base_model = os.path.basename(model)
        batch_size = get_batch(base_model)
        print(batch_size)
        train_loader, valid_loader, test_loader = train_valid_test_split(image_paths, target_paths,batch_size)
        print('Model: {}'.format(base_model))
        match = re.search('arch(\d+)', base_model)
        net_size = match.group(1)
        net_size = int(net_size)
        net = Net(net_size)
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(model,map_location=lambda storage, loc: storage))
        #net.cuda()
        accuracy = model_eval(test_loader, net)
        accList.append(accuracy)
    maxInd = accList.index(max(accList))
    print('Highest accuracy: {} for model: {}'.format(accList[maxInd],models[maxInd]))
    models[maxInd]
    return maxInd



image_paths = glob.glob('/exports/eddie/scratch/s1217815/AerialImageDataset/train/images/*.tif')
target_paths = glob.glob('/exports/eddie/scratch/s1217815/AerialImageDataset/train/gt/*.tif')
test_paths = glob.glob('/exports/eddie/scratch/s1217815/AerialImageDataset/test/images/*.tif') 
gd_models = glob.glob('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/saved_models/*batch*epochs100*.pt')

def get_batch(model):
    match = re.search('batch(\d+)',model)
    batch_size = match.group(1)
    batch_size = int(batch_size)
    return batch_size
    
maxInd = model_accuracy(gd_models)

