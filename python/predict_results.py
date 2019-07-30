#!/usr/bin/env python

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os, glob
import re

cwd = os.getcwd()
print(cwd)

filenames = ('austin20', 'kitsap30', 'tyrol-w15','chicago10','vienna15')

predArray, imgArray, gtArray = [],[],[]
for f in filenames:
    predArray.append(Image.open('{}/predict_{}.tif'.format(cwd,f)))
    imgArray.append(Image.open('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/train/images/{}.tif'.format(f))) 
    gtArray.append(Image.open('/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/train/gt/{}.tif'.format(f))) 
    bigArray = imgArray + gtArray + predArray

fileList=[]
for i in bigArray:
    fi = i.filename
    fname = os.path.basename(os.path.normpath(fi))[:-6]
    if 'predict_' in fname:
        fname = fname.replace('predict_','')
    fileList.append(fname)

fileList = np.array(fileList)
#bigArray = np.array(bigArray)
inds = fileList.argsort()

cols = ['{}'.format(col) for col in ['RGB', 'Ground Truth','Predicted']]  
rows = ['{}'.format(row) for row in (filenames)] 

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(6, 10)) 

for ax, col in zip(axes[0], cols): 
    ax.set_title(col)

for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=90, size='large') 

counter=0
for i,row in enumerate(axes):
    for j, cell in enumerate(row):
        cutout = bigArray[inds[counter]].crop((0,0,1000,1000))
        if (counter+1) % 3 == 0:
            color = 'binary'
        elif counter + 1 % 3 != 0:
            color = 'binary_r'
        else:
            color = None
        cell.imshow(cutout,cmap=color)
        counter += 1
        cell.set_xticks([])
        cell.set_yticks([])

fig.subplots_adjust(hspace=0,wspace=0)
fig.tight_layout()
plt.savefig('Predicted_Grid.png', bbox_inches='tight')
#plt.show()
