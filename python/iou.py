#!/usr/bin/env python

from PIL import Image
import numpy as np
import os, glob, time
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def calcIOU(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
    
def getArrays(city):
    gt_dir = '/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/AerialImageDataset/train/gt'
    pred_dir = os.getcwd()
    gt_all = glob.glob('{}/*{}*.tif'.format(gt_dir, city))
    pred_all = glob.glob('{}/*{}*.tif'.format(pred_dir, city))
    gt_all.sort()
    pred_all.sort()
    print(len(pred_all))
    gtArray, predArray, target, predicition = [],[],[],[]
    for i in range(0, len(pred_all)):
        gtArray.append(getArray(gt_all[i]))
        predArray.append(getArray(pred_all[i]))
    target = np.vstack(gtArray)
    prediction = np.vstack(predArray)
    return target, prediction

def getArray(image):
    np_arr = np.array(Image.open(image))
    np_arr = np_arr/np.max(np_arr)
    return np_arr 

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow 
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.savefig(title,bbox_inches='tight')



cities = ('austin','chicago','kitsap','tyrol-w', 'vienna')
df = pd.DataFrame(columns=['City','IOU'])
df['City'] = cities
print(df)
iou_scores = []
for city in cities:
    target, prediction = getArrays(city)
    iou_score = calcIOU(target, prediction)
    iou_scores.append(iou_score)
    print ('{} IOU Score: {}'.format(city, iou_score*100)) 
    y_actu = pd.Series(target.ravel(), name= 'Actual')
    y_pred = pd.Series(prediction.ravel(),name='Prediction')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    #df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    df_confusion.to_csv('{}_confusion_matrix_not_norm.csv'.format(city))
    plot_confusion_matrix(df_confusion,title='{}_Confusion_matrix_not_norm.png'.format(city))
    del y_actu
    del y_pred
    del target
    del prediction
    time.sleep(5)

df['IOU'] = iou_scores
print(df)
base=os.getcwd().split('\\')[-1] 
df.to_csv('{}_IOU_score.csv'.format(base))
