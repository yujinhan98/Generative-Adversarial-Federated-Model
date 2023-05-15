import numpy as np
import pandas as pd
import random
from keras.datasets import imdb
from torch.utils import data
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms,datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils import data
import sys,os
sys.path.append(r"/home/yh579/GAFM/GAFM_Criteo/models")
from criteo import CriteoDataset
from GAFM import train_GAFM
from bases import process_data,DataSet,seed_everything,get_logger
from tqdm import trange
import datetime
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler


#Load Data
def get_data(random_state,dataname="Criteo"):
  if dataname=='spam':
    raw_df = pd.read_csv(
      '/home/yh579/GAFM/GAFM/data/spam.csv')  # default of credit card clients.csv
    scaler = preprocessing.StandardScaler()
    raw_df.iloc[:,:-1] = pd.DataFrame(scaler.fit_transform(raw_df.iloc[:,:-1]), columns = raw_df.iloc[:,:-1].columns)
    train_loader, test_loader, features = process_data(raw_df, random_state=random_state)
    features =features.shape[-1]
  elif dataname == 'Criteo':
      config = {
          "batch_size": 256}
      dataset =CriteoDataset('/gpfs/ysm/home/yh579/GAFM/GAFM_Criteo/criteo/dac_sample.txt')

      train_length = int(len(dataset) * 0.8)
      test_length = len(dataset) - train_length
      train_dataset,test_dataset = torch.utils.data.random_split(
          dataset, (train_length, test_length))
      train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=1)
      test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=1)
      features=dataset.field_dims
      # print('train features',features)

  return train_loader, test_loader, features



##Hyper-Parameters
dataname='Criteo'
delta=0.5
best=1
Epochs=100 #90
repeats=3
lr=1e-04#1e-4#1e-4
sigma=0.01
gamma=1
gamma_w=1

#Store Logs
log_path='/home/yh579/GAFM/GAFM_Criteo/logs'
now = datetime.datetime.now().strftime('%b%d_%H-%M')+dataname+str(repeats)+str(lr)+"Epoch"+str(Epochs)+'G'
log_dir = f'{log_path}/{now}'
isExist = os.path.exists(log_dir)
if not isExist:
  os.mkdir(log_dir)
logger= open(os.path.join(log_dir, 'log.txt'), 'w')

# Training GAFM
print('\nTraining GAFM random_fix...','Delta=',delta)
logger.write(
      '\n random seed{:},'.format(
          0))
logger.flush()
seed_everything(0)
train_loader,test_loader,features=get_data(0,dataname=dataname)
train_auc_GAFM_random,test_auc_GAFM_random,_,na_leak_auc_GAFM_random,ma_leak_auc_GAFM_random,mea_leak_auc_GAFM_random,_, _, _, _ ,_,_=train_GAFM(
    Epochs=Epochs,delta=delta,features=features,train_loader=train_loader,test_loader=test_loader,sigma=sigma,regenerate=False,mode='random_fix',lr=lr,info=True, gamma=gamma, gamma_w=gamma_w,standardization=True,logger=logger)
