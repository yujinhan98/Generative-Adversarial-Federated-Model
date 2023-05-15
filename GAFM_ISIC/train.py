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
sys.path.append(r"/home/yh579/GAFM/GAFM_ISIC/models")
from GAFM import train_GAFM
from bases import process_data,DataSet,seed_everything,get_logger
from tqdm import trange
import datetime

# Load Data

def get_data(random_state,dataname="ISIC"):
  if dataname=='spam':
    raw_df = pd.read_csv(
      '/home/yh579/GAFM/GAFM/data/spam.csv')  # default of credit card clients.csv
    scaler = preprocessing.StandardScaler()
    raw_df.iloc[:,:-1] = pd.DataFrame(scaler.fit_transform(raw_df.iloc[:,:-1]), columns = raw_df.iloc[:,:-1].columns)
    train_loader, test_loader, features = process_data(raw_df, random_state=random_state)
    features =features.shape[-1]
  elif dataname == 'ISIC':
    config = {
      "batch_size": 256}#256
    train_transform = transforms.Compose([transforms.Resize((64, 64)),
                                          transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor()
                                          ])

    train_data = datasets.ImageFolder('/home/yh579/GAFM/GAFM_rebuttal/ISIC-2020/train', train_transform)
    test_data = datasets.ImageFolder('/home/yh579/GAFM/GAFM_rebuttal/ISIC-2020/test', train_transform)
    train_loader = data.DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    test_loader = data.DataLoader(test_data, batch_size=config["batch_size"], shuffle=True)
    features=1

  return train_loader, test_loader, features


##Hyper Parameters
dataname='ISIC'
delta=0.05
best=1
Epochs=250
repeats=10
lr=1e-6#1e-4
sigma=0.05
gamma=1
gamma_w=20

##Store Logs
log_path='/home/yh579/GAFM/GAFM_ISIC/logs'
now = datetime.datetime.now().strftime('%b%d_%H-%M')+dataname+str(repeats)+str(lr)+str(Epochs)+'Gamma'+str(gamma)+"Gamma_W"+str(gamma_w)+'Test'
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
train_loader, test_loader, features = get_data(0, dataname=dataname)
train_auc_GAFM_random, test_auc_GAFM_random, _, na_leak_auc_GAFM_random, ma_leak_auc_GAFM_random, mea_leak_auc_GAFM_random, _, _, _, _, _, _ = train_GAFM(
    Epochs=Epochs, delta=delta, features=features, train_loader=train_loader, test_loader=test_loader, sigma=sigma,
    regenerate=False, mode='random_fix', lr=lr, info=True, gamma=gamma, gamma_w=gamma_w, standardization=True,
    logger=logger)
