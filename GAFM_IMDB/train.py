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
from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils import data
from torch.utils.data import DataLoader
import sys,os
sys.path.append(r"/home/yh579/GAFM/GAFM_IMDB/models")
from GAFM import train_GAFM
from bases import process_data,DataSet,seed_everything,get_logger
from tqdm import trange
import datetime

#######Load Data
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
# raw_df = pd.read_csv('/home/yh579/GAFM/GAFM/data/spam.csv')
# scaler = preprocessing.StandardScaler()
# raw_df.iloc[:,:-1] = pd.DataFrame(scaler.fit_transform(raw_df.iloc[:,:-1]), columns = raw_df.iloc[:,:-1].columns)
# train_loader,test_loader,features=process_data(raw_df)

#IMDB
config = {
    "batch_size":1028
}
from keras.datasets import imdb
def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def get_data(random_state):
    num_words = 500
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words, seed=random_state)
    train_features = vectorize_sequences(train_data, dimension=num_words)
    test_features = vectorize_sequences(test_data, dimension=num_words)
    train_labels = np.asarray(train_labels).astype('float32')
    test_labels = np.asarray(test_labels).astype('float32')
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        train_labels.shape[0], train_labels.sum(), 100 * train_labels.sum() / train_labels.shape[0]))
    features =train_features.shape[-1]
    # print('features',features)
    train_dataset = DataSet(train_features,
                            train_labels.astype(np.float64).reshape(-1, 1))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["batch_size"],
                                               shuffle=True)

    test_dataset = DataSet(test_features,
                           test_labels.astype(np.float64).reshape(-1, 1))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=True)
    return train_loader, test_loader, features



#######Hyperparameters
dataname='IMDB'
best=1
Epochs=300
repeats=10
lr=1e-4#1e-4
sigma=0.01
delta=0.1

#######Store Logs
log_path='/home/yh579/GAFM/GAFM_IMDB/logs'
now = datetime.datetime.now().strftime('%b%d_%H-%M')+dataname+str(repeats)+'Test'
log_dir = f'{log_path}/{now}'
isExist = os.path.exists(log_dir)
if not isExist:
  os.mkdir(log_dir)
logger= open(os.path.join(log_dir, 'log.txt'), 'w')
logger.write('\nDelta:{}\t Epoch={}\t lr={}\t'.format(
                        delta, Epochs, lr))
logger.flush()

#######Training GAFM
print('Training GAFM ...','Delta=',delta)
seed_everything(0)
train_loader,test_loader,features=get_data(0)
train_auc_GAFM, test_auc_GAFM, _, na_leak_auc_GAFM, ma_leak_auc_GAFM, mea_leak_auc_GAFM, _, _, _, _, _, _ = train_GAFM(
    Epochs=Epochs, delta=delta, features=features, train_loader=train_loader, test_loader=test_loader, sigma=sigma,
    regenerate=False, mode='random_fix', lr=lr, info=True, standardization=True, logger=logger)

