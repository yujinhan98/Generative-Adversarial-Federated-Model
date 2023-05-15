##Norm Attack
import torch
import sys,os
import pretrainedmodels
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
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
from abc import ABCMeta, abstractmethod
from torchvision import models
from efficientnet_pytorch import EfficientNet

import logging
import lightning.pytorch as pl
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger



def seed_everything(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

class AbstractAttacker(metaclass=ABCMeta):
    def __init__(self, splitnn):
        """attacker against SplitNN
        Args:
            splitnn: SplitNN
        """
        self.splitnn = splitnn

    def fit(self):
        pass

    @abstractmethod
    def attack(self):
        pass


class NormAttack(AbstractAttacker):
    def __init__(self, splitnn):
        """Class that implement normattack
        Args:
            splitnn (attack_splitnn.splitnn.SplitNN): target splotnn model
        """
        super().__init__(splitnn)
        self.splitnn = splitnn

    def attack(self, dataloader, criterion, device):
        """Culculate leak_auc on the given SplitNN model
           reference: https://arxiv.org/abs/2102.08504
        Args:
            dataloader (torch dataloader): dataloader for evaluation
            criterion: loss function for training
            device: cpu or GPU
        Returns:
            score: culculated leak auc
        """
        epoch_labels = []
        epoch_g_norm = []
        for i, data in enumerate(dataloader, 0):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = self.splitnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            self.splitnn.backward()

            grad_from_server = self.splitnn.client.grad_from_server
            g_norm = grad_from_server.pow(2).sum(dim=1).sqrt()
            epoch_labels.append(labels)
            epoch_g_norm.append(g_norm)

        epoch_labels = torch.cat(epoch_labels)
        epoch_g_norm = torch.cat(epoch_g_norm)
        score = roc_auc_score(epoch_labels, epoch_g_norm.view(-1, 1))
        return score

from torch.utils.data.dataset import Dataset
class DataSet(Dataset):
    """This class allows you to convert numpy.array to torch.Dataset
    Args:
        x (np.array):
        y (np.array):
        transform (torch.transform):
    Attriutes
        x (np.array):
        y (np.array):
        transform (torch.transform):
    """

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        """get the number of rows of self.x
        """
        return len(self.x)


def torch_auc(label, pred):
    return roc_auc_score(label.cpu().detach().numpy(),
                         pred.cpu().detach().numpy())



def process_data(raw_df,random_state,size=0.3,batch_size=1028):
    config = {
        "batch_size": batch_size
    }
    # Use a utility from sklearn to split and shuffle our dataset.
    train_df, test_df = train_test_split(raw_df, test_size=size,random_state=random_state)
    # train_df, val_df = train_test_split(train_df, test_size=0.2)

    # Form np arrays of labels and features.
    train_labels = np.array(train_df['label'])
    bool_train_labels = train_labels != 0
    # val_labels = np.array(val_df)
    test_labels = np.array(test_df['label'])

    train_features = np.array(train_df.drop(['label'], axis=1))
    test_features = np.array(test_df.drop(['label'], axis=1))
    print('Training labels shape:', train_labels.shape)
    print('Test labels shape:', test_labels.shape)

    print('Training features shape:', train_features.shape)
    print('Test features shape:', test_features.shape)
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
    return train_loader,test_loader,train_features

from sklearn.neighbors import KernelDensity
def totalvaraition(label,g):
    label=pd.DataFrame(label.cpu().detach().numpy())
    g=list(g.cpu().detach().numpy())
    label_ = label.copy()
    label_.reset_index(inplace=True, drop=True)
    a = list(label_.iloc[:, 0])
    ind_1 = [i for i, j in enumerate(a) if j == 1]
    g_1 = np.array([j for i, j in enumerate(g) if i in ind_1])#g.index(ind_1)
    g_0 = np.array(np.delete(g, ind_1).tolist())
    a=min(np.min(g_1),np.min(g_0))
    b=max(np.max(g_1),np.max(g_0))

    g_ = np.linspace(a,b,100)#np.arange(a,b,100)#linespace
    model_1 = KernelDensity(bandwidth=0.0001)
    model_1.fit(g_1.reshape(-1, 1))
    dens_1 =  np.exp(model_1.score_samples(g_.reshape(-1, 1)))

    model_0 = KernelDensity(bandwidth=0.0001)
    model_0.fit(g_0.reshape(-1, 1))
    dens_0 = np.exp(model_0.score_samples(g_.reshape(-1, 1)))

    return 0.5 * np.sum(np.abs(dens_0 - dens_1)*(b-a)/100)#0.5 * np.sum(np.abs(dens_0 - dens_1)*(b-a)/100)

hidden_dim = 16#16





class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias  # output size (batch_size, num_fields)

class FeaturesEmbedding(torch.nn.Module):
        def __init__(self, field_dims, embed_dim):
            super().__init__()
            self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
            self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
            torch.nn.init.xavier_uniform_(self.embedding.weight.data)

        def forward(self, x):
            """
            :param x: Long tensor of size ``(batch_size, num_fields)``
            """
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
            return self.embedding(x)  # output size (batch_size, num_fields, embed_dim)


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, hidden_layers, dropout, output_layer=True):  # input_dim = 64, embed_dim = (64,)
        super().__init__()
        layers = list()
        for hidden_layer_size in hidden_layers:  # for i in (64,) print i ---> 64
            layers.append(torch.nn.Linear(input_dim, hidden_layer_size))
            # generally put batchnorm after linearization and before activation
            layers.append(torch.nn.BatchNorm1d(hidden_layer_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = hidden_layer_size
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class FirstNet(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2):
        super().__init__()
        # print('First field',field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1).view(-1, 1)




class SecondNet(nn.Module):
    def __init__(self):
        super(SecondNet, self).__init__()
        self.L1 = nn.Linear(1,
                            hidden_dim)
        self.L2 = nn.Linear(hidden_dim,
                            1)

    def forward(self, x):
        x = self.L1(x)
        x = nn.functional.leaky_relu(x)
        x = self.L2(x)
        x = torch.sigmoid(x)
        return x
    
def Attacks(grad_from_server,labels):
  g=list(grad_from_server.detach().numpy())
  g_norm = grad_from_server.pow(2).sum(dim=1).sqrt()
  v_1=np.multiply(grad_from_server.detach().numpy(),labels.detach().numpy())
  mean_1=v_1.sum()/len(v_1[v_1!=0])
  mean_0=(grad_from_server.detach().numpy().sum()-
               v_1.sum())/len(v_1[v_1==0])
  g_mean=[]
  for a in g:
    if (a-mean_1)**2<(a-mean_0)**2:g_mean.append(1)
    else:g_mean.append(0)
  g_mean=torch.tensor(g_mean)
  g_inner=[]
  for a in g:
    if a>0:g_inner.append(1)
    else:g_inner.append(0)
  g_inner=torch.tensor(g_inner)
  return g_norm,g_mean,g_inner
