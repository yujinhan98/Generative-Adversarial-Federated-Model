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
import sys, os
np.set_printoptions(threshold=np.inf)
sys.path.append(r"/home/yh579/GAFM/GAFM/models")
from bases import FirstNet, SecondNet, torch_auc, totalvaraition, Attacks
# SplitNN
import torch
sys.path.append(r"/home/yh579/GAFM/GAFM_ISIC/models")
from Marvell import KL_gradient_perturb_function_creator
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_printoptions(precision=10, threshold=1000000)
# device='cpu'
hidden_dim = 10


class DisNet(nn.Module):
    def __init__(self):
        super(DisNet, self).__init__()
        self.L1 = nn.Linear(1,
                            hidden_dim)
        self.L2 = nn.Linear(hidden_dim,
                            hidden_dim)
        self.L3 = nn.Linear(hidden_dim,
                            1)

    def forward(self, x):
        x = self.L1(x)
        x = nn.functional.leaky_relu(x)
        x = self.L3(x)
        x = nn.functional.leaky_relu(x)  # ,negative_slope=3 ,negative_slope=10

        return x


# def addeNoise(sigma, Y):
#     # noise = np.random.uniform(0,1,N)
#     noise = np.random.normal(0, sigma, Y.shape[0])
#     noise = noise + Y
#     return torch.Tensor(noise).reshape(-1, 1)


from torch.autograd import Variable


def GAN_pertub(grad_recon, discriminator, server):
    b = grad_recon
    b.requires_grad = True
    b = Variable(b, requires_grad=True)
    discriminator.eval()
    server.eval()
    z = -discriminator(server(b))  # nn.functional.leaky_relu(params[0][0]*b+params[1])
    z.sum().backward()

    final_grad = b.grad.clone().detach()  # .reshape(-1,1)
    final_grad = torch.where(
        torch.isnan(final_grad),
        torch.full_like(final_grad, 0),
        final_grad)

    return final_grad


def Pen_pertub(grad_recon, labels, delta=0.1, Y_dot=None):
    b = grad_recon
    b.requires_grad = True
    b = Variable(b, requires_grad=True)
    labels = Y_dot
    # if Y_dot is not None:
    #     labels = Y_dot  # torch.abs(labels-0.5+delta)
    # else:
    #     labels = torch.abs(labels - 0.5 + delta)
    # z = -(labels * torch.log(b) + (1 - labels) * torch.log(1 - b))
    z = -(labels * torch.log(torch.sigmoid(b.sum(dim=1))) + (1 - labels) * torch.log(1 - torch.sigmoid(b.sum(dim=1))))

    z.sum().backward()
    final_grad = b.grad.clone().detach()  #
    final_grad = torch.where(
        torch.isnan(final_grad),
        torch.full_like(final_grad, 0),
        final_grad)

    return final_grad


class Client_GAFM(torch.nn.Module):
    def __init__(self, client_model):
        super().__init__()
        """class that expresses the Client on SplitNN
        Args:
            client_model (torch model): client-side model
        Attributes:
            client_model (torch model): cliet-side model
            client_side_intermidiate (torch.Tensor): output of
                                                     client-side model
            grad_from_server
        """

        self.client_model = client_model
        self.client_side_intermidiate = None
        self.grad_from_server = None

    def forward(self, inputs):
        """client-side feed forward network
        Args:
            inputs (torch.Tensor): the input data
        Returns:
            intermidiate_to_server (torch.Tensor): the output of client-side
                                                   model which the client sent
                                                   to the server
        """

        self.client_side_intermidiate = self.client_model(inputs)
        # send intermidiate tensor to the server
        intermidiate_to_server = self.client_side_intermidiate.detach().requires_grad_()

        return intermidiate_to_server

    def client_backward(self, grad_from_server):
        """client-side back propagation
        Args:
            grad_from_server: gradient which the server send to the client
        """
        self.grad_from_server = grad_from_server
        self.client_side_intermidiate.backward(grad_from_server)

    def train(self):
        self.client_model.train()

    def eval(self):
        self.client_model.eval()


class Server_GAFM(torch.nn.Module):
    def __init__(self, server_model):
        super().__init__()
        """class that expresses the Server on SplitNN
        Args:
            server_model (torch model): server-side model
        Attributes:
            server_model (torch model): server-side model
            intermidiate_to_server:
            grad_to_client
        """
        self.server_model = server_model

        self.intermidiate_to_server = None
        self.grad_to_client = None
        # self.intermidiate_to_server_pertub=None

    def forward(self, intermidiate_to_server):
        """server-side training
        Args:
            intermidiate_to_server (torch.Tensor): the output of client-side
                                                   model
        Returns:
            outputs (torch.Tensor): outputs of server-side model
        """
        self.intermidiate_to_server = intermidiate_to_server
        outputs = self.server_model(intermidiate_to_server)

        return outputs

    def server_backward(self):
        self.grad_to_client = self.intermidiate_to_server.grad.clone()
        return self.grad_to_client

    def train(self):
        self.server_model.train()

    def eval(self):
        self.server_model.eval()


class SplitNN_GAFM(torch.nn.Module):
    def __init__(self, client, server, discriminator,
                 client_optimizer, server_optimizer, discriminator_optimizer
                 ):
        super().__init__()
        """class that expresses the whole architecture of SplitNN
        Args:
            client (attack_splitnn.splitnn.Client):
            server (attack_splitnn.splitnn.Server):
            clietn_optimizer
            server_optimizer
        Attributes:
            client (attack_splitnn.splitnn.Client):
            server (attack_splitnn.splitnn.Server):
            clietn_optimizer
            server_optimizer
        """
        self.client = client
        self.server = server
        self.client_optimizer = client_optimizer
        self.server_optimizer = server_optimizer
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.intermidiate_to_server = None
        self.intermidiate_to_server_pertub = None
        self.grad_to_client = None
        self.grad_to_client_recon = None
        self.epoch = None

    def forward(self, inputs, labels, delta, Y_dot, gamma, gamma_w):
        # execute client - feed forward network
        self.intermidiate_to_server = self.client(inputs)
        self.labels = labels
        self.Y_dot = Y_dot
        self.delta = delta
        self.gamma = gamma
        self.gamma_w = gamma_w

        # execute server - feed forward netwoek
        # Decoder
        outputs = self.server(self.intermidiate_to_server)
        return outputs, self.intermidiate_to_server, self.discriminator

    def backward(self, standardization):
        self.intermidiate_to_server_pertub = GAN_pertub(self.intermidiate_to_server, self.discriminator,
                                                        self.server)

        self.grad_to_client_recon = Pen_pertub(self.intermidiate_to_server, self.labels, self.delta, self.Y_dot)
        if standardization:
            self.intermidiate_to_server_pertub = self.intermidiate_to_server_pertub / (
                self.intermidiate_to_server_pertub.pow(2).sum().sqrt())
            self.grad_to_client_recon = self.grad_to_client_recon / (self.grad_to_client_recon.pow(2).sum().sqrt())

        # print('self.gamma_w',self.gamma_w,'self.gamma',self.gamma)
        self.intermidiate_to_server_pertub = self.gamma_w * self.intermidiate_to_server_pertub
        self.grad_to_client_recon = self.gamma * (
            self.grad_to_client_recon)
        self.grad_to_client = self.grad_to_client_recon + self.intermidiate_to_server_pertub

        self.client.client_backward(self.grad_to_client)

    def zero_grads(self):
        self.client_optimizer.zero_grad()
        self.server_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def step(self):
        self.client_optimizer.step()
        self.server_optimizer.step()
        # self.discriminator_optimizer.step()

    def train(self):
        self.client.train()
        self.server.train()
        self.discriminator.train()

    def eval(self):
        self.client.eval()
        self.server.eval()
        self.discriminator.eval()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_GAFM(Epochs, features, train_loader, test_loader, gamma=1, gamma_w=1, sigma=0.01, delta=0.1, lr=1e-3,
               info=False, standardization=True,
               regenerate=False, mode='random_fix', logger=None,grads=False):
    '''
    :param Epochs:
    :param gamma: weights of penalty
    :param gamma_w:
    :param sigma:
    :param delta:
    :param lr:
    :param info:
    :param standardization:
    :param regenerate:
    :param mode: ['norandom','random_fix','random_B']
    :return:
    '''
    # clone=False
    logger.write('Start Training GAFM')
    logger.flush()
    logger.write(
        '\nGamma={:.3f}\t Gamma_w={:.3f},delta={:.3f}\t lr{:},'.format(
            gamma, gamma_w, delta, lr))
    logger.flush()
    # clear_output()
    clip = True
    clip_value = 0.1
    input_dim = features  # .shape[-1]
    model_1 = FirstNet(input_dim)
    model_1 = model_1.to(device)

    model_2 = SecondNet()
    model_2 = model_2.to(device)

    model_1.double()
    model_2.double()

    opt_1 = optim.Adam(model_1.parameters(), lr=lr)
    opt_2 = optim.Adam(model_2.parameters(), lr=lr)

    BCE = nn.BCELoss()  # nn.BCEWithLogitsLoss()#nn.BCELoss()

    client = Client_GAFM(model_1)
    server = Server_GAFM(model_2)

    discriminator = DisNet()
    discriminator = discriminator.to(device)
    discriminator.double()
    opt_D = optim.Adam(discriminator.parameters(), lr=lr)
    splitnn_GAFM = SplitNN_GAFM(client, server, discriminator, opt_1, opt_2, opt_D)

    splitnn_GAFM.train()
    train_auc_list = []
    grad_gan = []
    grad_recon = []
    grads = []
    training_labels = []
    outputs_list = []
    for epoch in range(Epochs):
        epoch_loss = 0
        epoch_outputs = []
        epoch_intermediates = []
        epoch_labels = []
        epoch_outputs_test = []
        epoch_intermediates_test = []
        epoch_labels_test = []
        epoch_g_norm = []
        epoch_g_mean = []
        epoch_g_inner = []
        epoch_g = []

        for i, data in enumerate(train_loader):
            splitnn_GAFM.zero_grads()
            inputs, labels = data
            inputs = inputs.to(device).long()
            labels = labels.to(device).double()
            Y_ = torch.normal(0, sigma, size=(labels.shape)).to(
                device).double() + labels

            if mode == 'norandom':
                Y_dot = None
            elif mode == 'random_fix':
                deltas = delta * torch.rand(labels.shape).to(device).double()
                Y_dot = (torch.abs(labels - 0.5 + deltas))  # .to(device)
            else:
                Y_dot = torch.bernoulli(torch.abs(labels - 0.5 + delta))



            outputs, intermidiate_to_server, discriminator_model = splitnn_GAFM(inputs, labels, delta, Y_dot, gamma,
                                                                                gamma_w)


            loss_D = gamma_w * (-torch.mean(splitnn_GAFM.discriminator(Y_.view(-1, 1))) + torch.mean(
                splitnn_GAFM.discriminator(outputs.detach())))  # torch.mean
            loss_D.backward()
            splitnn_GAFM.discriminator_optimizer.step()

            if clip:
                for p in splitnn_GAFM.discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)
               
            splitnn_GAFM.server_optimizer.zero_grad()
            splitnn_GAFM.discriminator.eval()
            loss_D_1 = -gamma_w * torch.mean(splitnn_GAFM.discriminator(outputs))

            # print('torch.abs(labels - 0.5 + delta)', torch.abs(labels - 0.5 + delta).view(-1,1).shape)
            # print('intermidiate_to_server', intermidiate_to_server.shape)
            # intermidiate_to_server=(intermidiate_to_server-intermidiate_to_server.min())/(intermidiate_to_server.max()-intermidiate_to_server.min())

            # loss_recon = gamma * BCE(intermidiate_to_server.view(-1,1), torch.abs(labels - 0.5 + delta).view(-1,1))
            # print('torch.sigmoid(intermidiate_to_server.sum(dim=1))',torch.sigmoid(intermidiate_to_server.sum(dim=1)).view(-1, 1))
            intermidiate_to_server=torch.sigmoid(intermidiate_to_server.sum(dim=1))
            loss_recon = gamma * BCE(intermidiate_to_server.view(-1, 1),
                                     Y_dot.unsqueeze(-1).type_as(intermidiate_to_server))
            # loss_recon = gamma * BCE(intermidiate_to_server, torch.abs(labels - 0.5 + delta).unsqueeze(-1).type_as(intermidiate_to_server))
            # loss_recon = gamma * BCE(outputs.detach().view(-1, 1), torch.abs(labels - 0.5 + delta).view(-1, 1))
            loss = loss_recon + loss_D_1  # +loss_class
            # print('loss==gamma*loss_recon',loss==gamma*loss_recon)
            loss.backward()
            splitnn_GAFM.backward(standardization)
            splitnn_GAFM.step()
            gamma_w = splitnn_GAFM.gamma_w

            epoch_loss += loss.item() / len(train_loader.dataset)
            epoch_outputs.append(outputs)
            epoch_labels.append(labels)

            grad_from_server = splitnn_GAFM.grad_to_client
            g_norm = grad_from_server.pow(2).sum(dim=1).sqrt()
            v_1 = np.multiply(grad_from_server.sum(dim=1).cpu().detach().numpy(), labels.cpu().detach().numpy())
            mean_1 = v_1.sum() / len(v_1[v_1 != 0])
            mean_0 = (grad_from_server.cpu().detach().numpy().sum() -
                      v_1.sum()) / len(v_1[v_1 == 0])
            # print('labels.expand(grad_from_server.shape)',labels.view(-1, 1).expand(grad_from_server.shape))
            # v_1 = grad_from_server.mul(labels.view(-1, 1).expand(grad_from_server.shape))
            # v_1_=v_1.sum(dim=1)
            # mean_1 = v_1.sum(dim=0) / len(v_1_[v_1_ != 0])
            # mean_0 = (grad_from_server.sum(dim=0) -
            #           v_1.sum(dim=0)) / len(v_1_[v_1_ == 0])
            # print('mean_1',mean_1)
            # print('mean_0', mean_0)

            # g = list(grad_from_server)
            g = list(grad_from_server.sum(dim=1).cpu().detach().numpy())
            g_mean = []
            for a in g:
                # print('a',a)
                # if torch.sqrt(torch.sum((a - mean_1) ** 2)) < torch.sqrt(torch.sum((a - mean_0) ** 2)):
                if (a - mean_1) ** 2 < (a - mean_0) ** 2:
                    g_mean.append([1])
                else:
                    g_mean.append([0])
            g_mean = torch.tensor(g_mean)
            g = list(grad_from_server.sum(dim=1).cpu().detach().numpy())
            g_inner = []
            for a in g:
                if a > grad_from_server.sum(dim=1).median().item():
                    g_inner.append(1)
                else:
                    g_inner.append(0)
            g_inner = torch.tensor(g_inner)

            epoch_g_norm.append(g_norm)
            epoch_g_mean.append(g_mean)
            epoch_g_inner.append(g_inner)
            epoch_g.append(grad_from_server.sum(dim=1))

            t = next(iter(test_loader))
            # t[1] = torch.tensor(t[1], dtype=torch.float32)
            outputs_test, intermidiate_to_server_test, _ = splitnn_GAFM(t[0].to(device).long(),
                                                                        t[1].to(device).double(), delta, Y_dot, gamma,
                                                                        gamma_w)
            labels_test = t[1].to(device).double()
            # print('labels_test',labels_test)
            epoch_outputs_test.append(outputs_test)
            epoch_labels_test.append(labels_test)
            epoch_intermediates.append(intermidiate_to_server)
            epoch_intermediates_test.append(torch.sigmoid(intermidiate_to_server_test.sum(dim=1)))

        if gamma_w == 0:
            train_auc = torch_auc(torch.cat(epoch_labels),
                                  torch.cat(epoch_outputs))
            test_auc = torch_auc(torch.cat(epoch_labels_test),
                                 torch.cat(epoch_outputs_test))

        else:
            train_auc = torch_auc(torch.cat(epoch_labels),
                                  torch.cat(epoch_outputs))
            test_auc = torch_auc(torch.cat(epoch_labels_test),
                                 torch.cat(epoch_outputs_test))
        train_tvd = totalvaraition(torch.cat(epoch_labels),
                                   torch.cat(epoch_g))
        na_leak_auc = max(torch_auc(torch.cat(epoch_labels), torch.cat(epoch_g_norm).view(-1, 1)),
                          1 - torch_auc(torch.cat(epoch_labels),
                                        torch.cat(epoch_g_norm).view(-1, 1)))
        ma_leak_auc = max(torch_auc(torch.cat(epoch_labels), torch.cat(epoch_g_mean).view(-1, 1)),
                          1 - torch_auc(torch.cat(epoch_labels),
                                        torch.cat(epoch_g_mean).view(-1, 1)))
        cos_leak_auc = max(torch_auc(torch.cat(epoch_labels), torch.cat(epoch_g_inner).view(-1, 1)),
                           1 - torch_auc(torch.cat(epoch_labels),
                                         torch.cat(epoch_g_inner).view(-1, 1)))
        train_auc_list.append(train_auc)
        grad_gan.append(splitnn_GAFM.intermidiate_to_server_pertub.sum(dim=1))
        grad_recon.append(splitnn_GAFM.grad_to_client_recon.sum(dim=1))
        grads.append(splitnn_GAFM.grad_to_client.sum(dim=1))
        training_labels.append(labels)
        outputs_list.append(outputs)

        if (epoch % 1 == 0 or epoch == Epochs - 1):
            logger.write(
                '\nEpoch:{}\t Training AUC={:.3f}\t Testing AUC={:.3f},TVD={:.3f}\t NA={:.3f},MA={:.3f}\t MeA={:.3f}'.format(
                    epoch, train_auc, test_auc, train_tvd, na_leak_auc, ma_leak_auc, cos_leak_auc))
            logger.flush()
            # clear_output()
        if (epoch % 1 == 0 or epoch == Epochs - 1):
            print('Gamma', gamma, "Discriminator", -loss_D.item(),
                  "Generator", loss_D_1.item(),
                  'Epoch', epoch, 'Training Loss', epoch_loss,
                  'Training AUC', train_auc,
                  'Testing AUC', test_auc,
                  "TVD", train_tvd,
                  'NA Leak AUC', na_leak_auc,
                  'MA Leak AUC', ma_leak_auc,
                  'Median Leak AUC', cos_leak_auc
                  )
        if (grads and epoch == Epochs - 1):
            print('training_labels:', training_labels[-1].flatten(),
                  "\noutputs_list:", outputs_list[-1].flatten(),
                  "\n grad_gan:", grad_gan[-1].flatten(),
                  "\n grad_recon:", grad_recon[-1].flatten()
                  )
            logger.write(
                '\ntraining_labels:{}\n outputs_list:{}\n grads:{}\n grad_gan:{}\n grad_recon:{} '.format(
                    training_labels[-1].flatten(), outputs_list[-1].flatten(), grads[-1].flatten(),
                    grad_gan[-1].flatten(), grad_recon[-1].flatten()))
            logger.flush()
            print('training_labels2:', training_labels[-2].flatten(),
                  "\noutputs_list2:", outputs_list[-2].flatten(),
                  "\n grad_gan2:", grad_gan[-2].flatten(),
                  "\n grad_recon2:", grad_recon[-2].flatten()
                  )
            logger.write(
                '\ntraining_labels:{}\n outputs_list:{}\n grads:{}\n grad_gan:{}\n grad_recon:{} '.format(
                    training_labels[-2].flatten(), outputs_list[-2].flatten(), grads[-2].flatten(),
                    grad_gan[-2].flatten(), grad_recon[-2].flatten()))
            logger.flush()

    return train_auc, test_auc, train_tvd, na_leak_auc, ma_leak_auc, cos_leak_auc, splitnn_GAFM, grad_gan, grad_recon, grads, training_labels, outputs_list

    # return train_auc, test_auc, train_tvd, na_leak_auc, ma_leak_auc, cos_leak_auc, splitnn_GAFM, grad_gan, grad_recon, grads

