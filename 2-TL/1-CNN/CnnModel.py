import torch
import torch.nn as nn
import math
import numpy as np


DROPOUT = 0.1
LOCAL_RESPONSE_NORM = 4

class SimpleCNN(torch.nn.Module):
    def __init__(self,use_cuda=False,pooling= False):
        super(SimpleCNN, self).__init__()
        self.use_cuda = use_cuda
        self.pooling = pooling
        self.conv_layer1 =  torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=2)
        self.pool_layer1 = torch.nn.MaxPool2d(kernel_size=2,stride=1)
        self.conv_layer2 = torch.nn.Conv2d(in_channels=16,out_channels=16,kernel_size=5,stride=2)
        self.pool_layer2 = torch.nn.MaxPool2d(kernel_size=2,stride=1)
        # self.dropout1 = torch.nn.Dropout(DROPOUT)
        # self.dropout2 = torch.nn.Dropout(DROPOUT)
        # self.dropout3 = torch.nn.Dropout(DROPOUT)
        
        if pooling:
            self.fully_connected_layer = nn.Linear(64,64)
            self.final_layer = nn.Linear(64,11)
        else:
            self.fully_connected_layer = nn.Linear(1600, 64)
            self.final_layer = nn.Linear(64, 11)
    def forward(self,inp):
        x = torch.nn.functional.relu(self.conv_layer1(inp))
        if self.pooling:
            x = self.pool_layer1(x)
        # x = self.dropout1(x)
        x = torch.nn.functional.relu(self.conv_layer2(x))
        if self.pooling:
            x = self.pool_layer2(x)
        # x = self.dropout2(x)
        x = x.reshape(x.size(0),-1)
        x = torch.nn.functional.relu(self.fully_connected_layer(x))
        # x = self.dropout3(x)
        x = self.final_layer(x)
        return x



class ConvNet(nn.Module):
    
    sample_hparams = {
    'input_image_size':[50, 50],
    'batch_size': 16,
    'epochs': 50,
    'lr': 0.0001,
    'loss_function': nn.CrossEntropyLoss(),
    'optimizer':'adam',
    'optimizer_params':None,
    'num_conv_blocks': 3,
    'num_dense_layers':3,
    'conv_kernel_size':5,
    'pool_kernel_size':2,
    'conv_padding':'valid',
    'pool_padding':0,
    'conv_activations':'relu',
    'dense_activations':'relu',
    'conv_batchnorm':'before',
    'dense_batchnorm':'before',
    'conv_stride':1,
    'pool_stride':1,
    'dense_sizes':[128,64,32],
    'conv_dropout':0.1,
    'dense_dropout':0.1,
    'L2':0.001,
    'num_classes':11,
    'use_cuda':True
    }

    
    def __init__(self, hparams:dict):
        super(ConvNet, self).__init__()
        self.hparams = hparams
        self.layers = []
        self._img_size = self.hparams['input_image_size']
        self._h, self._w = self._img_size
        self._img_size_list = [[3, self._h, self._w]]
        self._actdict = {'relu':nn.ReLU(), 'leakyrelu':nn.LeakyReLU(0.1), 'sigmoid':nn.Sigmoid(), 'tanh':nn.Tanh()}
        self.use_cuda = hparams['use_cuda'] if hparams.get('use_cuda') else False

        # Constructing the encoder (feature extractor)
        in_channels = 3
        out_channels = 16
        for i in range(self.hparams['num_conv_blocks']):
            self.layers.append(nn.Conv2d(in_channels, out_channels, self.hparams['conv_kernel_size'], padding=self.hparams['conv_padding'], stride=self.hparams['conv_stride']))
            self._update_image_size(out_channels, 'conv')
            if self.hparams['conv_batchnorm']=='before':
                # self.layers.append(nn.BatchNorm2d(out_channels))
                self.layers.append(nn.LocalResponseNorm(LOCAL_RESPONSE_NORM))
            self.layers.append(self._actdict[self.hparams['conv_activations']])
            if self.hparams['conv_batchnorm']=='after':
                self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.MaxPool2d(self.hparams['pool_kernel_size'], padding=self.hparams['pool_padding'], stride=self.hparams['pool_stride']))
            self._update_image_size(out_channels, 'pool')
            if self.hparams.get('conv_dropout'):
                self.layers.append(nn.Dropout2d(self.hparams['conv_dropout']))
            if i < self.hparams['num_conv_blocks'] - 1:
                in_channels = out_channels
                out_channels *= 2

        # Flattening (Image embedding)
        self.layers.append(nn.Flatten())

        # Constructing the decoder (classifier)
        if self.hparams.get('dense_sizes'):
            size_vec = self.hparams.get('dense_sizes')
        else:
            size_vec = [0] * self.hparams['num_dense_layers']
            size_vec[-1] = np.power(2, np.ceil(np.log2(self.hparams['num_classes']))).astype(int)
            for i in range(self.hparams['num_dense_layers']-2, -1, -1):
                size_vec[i] = size_vec[i+1] * 2
        in_size = out_channels*self._h*self._w
        out_size = size_vec[0]
        for i in range(len(size_vec)):
            self.layers.append(nn.Linear(in_size, out_size))
            if self.hparams['dense_batchnorm'] == 'before':
                self.layers.append(nn.BatchNorm1d(out_size))
                # self.layers.append(nn.LocalResponseNorm(LOCAL_RESPONSE_NORM))
            self.layers.append(self._actdict[self.hparams['dense_activations']])
            if self.hparams['dense_batchnorm'] == 'after':
                self.layers.append(nn.BatchNorm1d(out_size))
            if self.hparams.get('dense_dropout'):
                self.layers.append(nn.Dropout(self.hparams['dense_dropout']))
            if i < len(size_vec) - 1:
                in_size = out_size
                out_size = size_vec[i+1]
        self.layers.append(nn.Linear(out_size, self.hparams['num_classes']))

        # Constructing model
        self.net = nn.Sequential(*self.layers)

                
    def _calc_size(self, size_in:int, padding:int, kernel_size:int, stride:int):
        if padding == 'valid':
            padding=0
        if padding=='same':
            return size_in
        else:
            return math.floor((size_in + 2*padding - (kernel_size-1) - 1)/stride + 1)
        
    
    def _update_image_size(self, out_channels, ops:str='conv'):
        (self._h, self._w) = (self._calc_size(sz, self.hparams[ops+'_padding'], self.hparams[ops+'_kernel_size'], self.hparams[ops+'_stride']) for sz in (self._h,self._w))
        self._img_size = (self._h, self._w)
        #print("new size: ",self._img_size)
        self._img_size_list.append([out_channels, self._h, self._w])

    def forward(self, x:torch.Tensor):
        return self.net(x)