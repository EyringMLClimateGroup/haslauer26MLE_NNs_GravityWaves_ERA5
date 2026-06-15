# TRAINING SCRIPT
# Training of NNs for GWMFs on ERA5 data


# LIBRARIES

import numpy as np

import datetime as dt
from tqdm import tqdm

import subprocess as sp

import torch
from torch import nn

from torchinfo import summary

# import torch.nn.functional as F
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torcheval.metrics import R2Score
# import torch.optim.lr_scheduler

from _utils import __ml_networks as mln


device = (
    "cuda"
    if torch.cuda.is_available()
      else "cpu"
)


####################################################################################################
## SETTINGS ########################################################################################


# DIRECTORIES TO LOAD PREPROCESSED DATA

path_preprocessed = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML_new/_n32/IG/_preprocessed_full2024_land_/'

path_training = path_preprocessed + '_train_.npz'
path_testing = path_preprocessed + '_test_.npz'


# SELECT NETWORK TYPE AND PARAMETERS

mode = 'UNet_2'       # available: FCNN, CNN, UNet_2

# General parameters
epochs = 40
batch_size = 512
activation = 'LeakyReLU'
learning_rate = 0.0001
scheduler_type = None 
l1_lambda = 0.0             # 1 * 10e-6
l2_lambda = 0.0             # 1.0 * 10e-5
weight_dec = 1.0 * 10e-5
bn = False                  # Batch normalization
dr = 0.0                    # Dropout percentage

# FCNN parameters
n_layers = 2
n_hidden = 1000

# CNN & U-Net parameters
levels = 37
vector_feat = 4
scalar_feat = 5
fc_hidden = 128

# U-Net only parameters
base_filters = 128

# Evaluation step
frequ_stat = 1


# Model path
path_save = path_preprocessed + '__models/' + mode + '_2026-04-15i_UNet_2_IG_land.pt'
sp.run(['mkdir -p ' + path_preprocessed + '__models/'], shell=True)

####################################################################################################


# LOAD DATA

print(path_preprocessed)

with np.load(path_training) as data:
    X_train, y_train = torch.tensor(data['feat']), torch.tensor(data['targ'])

with np.load(path_testing) as data:
    X_test, y_test = torch.tensor(data['feat']), torch.tensor(data['targ'])


# DATASET & DATALOADER

n_feat = X_train.size()[1]
n_targ = y_train.size()[1]
    
train_dataset = mln.FluxData(X_train, y_train)
test_dataset = mln.FluxData(X_test, y_test)


# X_train.size(), y_train.size(), X_test.size(), y_test.size()


# DEFINE TRAINING FUNCTION

def train_NN(mode = 'FCNN', epochs=10, batch_size=1024, activation='ReLU', learning_rate=0.00001,
             scheduler_type=None,  l1_lambda=0.0, l2_lambda=0.0, bn=False, weight_dec=0.0, dr=0.0,
             frequ_stat=1, n_layers=0, n_hidden=0, levels=0,
             vector_feat=0, scalar_feat=0, fc_hidden=0, base_filters=64):

    if mode == 'FCNN':
        model = mln.FCNN(in_dim=n_feat, out_dim=n_targ, n_hidden=n_hidden, n_layers=n_layers, bn=bn, dr=dr).to(device)
    
    elif mode == 'CNN':
        model = mln.ConvNet(vec_f=vector_feat, sca_f=scalar_feat, levels=levels, fc_hidden=fc_hidden).to(device)

    elif mode == 'UNet_2':
        model = mln.UNet_2(vec_f=vector_feat, sca_f=scalar_feat, fc_hidden=fc_hidden, base_filters=base_filters, activation=activation, bn=bn).to(device)

    train_load, test_load = mln.load_data(train_dataset, test_dataset, batch_size)
    

    loss_fn = nn.MSELoss() # nn.MSELoss(), nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_dec)
    
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    elif scheduler_type == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_load))


    now = dt.datetime.now()
    print(now.strftime('%d/%m/%Y  %H:%M:%S') + '\n')
    print(f'batch_size  = {batch_size:6} \nn_layers    = {n_layers:6}\nn_neurons   = {n_hidden:6} \nlr          =  {learning_rate} \nepochs      = {epochs:6}\n--------------------\n\n')
    print(summary(model))
    print('\n\n')

    if scheduler_type is not None:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.25, total_iters=epochs)

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        desc='Epoch ' + str(epoch+1).zfill(2)
        for batch, (X, y) in tqdm(enumerate(train_load), desc=desc, total=len(train_load)):
            
            X, y = X.to(device), y.to(device)
            pred = model(X)


            # Regularization

            if l1_lambda > 0:
                l1_regularization = 0.
                for param in model.parameters():
                    l1_regularization += l1_lambda * param.abs().sum()
            else: l1_regularization = 0

            if l2_lambda > 0:
                l2_regularization = 0.
                for param in model.parameters():
                    l2_regularization += l2_lambda * (param**2).sum()
            else: l2_regularization = 0

            loss = loss_fn(pred, y) + l1_regularization + l2_regularization
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler_type is not None: scheduler.step()

        if (epoch+1) % frequ_stat == 0 or epoch == epochs: 
            print('Epoch ' + str(epoch+1).zfill(2) + ':')
            mln.eval(model, train_dataset, test_dataset, mode=mode, levels=levels, vector_feat=vector_feat, scalar_feat=scalar_feat, fc_hidden=fc_hidden)
        
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler.step(train_loss/len(train_load))
    elif scheduler_type == 'OneCycleLR':
        scheduler.step()


    return model


# START TRAINING

TRAINING = True

if TRAINING is True:

    model_trained = train_NN(mode = mode, epochs=epochs, batch_size=batch_size, activation=activation, learning_rate=learning_rate,
                             scheduler_type=scheduler_type, l1_lambda=l1_lambda, l2_lambda=l2_lambda, weight_dec=weight_dec, bn=bn, dr=dr,
                             frequ_stat=frequ_stat, n_layers=n_layers, n_hidden=n_hidden, levels=levels,
                             vector_feat=vector_feat, scalar_feat=scalar_feat, fc_hidden=fc_hidden, base_filters=base_filters)
    
    mln.eval(model_trained, train_dataset, test_dataset)

    torch.save(model_trained, path_save)
    print('Model saved: ' + path_save)