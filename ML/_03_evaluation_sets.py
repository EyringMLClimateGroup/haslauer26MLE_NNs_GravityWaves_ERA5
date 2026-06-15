# EVALUATION OF NEURAL NETWORK PREDICTIONS ON DIFFERENT SETS

# Import of libraries

import sys

import numpy as np
import xarray as xr
import torch

import datetime as dt

sys.path.append('/p/project1/icon-a-ml/haslauer1/gravity_waves/__git/haslauer23_gravitywaves_parameterization/ML/_utils/')
from _utils import __ml_networks as mln

device = (
    "cuda"
    if torch.cuda.is_available()
      else "cpu"
)


# Definition of directories of NNs and training / test sets

igtype = 'IG'

nn_path = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML_new/_n32/IG/_preprocessed_full2024_land_/__models/UNet_2_2026-04-15_UNet_2_IG_land.pt'
model = torch.load(nn_path, map_location=torch.device(device), weights_only=False)


data_path = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML_new/_n32/' + igtype + '/_preprocessed_full2024_land_/'

train      = data_path + '_train_.npz'

test      = data_path + '_test_1_.npz'
test_orog = data_path + '_test_1_orog.npz'
test_flat = data_path + '_test_1_flat.npz'
#test_high = data_path + '_test_1_STD_HIGH.npz'


# Testing function

def test_data(model, path_test, path_test_orog, path_test_flat):#, path_train, path_train_orog, path_train_flat):
    with np.load(path_test) as data:
        X_test, y_test = torch.tensor(data['feat']), torch.tensor(data['targ'])
    with np.load(path_test_orog) as data:
        X_test_orog, y_test_orog = torch.tensor(data['feat']), torch.tensor(data['targ'])
    with np.load(path_test_flat) as data:
        X_test_flat, y_test_flat = torch.tensor(data['feat']), torch.tensor(data['targ'])
    # with np.load(path_test_high) as data:
    #     X_test_high, y_test_high = torch.tensor(data['feat']), torch.tensor(data['targ'])


    test_ds      = mln.FluxData(X_test, y_test)
    test_orog_ds = mln.FluxData(X_test_orog, y_test_orog)
    test_flat_ds = mln.FluxData(X_test_flat, y_test_flat)
    #test_high_ds = mln.FluxData(X_test_high, y_test_high)


    r2_test      = mln.eval_1_set(model, test_ds)
    r2_test_orog = mln.eval_1_set(model, test_orog_ds)
    r2_test_flat = mln.eval_1_set(model, test_flat_ds)
    #r2_test_high = mln.eval_1_set(model, test_high_ds)

    

    print('R2 test wind:           ' + str(r2_test) + '\n')
    print('R2 test wind+std:       ' + str(r2_test_orog) + '\n')
    print('R2 test wind+std+high:  ' + str(r2_test_flat) + '\n')
    #print('R2 test std+high:       ' + str(r2_test_high) + '\n\n')

   

def train_data(model, path_train):
    with np.load(path_train) as data:
        X_train, y_train = torch.tensor(data['feat']), torch.tensor(data['targ'])
    train_ds      = mln.FluxData(X_train, y_train)
    r2_train      = mln.eval_1_set(model, train_ds)

    print('R2 train:      ' + str(r2_train) + '\n')



print('Testing model: ' + nn_path + '\n')

train_data(model, train)#, train, train_orog, train_flat)
test_data(model, test, test_orog, test_flat)#, train, train_orog, train_flat)


