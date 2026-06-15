# %%
# %%
# %%
# LIBRARIES

import numpy as np

import sys
import os

# Get the path to the _utils folder
utils_path = os.path.join(os.path.dirname(__file__), '_utils')

# Add the _utils folder itself to the system path
# This allows 'import __ml_networks' to work directly
sys.path.append(utils_path)

from _utils import __ml_networks as mln


import numpy as np
import xarray as xr

from matplotlib import pyplot as plt

import datetime as dt
import copy
from tqdm import tqdm

import torch
from torch import nn

from torchinfo import summary

import shap

# import torch.nn.functional as F

# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

# from torcheval.metrics import R2Score


device = (
    "cuda"
    if torch.cuda.is_available()
      else "cpu"
)

# %%
# LOAD PREPROCESSED DATA FROM NPZ

path_preprocessed_ig  = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML_new/_n32/IG/_preprocessed_full2024_land_/'
path_model_ig = path_preprocessed_ig + '__models/'

path_training_ig      = path_preprocessed_ig + '_train_.npz'
path_testing_ig       = path_preprocessed_ig + '_test_.npz'

model_ig = 'UNet_2_2026-04-15_UNet_2_IG_land.pt'

path_test_ig          = path_model_ig + model_ig#/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML/_n32/IG/_preprocessed_6m_lo/__models/UNet_128_IG_scfc.pt'

vector_feat        =   4
scalar_feat        =   5
levels             =  37

model_to_test_ig      = torch.load(path_test_ig, map_location=torch.device(device))

with np.load(path_training_ig) as data:
    X_train_ig, y_train_ig = torch.tensor(data['feat']), torch.tensor(data['targ'])

with np.load(path_testing_ig) as data:
    X_test_ig, y_test_ig = torch.tensor(data['feat']), torch.tensor(data['targ'])

# DATASET & DATALOADER

n_feat_ig = X_train_ig.size()[1]
n_targ_ig = y_train_ig.size()[1]
    
train_dataset_ig = mln.FluxData(X_train_ig, y_train_ig)
test_dataset_ig = mln.FluxData(X_test_ig, y_test_ig)
#########################################################################################################

X_train_ig.size(), y_train_ig.size(), X_test_ig.size(), y_test_ig.size()

lev_r2s_train_IG, lev_r2s_test_IG = mln.eval_single_levs(model_to_test_ig, train_dataset_ig, test_dataset_ig, batch_size=4096, mode='UNet_2', levels=37, vector_feat=0, scalar_feat=0, fc_hidden=0, print_results=True)

# %%

path_preprocessed_sg  = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML_new/_n32/SGIG16/_preprocessed_full2024_land_/'
path_model_sg = path_preprocessed_sg + '__models/'

path_training_sg      = path_preprocessed_sg + '_train_.npz'
path_testing_sg       = path_preprocessed_sg + '_test_.npz'

model_sg = 'UNet_2_2026-04-15_UNet_2_SG_land.pt'

path_test_sg          = path_model_sg + model_sg#/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML/_n32/IG/_preprocessed_6m_lo/__models/UNet_128_IG_scfc.pt'

vector_feat        =   4
scalar_feat        =   5
levels             =  37

model_to_test_sg      = torch.load(path_test_sg, map_location=torch.device(device))

with np.load(path_training_sg) as data:
    X_train_sg, y_train_sg = torch.tensor(data['feat']), torch.tensor(data['targ'])

with np.load(path_testing_sg) as data:
    X_test_sg, y_test_sg = torch.tensor(data['feat']), torch.tensor(data['targ'])

# DATASET & DATALOADER

n_feat_sg = X_train_sg.size()[1]
n_targ_sg = y_train_sg.size()[1]
    
train_dataset_sg = mln.FluxData(X_train_sg, y_train_sg)
test_dataset_sg = mln.FluxData(X_test_sg, y_test_sg)
#########################################################################################################

X_train_sg.size(), y_train_sg.size(), X_test_sg.size(), y_test_sg.size()

lev_r2s_train_SG, lev_r2s_test_SG = mln.eval_single_levs(model_to_test_sg, train_dataset_sg, test_dataset_sg, batch_size=4096, mode='UNet_2', levels=37, vector_feat=0, scalar_feat=0, fc_hidden=0, print_results=True)



print('IG')
print(np.array(lev_r2s_train_IG))
print(np.array(lev_r2s_test_IG))
print('SG')
print(np.array(lev_r2s_train_SG))
print(np.array(lev_r2s_test_SG))

numpy_IG_train = np.array(lev_r2s_train_IG)
numpy_IG_test = np.array(lev_r2s_test_IG)
numpy_SG_train = np.array(lev_r2s_train_SG)
numpy_SG_test = np.array(lev_r2s_test_SG)

orostd = ''

np.save('/p/project1/icon-a-ml/haslauer1/gravity_waves/__git/_data/numpy_20260415_levs_IG_train_'+orostd+'.npy', numpy_IG_train)
np.save('/p/project1/icon-a-ml/haslauer1/gravity_waves/__git/_data/numpy_20260415_levs_IG_test_'+orostd+'.npy', numpy_IG_test)
np.save('/p/project1/icon-a-ml/haslauer1/gravity_waves/__git/_data/numpy_20260415_levs_SG_train_'+orostd+'.npy', numpy_SG_train)
np.save('/p/project1/icon-a-ml/haslauer1/gravity_waves/__git/_data/numpy_20260415_levs_SG_test_'+orostd+'.npy', numpy_SG_test)



if orostd == '':
    points = int(2794)
else:
    points = int(2794/2)
cols_r2s_test_ig = mln.eval_single_cols(model_to_test_ig, train_dataset_ig, test_dataset_ig, batch_size=points, mode='UNet_2', levels=37, vector_feat=0, scalar_feat=0, fc_hidden=0, print_results=True, points=points)
cols_r2s_test_sg = mln.eval_single_cols(model_to_test_sg, train_dataset_sg, test_dataset_sg, batch_size=points, mode='UNet_2', levels=37, vector_feat=0, scalar_feat=0, fc_hidden=0, print_results=True, points=points)

np.save('/p/project1/icon-a-ml/haslauer1/gravity_waves/__git/_data/numpy_20260415_map_IG_test_'+orostd+'.npy', np.asarray(cols_r2s_test_ig))
np.save('/p/project1/icon-a-ml/haslauer1/gravity_waves/__git/_data/numpy_20260415_map_SG_test_'+orostd+'.npy', np.asarray(cols_r2s_test_sg))