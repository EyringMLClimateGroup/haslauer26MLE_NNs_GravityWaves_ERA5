# SCRIPT FOR CALCULATION OF SHAP VALUES


# LIBRARIES

import numpy as np

import copy

import torch
from torchinfo import summary

import shap

from _utils import __ml_networks as mln


device = (
    "cuda"
    if torch.cuda.is_available()
      else "cpu"
)



####################################################################################################
##  LOAD PREPROCESSED DATA FROM NPZ ################################################################

path_preprocessed = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML_new/_n32/IG/_preprocessed_full2024_land_/'

path_training = path_preprocessed + '_train_.npz'
path_testing = path_preprocessed + '_test_.npz'

with np.load(path_training) as data:
    X_train, y_train = torch.tensor(data['feat']), torch.tensor(data['targ'])

with np.load(path_testing) as data:
    X_test, y_test = torch.tensor(data['feat']), torch.tensor(data['targ'])



# DATASET & DATALOADER

n_feat = X_train.size()[1]
n_targ = y_train.size()[1]
    
train_dataset = mln.FluxData(X_train, y_train)
test_dataset = mln.FluxData(X_test, y_test)

X_train.size(), y_train.size(), X_test.size(), y_test.size()



# TEST NN

path_test = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML_new/_n32/IG/_preprocessed_full2024_land_/__models/UNet_2_2026-04-15_UNet_2_IG_land.pt'# path_preprocessed + '__models/UNet_2_20251121-UNet_2-bf256-mse.pt'

model_to_test = torch.load(path_test, map_location=torch.device(device))
print(summary(model_to_test))



# SHAPLEY VALUES GLOBAL (shap_size=1000 needs 20 min)

LEVS=37

path_shap = path_test
shap_size, shap_split = 1200, 0.5

model_to_shap = torch.load(path_shap, map_location=torch.device(device))

shap_idx = np.random.choice(test_dataset.features.shape[0], size=shap_size, replace=False)
background_idx, explain_idx = shap_idx[:int(shap_size*shap_split)], shap_idx[int(shap_size*shap_split):]

shap_X_test = test_dataset.features

X_background, X_explain = shap_X_test[background_idx].to(device), shap_X_test[explain_idx].to(device)

shap_model = copy.deepcopy(model_to_shap)
shap_model.to(device)

explainer = shap.GradientExplainer(shap_model, X_background)
shap_vals = explainer.shap_values(X_explain)

np.save(path_test[:-3] + '_shap_vals.npy', shap_vals)
np.save(path_test[:-3] + '_X_explain.npy', X_explain.cpu())

shap_vals.shape