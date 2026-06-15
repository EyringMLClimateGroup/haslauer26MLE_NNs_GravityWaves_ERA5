# SCRIPT FOR PREPROCESSING TEST DATASET

# LIBRARIES

import numpy as np
import subprocess as sp
import torch

from _utils import __ml_functions as mlf


####################################################################################################
## PARAMETERS ######################################################################################

DATA_BASE_DIRECTORY   = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML_new/'

IGTYPE                = 'IG'
GRIDTYPE              = 'n32'

DATA_DIRECTORY        = DATA_BASE_DIRECTORY + '_' + GRIDTYPE + '/' + IGTYPE + '/'

DATA_YEARS            = [2022]
DATA_MONTHS           = [1,2,3,4,5,6,7,8,9,10,11,12]
DAYS                  = [1,11,21]                   # Select no day to select all days in every month

REGIONS               = ['GLOBAL_70']

FEATURES_3D           = ['ua', 'va', 'wa', 'ta']
FEATURES_SP           = []
FEATURES_2D           = ['geop', 'oro_std', 'oro_anis', 'oro_angle', 'oro_slope', 'land_sea_mask']
TARGETS_3D            = ['zf', 'mf']


LAND_ONLY             = True
OROSTD_ONLY           = False
FLAT_ONLY             = False
HIGH_MTS              = False
HIGH_WIND             = False

NORMALIZE             = True
PRESET_STATISTICS     = True

name = '/_preprocessed_full2024_land_'

#if LAND_ONLY is True: name += 'land_'
#if OROSTD_ONLY is True: name += 'orostd_'
#if FLAT_ONLY is True: name += 'flat_'
    
if NORMALIZE is False: name += 'nonorm_'
name += '/'

path_preprocessed = DATA_BASE_DIRECTORY + '_' + GRIDTYPE + '/' + IGTYPE + name
statistics_path   = path_preprocessed


#########################################################################################################


sp.run(['mkdir -p ' + path_preprocessed], shell=True)
sp.run(['mkdir -p ' + statistics_path], shell=True)

FEAT, TARG, COLS = {}, {}, []

if len(DAYS) != 0:
    INTERVALS = mlf.get_intervals(DATA_YEARS, DATA_MONTHS, DAYS)
else:
    INTERVALS = mlf.get_intervals(DATA_YEARS, DATA_MONTHS)

DATA = mlf.get_data(DATA_DIRECTORY, INTERVALS, REGIONS, add='')

LEVS = DATA[list(DATA.keys())[0]][0]['zf'].shape[-1]
STEPS = len(DATA[list(DATA.keys())[0]])

for l in REGIONS:
    FEAT[l], TARG[l], _ = mlf.assemble_COLS3D(DATA[l], features_3d=FEATURES_3D, features_2d=FEATURES_2D, features_sp=FEATURES_SP, targets_3d=TARGETS_3D)

if LAND_ONLY is False:
    n_feat, n_targ = len(FEAT[list(FEAT.keys())[0]][0,0,:])-1, len(TARG[list(TARG.keys())[0]][0,0,:])
elif LAND_ONLY is True:
    n_feat, n_targ = len(FEAT[list(FEAT.keys())[0]][0,0,:-1]), len(TARG[list(TARG.keys())[0]][0,0,:])

FEAT_TENSOR, TARGET_TENSOR = mlf.get_feat_targ_regions(FEAT, TARG, REGIONS)
COLS = TARGET_TENSOR.size()[1]
#########################################################################################################

# check
print(list(DATA.keys()), list(DATA[list(DATA.keys())[0]][0]), list(DATA[list(DATA.keys())[0]][0]['zf'].shape))
print(FEAT[list(FEAT.keys())[0]].size(), TARG[list(TARG.keys())[0]].size())



# NORMALIZATION

if NORMALIZE is True:
    if PRESET_STATISTICS:
       with np.load(statistics_path + '_statistics.npz') as f:
           feat_mean, feat_std, targ_mean, targ_std = f['feat_mean'], f['feat_std'], f['targ_mean'], f['targ_std']

    if PRESET_STATISTICS is False:
        feat = torch.zeros_like(FEAT_TENSOR)
        feat[:,:,:-1], feat_mean, feat_std  = mlf.normalize_timelev(FEAT_TENSOR[:,:,:-1], STEPS, COLS)
        feat[:,:,-1] = FEAT_TENSOR[:,:,-1]
        targ, targ_mean, targ_std = mlf.normalize_timelev(TARGET_TENSOR, STEPS, COLS)
        sp.run(['mkdir -p ' + statistics_path], shell=True)
        np.savez(statistics_path + '_statistics.npz', 
            feat_mean = feat_mean,
            feat_std = feat_std,
            targ_mean = targ_mean,
            targ_std = targ_std)
        
    if PRESET_STATISTICS is True:
        feat = torch.zeros_like(FEAT_TENSOR)
        feat[:,:,:-1] = mlf.normalize_timelev(FEAT_TENSOR[:,:,:-1], STEPS, COLS, new_calc=False, mean_vec=feat_mean, std_vec=feat_std)
        feat[:,:,-1] = FEAT_TENSOR[:,:,-1]
        targ = mlf.normalize_timelev(TARGET_TENSOR, STEPS, COLS, new_calc=False, mean_vec=targ_mean, std_vec=targ_std)

else:
    feat = FEAT_TENSOR[:,:,:]
    targ = TARGET_TENSOR


# DEFINE TEST AND TRAINING SETS, INCLUDING LAND SEA MASK

test_feat, test_targ = feat[:,:,:].reshape(-1, n_feat+1), targ[:,:,:].reshape(-1, n_targ)

if LAND_ONLY is True:
    lsm_mask_ids_test  = torch.nonzero(test_feat[:,-1],as_tuple=True)[0]
    test_feat, test_targ = test_feat[lsm_mask_ids_test,:-1], test_targ[lsm_mask_ids_test,:]

if OROSTD_ONLY is True:
    orostd_mask_ids_test = test_feat[:,-4] > test_feat[:,-4].median()
    test_feat, test_targ = test_feat[orostd_mask_ids_test,:], test_targ[orostd_mask_ids_test,:]

if FLAT_ONLY is True:
    flat_mask_ids_test = test_feat[:,-4] < test_feat[:,-4].median()
    test_feat, test_targ = test_feat[flat_mask_ids_test,:], test_targ[flat_mask_ids_test,:]

if HIGH_MTS is True:
    hm_mask_ids_test = test_feat[:,-5] > (((1078.0572509765623*9.81) - feat_mean[-5]) / feat_std[-5])        # 1500m * 9.81
    test_feat, test_targ = test_feat[hm_mask_ids_test,:], test_targ[hm_mask_ids_test,:]

if HIGH_WIND is True:
    u_vals = test_feat[:, 36:37]
    v_vals = test_feat[:, 73:74]
    
    # If the data is normalized, we MUST un-normalize it first to calculate physical wind speed
    if NORMALIZE is True:
        u_vals = u_vals * feat_std[36:37] + feat_mean[36:37]
        v_vals = v_vals * feat_std[73:74] + feat_mean[73:74]
        
    # Calculate absolute wind speed: sqrt(u^2 + v^2)
    wind_speed = torch.sqrt(u_vals**2 + v_vals**2)
    
    # Average across the 5 levels (dim=1 is the level dimension for these sliced arrays)
    wind_speed_mean = wind_speed.mean(dim=1)
    
    # Create mask for mean wind speed > 8 m/s
    hw_mask_ids_test = wind_speed_mean > 4.24764585
    
    # Apply mask to keep only high wind entries
    test_feat, test_targ = test_feat[hw_mask_ids_test, :], test_targ[hw_mask_ids_test, :]


# Mask NaNs
test_mask = ~(torch.isnan(test_feat).any(dim=1) | torch.isnan(test_targ).any(dim=1))
test_feat, test_targ = test_feat[test_mask], test_targ[test_mask]



# SAVE PREPROCESSED DATA TO NPZ

np.savez(path_preprocessed + '_test_.npz', 
    feat = test_feat,
    targ = test_targ)