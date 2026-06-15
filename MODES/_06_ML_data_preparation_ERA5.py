#### PREPARATION OF VARIOUS VARIABLES FOR ML-WORK

# %%
# Import of all required packages


# Python packages

import sys
import datetime as dt
import multiprocessing as mp
import subprocess as sp

import numpy as np
import xarray as xr


# Python files with functions

from _utils import profiles
from _utils import zgridERA5

# Additionally, grid_h_files-ERA5_once_plus.sh and grid_h_files-ERA5_plus.sh have to be in the directories '/_utils/'!


# %%
# Input of directories for simulation data, temporary data and results; structure of file names; selection of target grid; selection of time steps


# Directories for simulation raw data ('simulation') and for temporary data and results, prefix of simulation file names

simulation = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/'
prefix = 'p'

dir = simulation + '_MODES/'


# Selection of time steps and target grids

t_start = dt.datetime(2022,1,1,0)
t_end = dt.datetime(2022,1,1,23)
t_delta = dt.timedelta(hours=1)

ts = []  # ts in datetime format

ts.append(t_start)

while ts[-1] < t_end:
    ts.append(ts[-1] + t_delta)
    
tc = []        # tc is simple string
suffixes = []  # suffixes is T-Z-string
              
for i in range(len(ts)):
    tc.append(ts[i].strftime('%Y%m%d%H%M%S'))
    suffixes.append(ts[i].strftime('%Y-%m-%d')+'T'+ts[i].strftime('%H:%M:%S'))

hgridres = 'n360'        # Target horizontal resolution - nXX/nXXX defines target Gaussian grid, is input grid for grid_h! 
vgridres = 's37'         # Target vertical resolution - sXX/sXXX only the name - will be changed if V_cut != '0' or V_reduce == True

files = dir + '__' + hgridres + vgridres + '/'


# %%
# Coarse-graining of all variables

grid='n32'
igtype='IG'
month=dt.datetime.strftime(ts[0],'%Y-%m')


directory = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_VARS_NEW/_'+grid+'/__'+month+'/'
fluxes = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_FLUXES_NEW/_'+grid+'/'+igtype+'/__'+month+'/'
ml_dir = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_ML_plus/_'+grid+'/'+igtype+'/__'+month+'/'
month_dir = '__'+dt.datetime.strftime(ts[0],'%Y-%m')+'/' #klappt wenn monatsweise!

sp.run(['mkdir -p '+ directory], shell = True)
sp.run(['mkdir -p '+ ml_dir], shell = True)


temp = directory + 'temp/'

tf0 = dt.datetime.strftime(ts[0], '%Y-%m-%d')
simulation3d = simulation + '__raw/' +month_dir+ tf0+'_3d.grb'
simulation2d = simulation + '__raw/' +month_dir+ tf0+'_2d.grb'
suffix = suffixes[0]

# Processing of orography, should be done only once
sp.run(['./_utils/grid_h_files-ERA5_once_plus.sh'+' '+simulation3d +' '+ simulation2d +' '+fluxes+' '+prefix+' '+suffix+' '+grid+' '+directory], shell = True, capture_output = False)

# Processing of the other variables
def conv_hori(suff, tcid):
    # ua, va, wa, ta, hus, geop, precip
    sp.run(['./_utils/grid_h_files-ERA5_plus.sh'+' '+simulation3d +' '+ simulation2d +' '+fluxes+' '+igtype+' '+suff+' '+grid+' '+directory+' '+tcid], shell = True, capture_output = False)
 
for j in range(len(ts)):
    tf = dt.datetime.strftime(ts[j], '%Y-%m-%d')
    simulation3d = simulation + '__raw/' +month_dir+ tf+'_3d.grb'
    simulation2d = simulation + '__raw/' +month_dir+ tf+'_2d.grb'
    suff = suffixes[j]
    tcid = tc[j]
    conv_hori(suff, tcid)

# %%
# Preparation of .npz-files with variables, one file per time step

for i in range(len(ts)):
    ua_dict = np.flip(xr.open_dataset(directory + 'ua_' + tc[i] + '_' + grid + '.grb').u.drop(['time', 'step', 'valid_time']).values.swapaxes(1,2), axis=0)
    va_dict = np.flip(xr.open_dataset(directory + 'va_' + tc[i] + '_' + grid + '.grb').v.drop(['time', 'step', 'valid_time']).values.swapaxes(1,2), axis=0)
    wa_dict = np.flip(xr.open_dataset(directory + 'wa_' + tc[i] + '_' + grid + '.grb').w.drop(['time', 'step', 'valid_time']).values.swapaxes(1,2), axis=0)
    ta_dict = np.flip(xr.open_dataset(directory + 'ta_' + tc[i] + '_' + grid + '.grb').t.drop(['time', 'step', 'valid_time']).values.swapaxes(1,2), axis=0)

    # rho_dict[i] = xr.open_dataset(directory + '_rho_phy_ml_R02B05_' + suffixes[i] + '.nc').den.drop(['clon', 'clat', 'time']).values[0,:,:]
    # pfull_dict[i] = xr.open_dataset(directory + '_extra_3d_pfull_ml_R02B05_' + suffixes[i] + '.nc').pres.drop(['clon', 'clat', 'time']).values[0,:,:]
    # pr_dict = xr.open_dataset(directory + 'precip_' + tc[i] + '_' + grid + '.grb').tp.drop(['time', 'step', 'surface', 'valid_time']).values.swapaxes(0,1)
    # uas_dict[i] = xr.open_dataset(directory + '_uas_atm2d_ml_R02B05_' + suffixes[i] + '.nc').uas.drop(['clon', 'clat', 'time']).values[0,:]
    # vas_dict[i] = xr.open_dataset(directory + '_vas_atm2d_ml_R02B05_' + suffixes[i] + '.nc').vas.drop(['clon', 'clat', 'time']).values[0,:]
    
    geop_dict = xr.open_dataset(directory + '_geop_' + grid + '.grb').z.values.swapaxes(0,1)
    oro_std_dict = xr.open_dataset(directory + '_oro_std_' + grid + '.grb').sdor.values.swapaxes(0,1)
    oro_anis_dict = xr.open_dataset(directory + '_oro_anis_' + grid + '.grb').isor.values.swapaxes(0,1)
    oro_angle_dict = xr.open_dataset(directory + '_oro_angle_' + grid + '.grb').anor.values.swapaxes(0,1)
    oro_slope_dict = xr.open_dataset(directory + '_oro_slope_' + grid + '.grb').slor.values.swapaxes(0,1)
    land_sea_mask_dict = xr.open_dataset(directory + '_land_sea_mask_' + grid + '.grb').lsm.drop(['time', 'step', 'surface', 'valid_time']).values.swapaxes(0,1)

    zf_dict = xr.open_dataset(directory + 'MFx_' + igtype + '_' + tc[i] + '_' + grid + '.grb', decode_times=False).drop(['time']).u.values[0,:,:,:].swapaxes(1,2)
    mf_dict = xr.open_dataset(directory + 'MFy_' + igtype + '_' + tc[i] + '_' + grid + '.grb', decode_times=False).drop(['time']).v.values[0,:,:,:].swapaxes(1,2)


    np.savez(ml_dir + tc[i] + '.npz', 
        ua = ua_dict.T,
        va = va_dict.T,
        wa = wa_dict.T,
        ta = ta_dict.T,
        # rho = rho_dict[i].T,
        # pfull = pfull_dict[i].T,
        # pr = pr_dict.T,
        # uas = uas_dict[i].T,
        # vas = vas_dict[i].T,

        geop = geop_dict.T,
        oro_std = oro_std_dict.T,
        oro_anis = oro_anis_dict.T,
        oro_angle = oro_angle_dict.T,
        oro_slope = oro_slope_dict.T,
        land_sea_mask = land_sea_mask_dict.T,     

        zf = zf_dict.T,
        mf = mf_dict.T)