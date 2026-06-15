# %% [markdown]

#### 04 / CALCULATION OF GRAVITY WAVE MOMENTUM FLUXES

# Includes regridding of wa (vertical velocity) from ERA5 to n360 grid


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

# Additionally, fluxes.sh and grid_h_files-ERA5_wa.sh have to be in ./_utils/!

# convert u v w t rho p to n360
# compute u*w, then n32

# %%
# Input of directories for simulation data, temporary data and results; structure of file names; selection of target grid; selection of time steps

# Directories for simulation raw data ('simulation') and for temporary data and results, prefix of simulation file names

simulation = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/'
modes = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_MODES/__n360s37/_30-400-230/_ERA5/'
prefix = 'p'

dir = simulation + '_MODES/'

grid = 'n32'
igtypes = ['IG', 'SGIG16', 'SGIG32']

t_start = dt.datetime(2022,1,1,0)
t_end = dt.datetime(2022,1,1,23)
t_delta = dt.timedelta(hours=1)

hgridres = 'n360'        # Target horizontal resolution - nXX/nXXX defines target Gaussian grid, is input grid for grid_h_files-ERA5_wa.sh! 
vgridres = 's37'         # Target vertical resolution - sXX/sXXX only the name - will be changed if V_cut != '0' or V_reduce == True

################################################################################################################

# Technical preparation

ts = []  # ts in datetime format

ts.append(t_start)

while ts[-1] < t_end:
    ts.append(ts[-1] + t_delta)
    
tc = []        # tc is simple string
suffixes = []  # suffixes is T-Z-string
                                 
for i in range(len(ts)):
    tc.append(ts[i].strftime('%Y%m%d%H%M%S'))
    suffixes.append(ts[i].strftime('%Y-%m-%d')+'T'+ts[i].strftime('%H:%M:%S'))

files = dir + '__' + hgridres + vgridres + '/'

month=dt.datetime.strftime(ts[0],'%Y-%m')

directory = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_VARS_NEW/_'+hgridres+'/__'+month+'/'
month_dir = '__'+dt.datetime.strftime(ts[0],'%Y-%m')+'/' #klappt wenn monatsweise!

sp.run(['mkdir -p '+ directory], shell = True)

temp = directory + 'temp/'

tf0 = dt.datetime.strftime(ts[0], '%Y-%m-%d')


# %%
# Regridding of vertical velocities wa

def conv_hori(suff, tcid):
    sp.run(['./_utils/grid_h_files-ERA5_wa.sh'+' '+simulation3d +' '+ simulation2d +' '+files+' '+prefix+' '+suff+' '+hgridres+' '+directory+' '+tcid], shell = True, capture_output = False)

for j in range(len(ts)):
    tf = dt.datetime.strftime(ts[j], '%Y-%m-%d')
    simulation3d = simulation + '__raw/' +month_dir+ tf+'_3d.grb'
    simulation2d = simulation + '__raw/' +month_dir+ tf+'_2d.grb'
    suff = suffixes[j]
    tcid = tc[j]
    conv_hori(suff, tcid)



# %%
# Caclulation of Fluxes

for igtype in igtypes:

    month = dt.datetime.strftime(ts[0],'%Y-%m')

    modes_era = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_MODES/__n360s37/_30-400-230/_ERA5_'+igtype+'/__'+month+'/'
    directory = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_FLUXES_NEW/_'+grid+'/'+igtype+'/__'+month+'/'
    temp = directory + 'temp/'
    sp.run(['mkdir -p '+ directory], shell = True)
    VARS = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_VARS_NEW/_n360/__'+month+'/'

    def flux_calc(infile_1, infile_2, infile_w, target_1, target_2):
        sp.run(['./_utils/fluxes.sh'+' '+ infile_1 +' '+ infile_2 +' '+ infile_w + ' ' +target_1 + ' ' + target_2 + ' ' + grid], shell = True, capture_output = False)

    idx = np.arange(0, len(tc), 1)

    for i in idx:

        tf = dt.datetime.strftime(ts[i], '%Y-%m-%d')

        infile_1 = modes_era + igtype + '_' + tc[i] + '_u.grb'
        infile_2 = modes_era + igtype + '_' + tc[i] + '_v.grb'
        infile_w = VARS + 'wa_' + tc[i] + '_n360.grb'
        target_1 = directory + 'MFx_' + igtype + '_' + tc[i] + '_'+grid+'.grb'
        target_2 = directory + 'MFy_' + igtype + '_' + tc[i] + '_'+grid+'.grb'

        flux_calc(infile_1, infile_2, infile_w, target_1, target_2)