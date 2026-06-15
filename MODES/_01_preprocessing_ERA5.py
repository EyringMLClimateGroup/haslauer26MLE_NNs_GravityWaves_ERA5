# %% [markdown]

#### 01 / PREPROCESSING ERA5

# NOTICE:

# - Specify the variables simulation (directory of simulation), prefix (first part of filenames of simulation), dir (target directory to store preprocessed data)
# - This script needs the scripts profiles.py, zgrid.py, grid_h_new-ERA5.sh, grid_h_new-ERA5_p2.sh, grid_z_new-ERA5.sh to be in the directory '/_utils/'


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

# Additionally, grid_h_new-ERA5.sh, grid_h_new-ERA5_p2.sh, and grid_z_new-ERA5.sh have to be in the directory '/_utils/'!

# %%
# Input of directories for simulation data, temporary data and results; structure of file names; selection of target grid; selection of time steps


# Directories for simulation raw data ('simulation') and for temporary data and results, prefix of simulation file names

simulation = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/'
prefix = 'p'

dir = simulation + '_MODES/'


# Definition of time steps and target grids

tc = sys.argv[1]
ts = dt.datetime.strptime(tc, '%Y%m%d%H%M%S')
times = [ts]

hgridres = 'n360'        # Target horizontal resolution - nXX/nXXX defines target Gaussian grid, is input grid for grid_h! 
vgridres = 's37'         # Target vertical resolution - sXX/sXXX only the name - will be changed if V_cut != '0' or V_reduce == True

files = dir + '__' + hgridres + vgridres + '/'

preprocessed = files + 'preprocessed/' + tc + '/'
height_files = preprocessed + 'height_files/'
temp         = preprocessed + '/temp/'


# Selection of technical parameters

Parallelize = True      # Set to 'True', if parallelization shall be used; recommended for processing of multiple time steps
V_cut = '1'             # Level number where to start (counted from top) / number-1 of vertical levels to exclude at the top ('integer'), for QUBICC, level [39]/40 is at 52000 km
V_reduce = False        # Set to 'True', for regridding to half of the vertical layers (at the moment the only option)

# %%
# Some preprocessing - create directories and suffixes (suffixes are specific for simulation!)

sp.run(['mkdir -p '+dir], shell = True)
sp.run(['mkdir -p '+files], shell = True)
sp.run(['mkdir -p '+preprocessed], shell = True)
sp.run(['mkdir -p '+height_files], shell = True)
sp.run(['mkdir -p '+temp], shell = True)

suffixes = []
for i in range(len(times)): suffixes.append(times[i].strftime('%Y-%m-%d')+'T'+times[i].strftime('%H:%M:%S'))

# %%
# Horizontal interpolation

#strftime something

tf = dt.datetime.strftime(ts, '%Y-%m-%d')
tm = dt.datetime.strftime(ts, '%Y-%m')

simulation3d = simulation + '__raw/__' + tm + '/' + tf +'_3d.grb'
simulation2d = simulation + '__raw/__' + tm + '/' + tf +'_2d.grb'


if Parallelize == True:

    def conv_hori(suffix):
        sp.run(['./_utils/grid_h_new-ERA5.sh'+' '+simulation3d +' '+ simulation2d +' '+files+' '+prefix+' '+suffix+' '+hgridres+' '+V_cut+' '+preprocessed], shell = True, capture_output = False)

    skip = 4

    idx = np.arange(0, len(times), skip)
    
    for i in idx:
        
        processes = []

        for suffix in suffixes[i:i+skip]:
            task = mp.Process(target = conv_hori, args = (suffix,))
            processes.append(task)
            task.start()

        for task in processes:
            task.join()

# else:
#     for suffix in suffixes: sp.run(['./_utils/grid_h_new-ERA5.sh'+' '+simulation3d +' '+ simulation2d +' '+files+' '+prefix+' '+suffix+' '+hgridres+' '+V_cut+' '+preprocessed], shell = True)


# %%
uvtq = xr.open_dataset(preprocessed + '/temp/_uvtq_'+suffixes[0]+'_'+hgridres+'.grb')

levs = [100.0, 200.0, 300.0, 500.0, 700.0, 
        1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 
        10000.0, 12500.0, 15000.0, 17500.0, 20000.0, 22500.0, 25000.0,
        30000.0, 35000.0, 40000.0, 45000.0, 50000.0, 55000.0, 60000.0, 65000.0,
        70000.0, 75000.0, 77500.0, 80000.0, 82500.0, 85000.0, 87500.0, 90000.0, 92500.0, 95000.0, 97500.0, 100000.0]

p_levels = np.zeros((37,720,1440))
for i in range(720):
    for j in range(1440):
     p_levels[:,i,j] = levs

lat = uvtq.latitude.values
lon = uvtq.longitude.values
lev = np.arange(37)

p_levels_ds = xr.Dataset(data_vars = dict(var131=(['generalVertical','latitude','longitude'], p_levels)),
                        coords = dict(longitude=lon, latitude=lat, generalVertical=lev))

p_levels_ds.to_netcdf(preprocessed+'p_levels.nc')

for suffix in suffixes:
    sp.run(['./_utils/grid_h_new-ERA5_p2.sh'+' '+simulation3d +' '+ simulation2d +' '+files+' '+prefix+' '+suffix+' '+hgridres+' '+V_cut+' '+preprocessed], shell = True, capture_output = False)


# %%
# Generation of auxiliary vertical grid file ('vgridfile') and vertical interpolation

vgridfile, vgridlev = zgridERA5.create_sigma(preprocessed, suffixes, hgridres, reduce = V_reduce)
vgridres = 's' + str(vgridlev)


# %%
if Parallelize == True:

    def conv_vert(suffix):
        sp.run(['./_utils/grid_z_new-ERA5.sh'+' '+simulation+' '+files+' '+prefix+' '+suffix+' '+hgridres+' '+vgridres+' '+vgridfile+' '+preprocessed], shell = True, capture_output = False)

    idx = np.arange(0, len(times), 4)
    
    for i in idx:
        
        processes = []

        for suffix in suffixes[i:i+10]:
            task = mp.Process(target = conv_vert, args = (suffix,))
            processes.append(task)
            task.start()

        for task in processes:
            task.join()

else:
    for suffix in suffixes: sp.run(['./_utils/grid_z_new-ERA5.sh'+' '+simulation+' '+files+' '+prefix+' '+suffix+' '+hgridres+' '+vgridres+' '+vgridfile+' '+preprocessed], shell = True)


# %%
# Generation of input files for MODES, containing u, v, t, q, s, z

data_inter2d_z = xr.open_dataset(preprocessed+'temp/_zs_'+hgridres+'.grb')
z = [[data_inter2d_z.z.values]]

lat = data_inter2d_z.latitude.values
lon = data_inter2d_z.longitude.values
lev = np.arange(1,vgridlev+1,1)
lev_2 = np.arange(1,2,1)
lev_ps = np.arange(191,192,1)

def write_dataset(i):
    data_inter3d = xr.open_dataset(preprocessed+'temp/_uvtq_'+suffixes[i]+'_'+hgridres+vgridres+'.grb').isel(isobaricInPa=slice(None, None, -1))
    data_inter2d_lnps = xr.open_dataset(preprocessed+'temp/_ln_'+suffixes[i]+'_'+hgridres+'.grb')

    u = [data_inter3d.u.values]
    v = [data_inter3d.v.values]
    t = [data_inter3d.t.values]
    q = [data_inter3d.q.values]
    s = [[data_inter2d_lnps.sp.values]]
    #z = [[data_inter2d_z.h.values]]

    DATASET = xr.Dataset(data_vars = dict(var131=(['time','lev','lat','lon'], u), var132=(['time','lev','lat','lon'], v), var130=(['time','lev','lat','lon'], t), var133=(['time','lev','lat','lon'], q),
                                        var152=(['time','lev_ps','lat','lon'], s), var129=(['time','lev_2','lat','lon'], z)),
                        coords = dict(time=[times[i]], lon=lon, lat=lat, lev=lev, lev_2=lev_2))

    t = times[i].strftime('%Y%m%d%H%M%S')
    DATASET.to_netcdf(preprocessed+'UVTQSZ_'+t+'0.nc')

    return DATASET


for i in range(len(times)): write_dataset(i)


# %%
# Generation of auxiliary files for MODES with sigma profile, gamma profile, and model levels

sigma = xr.open_dataset(vgridfile).sig.mean(dim=['time', 'lat', 'lon'])

sigma_file = profiles.sigma(sigma, preprocessed)
gamma_file = profiles.stability(sigma, times, preprocessed)
levels_file = profiles.levels(sigma, preprocessed)