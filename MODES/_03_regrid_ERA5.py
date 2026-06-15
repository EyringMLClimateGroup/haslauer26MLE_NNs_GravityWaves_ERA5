# %% [markdown]

#### 03 / REGRIDDING TO ERA5 GRID

# %%
# Import of all required packages

import sys

import datetime as dt
import subprocess as sp

# Additionally, grid_z_rev_ERA5.sh have to be in the directory '/_utils/'!


# %%
# Directories of MODES results and target directory

modes = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_MODES/__n360s37/_30-400-230/'

types = ['IG', 'SGIG16', 'SGIG32']

tc_arg = sys.argv[1]
ts = dt.datetime.strptime(tc_arg, '%Y%m%d%H%M%S')


simulation = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/'
prefix = 'p'

dir = simulation + '_MODES/'

ts = [ts]  # ts in datetime format
    
tc = [tc_arg]        # tc is simple string
suffixes = [ts[0].strftime('%Y-%m-%d')+'T'+ts[0].strftime('%H:%M:%S')]  # suffixes is T-Z-string
              
idx = range(1)


month = dt.datetime.strftime(ts[0],'%Y-%m')

lenlat, lenlon, lenlev = 720, 1440, 37

hgridres = 'n360'
vgridres = 's37'

files = dir + '__' + hgridres + vgridres + '/'


# %%
# Conversion to original ERA5 grid

def z_coord(type, step, preprocessed):
    infile = modes+'ERA5_'+type+'_'+ts[step].strftime('%Y%m%d%H%M%S')+'0.nc'
    vgridfile = preprocessed + 'sigma_n360s37.nc'

    outfile = ERA+type+'_'+tc[step]
    intermediatefile = ERA+type+'_'+tc[step]+'_interm.grb'
    sp.run(['./_utils/grid_z_rev_ERA5.sh'+' '+infile+' '+outfile+' '+intermediatefile+' '+preprocessed+' '+suffixes[step]+' '+vgridfile], shell = True)


for igtype in types:

    ERA = modes + '_ERA5_'+igtype+'/__' + month + '/'
    sp.run(['mkdir -p '+ERA], shell = True)

    for step in idx:
        preprocessed = files + 'preprocessed/' + tc[step] + '/'
        height_files = preprocessed + 'height_files/'
        z_coord(igtype, step, preprocessed)