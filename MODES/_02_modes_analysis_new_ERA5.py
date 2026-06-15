# %% [markdown]

#### 02 / MODES ANALYSIS ERA5

# With this script, one can run all steps of a MODES (Zagar 2015) analysis with ERA5 data. Data has to be prepared with the corresponding preprocessing script.


# %%
# Import of all required packages


# Python packages

import sys

import numpy as np
import time

import datetime as dt
import multiprocessing as mp
import subprocess as sp


# Python files with functions

from _utils import cnf_files_era5 as cnf_files

# Additionally, run_modes_01.sh, run_modes_03.sh, run_modes_04.sh and run_modes_05.sh have to be in the directory '/_utils/'!


# %%
tc = sys.argv[1]
ts = dt.datetime.strptime(tc, '%Y%m%d%H%M%S')

lenlat, lenlon, lenlev = 720, 1440, 37

hgridres = 'n360'
vgridres = 's37'

preprocessed = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_MODES/__n360s37/preprocessed/' + tc + '/'

vgridfile = preprocessed + 'sigma_n360s37.nc'
sigma_file = preprocessed + 'sigma.data'
gamma_file = preprocessed + 'gamma.data'
levels_file = preprocessed + 'levels.dat'

Parallelize = True


# %%
# MODES analysis part I: Gaussian grid, vertical structure functions, hough functions


# Selection of modes to be analyzed:

num_vmode = 30      # Vertical modes (select in a way such that the smallest equivalent depth is 1m)
num_zw = 400          # Zonal wavenumbers (needs to be < than # zonal grid points)
maxl = 230            # Number of each hough modes (EIG, WIG, ROT)


# Create directory for storing temporary data and results

new_results = '_'+str(num_vmode)+'-'+str(num_zw)+'-'+str(maxl)

files = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_MODES/__n360s37/'
if new_results != '':
    temp = files+new_results+'/temp_' + tc + '/'
    sp.run(['mkdir -p '+files+new_results+'/'], shell = True)
    sp.run(['mkdir -p '+ temp], shell = True)
    results = files+new_results+'/'


times = [ts]



# Technical preparation

timestep = 0

name = hgridres + vgridres
  
sp.run(['mkdir -p '+temp+'gauss'], shell = True)
sp.run(['mkdir -p '+temp+'vsf'], shell = True)
sp.run(['mkdir -p '+temp+'hough_cnfs'], shell = True)
sp.run(['mkdir -p '+temp+'hough'], shell = True)
sp.run(['mkdir -p '+temp+'coef'], shell = True)
sp.run(['mkdir -p '+temp+'inverse'], shell = True)


# %%
# Generation of input files and running of gauss.run, vsf.run, hort_struct.run

cnf_files.write_cnf_gauss(lenlat, temp)
cnf_files.write_cnf_vsf(lenlev, sigma_file, gamma_file, num_vmode, temp)

sp.run(['./_utils/run_modes_01.sh '+temp], shell = True)#, capture_output = True)

# %%
# MODES analysis part II: Expansion of data, inversion back to physical space


# Selection of filtered modes [eig_n_s > 0, eig_n_e, wig_n_s, wig_n_e, rot_n_s, rot_n_e, kmode_s, kmode_n, vmode_s, vmode_e]
# Per default, there are three different filtering settings: ALL, IG, ROT

# No filtering, if ..*_s is greater than ..*_e
# If filtering shall be applied, please verify that values selected for num_vmode, num_zw, maxl are not exceeded!
# Comment lines ALL, IG, or ROT, if MODES shall not be applied

LIST = ['IG', 'SGIG16', 'SGIG32']

IG  =        [ 10000,      1, 10000,    1,     0,  maxl, 10000,  1, 10000, 1]
SGIG16  =    [ 10000,      1, 10000,    1,     0,  maxl,     0, 16, 10000, 1]
SGIG32  =    [ 10000,      1, 10000,    1,     0,  maxl,     0, 32, 10000, 1]


# %%
# Functions for hough, expansion and inversion


def mod_hough(lenlat, num_vmode, start_zw, end_zw, maxl, temp, temp_t):
    
    time.sleep(start_zw)
    cnf_files.write_cnf_hough(lenlat, num_vmode, start_zw, end_zw, maxl, temp, temp_t)
    sp.run(['./_utils/run_modes_03.sh '+temp_t], shell = True)


def mod_exp(lenlon, lenlat, lenlev, times, timestep, sigma_file, levels_file, num_vmode, num_zw, maxl,
                            origin = False, temp = temp):
    
    cnf_files.write_cnf_exp(lenlon, lenlat, lenlev, times, timestep, sigma_file, levels_file, num_vmode, num_zw, maxl,
                            origin = origin, temp = temp)
    sp.run(['./_utils/run_modes_04.sh '+temp], shell = True)


def mod_inv(lenlon, lenlat, lenlev, times, timestep, sigma_file, levels_file, num_vmode, num_zw, maxl,
                        eig_n_s = 10000, eig_n_e = 1, wig_n_s = 10000, wig_n_e = 1, rot_n_s = 10000, rot_n_e = 0, kmode_s = 10000, kmode_e = 1, vmode_s = 10000, vmode_e = 1,
                        temp = temp, result = temp):
    
    cnf_files.write_cnf_inv(lenlon, lenlat, lenlev, times, timestep, sigma_file, levels_file, num_vmode, num_zw, maxl,
                        eig_n_s, eig_n_e, wig_n_s, wig_n_e, rot_n_s, rot_n_e, kmode_s, kmode_e, vmode_s, vmode_e,
                        temp, result = result)   
    sp.run(['./_utils/run_modes_05.sh '+temp], shell = True)


# %%
# MODES Houghs

if Parallelize == True and num_zw > 10:

    bins = 5

    if num_zw > 5:
        start_zw = np.arange(0,num_zw+1, bins)
        end_zw = np.arange(0,num_zw+1, bins)
        for i in range(len(start_zw) - 1): end_zw[i] = end_zw[i] + bins - 1
        end_zw[-1] = num_zw

    # This is parallelization with respect to calculation of Hough harmonics

    processes = []

    for i in range(len(start_zw)):
        temp_t = temp+'hough/h_'+str(start_zw[i])+'/'
        sp.run(['mkdir -p ' + temp_t], shell=True)
        task = mp.Process(target = mod_hough, args = (lenlat, num_vmode, start_zw[i], end_zw[i], maxl, temp, temp_t))
        processes.append(task)
        task.start()

    for task in processes:
        task.join()

else:

    mod_hough(lenlat, num_vmode, 0, num_zw, maxl, temp)


# %%
# MODES NMF expansion

mod_exp(lenlon, lenlat, lenlev, times, timestep, sigma_file, levels_file, num_vmode, num_zw, maxl, origin = preprocessed, temp = temp)


# %%
# MODES inversion

for X in LIST:
        mod_inv(lenlon, lenlat, lenlev, times, timestep, sigma_file, levels_file, num_vmode, num_zw, maxl,
                                        eval(X)[0], eval(X)[1], eval(X)[2], eval(X)[3], eval(X)[4], eval(X)[5], eval(X)[6], eval(X)[7], eval(X)[8], eval(X)[9],
                                        temp = temp, result = results+'ERA5_'+X+'_')