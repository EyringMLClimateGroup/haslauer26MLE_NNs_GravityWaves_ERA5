# Script for calculating sigma and stability profiles for MODES

import numpy as np
import xarray as xr

import datetime as dt



def sigma(sigma, path):

    sigma_np = np.flip(np.array(sigma, dtype = np.float64))
    
    with open(path+'sigma.data', 'wb') as f:
        sigma_np.tofile(f)

    return path+'sigma.data'

def sigma_v2(sigma, path):

    sigma_np = np.flip(np.array(sigma, dtype = np.float64))
    
    with open(path+'sigma.txt', 'w') as f:
        np.savetxt(f, sigma_np, delimiter='\n')

    return path+'sigma.txt'




def stability(sigma, times, path):

    temp_list = []

    for i in range(len(times)):
        t = times[i].strftime('%Y%m%d%H%M%S')
        r = xr.open_dataset(path+'/UVTQSZ_'+t+'0.nc')
        temp_list.append(r.var130.mean(dim = ['lat', 'lon']).values)
        
    ta = np.stack(temp_list).mean(axis = 0)[0]

    kappa = 0.286
    gamma = np.zeros(len(sigma))

    s_u = sigma[0]
    t_u = ta[0]

    for i in range(len(sigma)):
        sig = sigma[i]
        tem = ta[i]

        if (i < len(sigma)-1):
            s_l = 0.5 * (sig + sigma[i+1])
            t_l = 0.5 * (tem + ta[i+1])
        else:
            s_l = sigma[i]
            t_l = ta[i]

        gamma[i] = kappa * ta[i] / sigma[i] - ((t_u-t_l) / (s_u-s_l))

        s_u = s_l
        t_u = t_l
    
    gamma_np = np.flip(np.array(gamma, dtype = np.float64))

    with open(path+'gamma.data', 'wb') as f:
        gamma_np.tofile(f)

    return path+'gamma.data'


def stability_v2(sigma, times, path):

    temp_list = []

    for i in range(len(times)):
        t = times[i].strftime('%Y%m%d%H%M%S')
        r = xr.open_dataset(path+'/UVTQSZ_'+t+'0.nc')
        temp_list.append(r.var130.mean(dim = ['lat', 'lon']).values)
        
    ta = np.stack(temp_list).mean(axis = 0)[0]

    kappa = 0.286
    gamma = np.zeros(len(sigma))

    s_u = sigma[0]
    t_u = ta[0]

    for i in range(len(sigma)):
        sig = sigma[i]
        tem = ta[i]

        if (i < len(sigma)-1):
            s_l = 0.5 * (sig + sigma[i+1])
            t_l = 0.5 * (tem + ta[i+1])
        else:
            s_l = sigma[i]
            t_l = ta[i]

        gamma[i] = kappa * ta[i] / sigma[i] - ((t_u-t_l) / (s_u-s_l))

        s_u = s_l
        t_u = t_l
    
    gamma_np = np.flip(np.array(gamma, dtype = np.float64))



    with open(path+'gamma.txt', 'w') as f:
        np.savetxt(f, gamma_np, delimiter='\n')

    return path+'gamma.txt'




def levels(sigma, path):

    pfulls = sigma*1013.25

    phalfs = np.empty(len(sigma))
    phalfs[len(sigma)-1] = 1013.25

    for i in range(len(sigma)-2, -1, -1):
        phalfs[i] = 0.5*(pfulls[i] + pfulls[i+1])

    with open(path+'levels.dat', 'w') as f:
        f.write('{:>5}{:>17.6f}{:>14.8f}{:>14.3f}{:>14.3f}\n'.format(0,0,0,0,0))
        for i in range(len(sigma)):
            f.write('{:>5}{:>17.6f}{:>14.8f}{:>14.3f}{:>14.3f}\n'.format(i+1,0,sigma[i],phalfs[i],pfulls[i]))

    return path+'levels.dat'