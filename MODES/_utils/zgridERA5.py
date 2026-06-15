# Preparation of vertical sigma profile for MODES

import datetime as dt
import numpy as np
import xarray as xr

def create_sigma(preprocessed, suffixes, hgridres, reduce = False):
    
    list = []

    for suffix in suffixes:
        s = xr.open_dataset(preprocessed + 'height_files/_pdivps_'+suffix+'_'+hgridres+'.grb')
        list.append(s.var131.mean(dim = ['latitude', 'longitude']).values)
        
    vec = np.stack(list).mean(axis = 0)


    time = [dt.datetime(2004,4,1)]
    lat = s.latitude.values
    lon = s.longitude.values

    if reduce == False: lev = np.arange(1, len(vec) + 1, 1)
    else: lev = np.arange(1, len(vec) // 2 + 1, 1)

    sigma_grid = np.zeros([len(lev),len(lat),len(lon)])

    for k in range(len(lev)):
        for i in range(len(lat)):
            for j in range(len(lon)):

                if reduce == False: sigma_grid[k,i,j] = vec[k]
                else: sigma_grid[k,i,j] = vec[k*2]

                if sigma_grid[k,i,j] > 1:
                    sigma_grid[k,i,j] = 1 - ((len(lev)-k)/100000)

    sigma_xr = xr.Dataset(data_vars = dict(sig=(['time', 'height','lat', 'lon' ], [sigma_grid])),
                            coords = dict(time=time, height=lev, lat=lat, lon=lon))

    path = preprocessed + 'sigma_'+hgridres+'s'+str(len(lev))+'.nc'

    sigma_xr.to_netcdf(path)

    return path, len(lev)

