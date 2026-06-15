# Script for preprocessing ERA5 data for MODES
# This script loops over multiple time steps (1 day) and calls the script _01_preprocessing_ERA5.py


# %%
import datetime as dt
import subprocess as sp


# %%
# Select desired time steps

t_start = dt.datetime(2024,1,12,23) #letzte 2024-02-29 & 2020-03-21
t_end = dt.datetime(2024,1,13,23)
t_delta = dt.timedelta(hours=1)

times = []

times.append(t_start)

while times[-1] < t_end:
    times.append(times[-1] + t_delta)

# %%
# Call _01_preprocessin_ERA5.py
for i in range(len(times)):
    tt = times[i].strftime('%Y%m%d%H%M%S')
    sp.run(['python /p/project1/icon-a-ml/haslauer1/gravity_waves/__git/haslauer23_gravitywaves_parameterization/__ERA5/MODES/_01_preprocessing_ERA5.py ' + tt], shell=True)