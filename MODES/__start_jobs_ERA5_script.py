# Script for submitting multiple MODES jobs, including preprocessing and regridding of the results
 
# Specify path for temporary data and desired time steps. This notebook will copy all relevant files in the temporary folder and submit SLURM jobs for the chosen time steps.

import datetime as dt
import subprocess as sp

# Directory where temporary data can be stored
jobs = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/_MODES/_jobs/'

# Select time steps
t_start = dt.datetime(2022,1,1,0)
t_end   = dt.datetime(2022,1,1,23)
t_delta = dt.timedelta(hours=1)

times = []

times.append(t_start)

while times[-1] < t_end:
    times.append(times[-1] + t_delta)


# Copy temporary scripts, start the jobs
for i in range(len(times)):
    tt = times[i].strftime('%Y%m%d%H%M%S')
    sp.run(['mkdir -p ' + jobs + tt], shell=True)

    sp.run(['cp /p/project1/icon-a-ml/haslauer1/gravity_waves/__git/haslauer23_gravitywaves_parameterization/__ERA5/MODES/__run_modes_ERA5.sh ' + jobs + tt + '/___run_modes.sh'], shell=True)
    sp.run(['cp -r /p/project1/icon-a-ml/haslauer1/gravity_waves/__git/haslauer23_gravitywaves_parameterization/__ERA5/MODES/_utils/ ' + jobs + tt + '/_utils/'], shell=True)
    sp.run(['cp /p/project1/icon-a-ml/haslauer1/gravity_waves/__git/haslauer23_gravitywaves_parameterization/__ERA5/MODES/_01_preprocessing_ERA5.py ' + jobs + tt + '/_01_preprocessing_ERA5.py'], shell=True)
    sp.run(['cp /p/project1/icon-a-ml/haslauer1/gravity_waves/__git/haslauer23_gravitywaves_parameterization/__ERA5/MODES/_02_modes_analysis_new_ERA5.py ' + jobs + tt + '/_02_modes_analysis_new_ERA5.py'], shell=True)
    sp.run(['cp /p/project1/icon-a-ml/haslauer1/gravity_waves/__git/haslauer23_gravitywaves_parameterization/__ERA5/MODES/_03_regrid_ERA5.py ' + jobs + tt + '/_03_regrid_ERA5.py'], shell=True)

    sp.run(['chmod -R +x ' + jobs + tt], shell=True)
    sp.run(['cd ' + jobs +tt + '; sbatch --begin=2026-03-05T14:30:00 ___run_modes.sh ' + tt], shell=True)