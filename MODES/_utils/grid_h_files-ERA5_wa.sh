# Script to prepare ERA5 data (regridding of vertical velocities)


simulation=$1
simulation2d=$2
files=$3
prefix=$4
suffix=$5
hgridres=$6
preprocessed=$7
timestep=$8



cd ${preprocessed}
mkdir -p temp

cdo -P 12 setmisstonn -remapcon,${hgridres} -selname,var135 -seldate,${suffix} ${simulation} ${preprocessed}wa_${timestep}_${hgridres}.grb