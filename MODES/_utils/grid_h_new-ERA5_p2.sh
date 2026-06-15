# Script to prepare ERA5 data (horizontal regridding part 2)


simulation=$1
simulation2d=$2
files=$3
prefix=$4
suffix=$5
hgridres=$6
vc=$7
preprocessed=$8



cdo div ${preprocessed}p_levels.nc -enlarge,${preprocessed}p_levels.nc ${preprocessed}temp/_ps_${suffix}_${hgridres}.grb ${preprocessed}height_files/_pdivps_${suffix}_${hgridres}.grb
cdo -O ln ${preprocessed}temp/_ps_${suffix}_${hgridres}.grb ${preprocessed}temp/_ln_${suffix}_${hgridres}.grb