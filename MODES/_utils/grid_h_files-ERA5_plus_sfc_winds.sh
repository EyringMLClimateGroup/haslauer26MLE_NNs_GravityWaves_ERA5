# Script to prepare ERA5 data


simulation2d=$1
suffix=$2
hgridres=$3
preprocessed=$4
timestep=$5


cd ${preprocessed}
mkdir -p temp

cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var165 -seldate,${suffix} ${simulation2d} ${preprocessed}ua_10m_${timestep}_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var166 -seldate,${suffix} ${simulation2d} ${preprocessed}va_10m_${timestep}_${hgridres}.grb









