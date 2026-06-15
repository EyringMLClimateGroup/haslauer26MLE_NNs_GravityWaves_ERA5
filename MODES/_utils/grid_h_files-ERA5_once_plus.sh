# Script to prepare ERA5 data


simulation=$1
simulation2d=$2
files=$3
prefix=$4
suffix=$5
hgridres=$6
preprocessed=$7



cd ${preprocessed}
mkdir -p temp

#2d
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var129 -seldate,${suffix} ${simulation2d} ${preprocessed}_geop_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var160 -seldate,${suffix} ${simulation2d} ${preprocessed}_oro_std_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var161 -seldate,${suffix} ${simulation2d} ${preprocessed}_oro_anis_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var162 -seldate,${suffix} ${simulation2d} ${preprocessed}_oro_angle_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var163 -seldate,${suffix} ${simulation2d} ${preprocessed}_oro_slope_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var172 -seldate,${suffix} ${simulation2d} ${preprocessed}_land_sea_mask_${hgridres}.grb