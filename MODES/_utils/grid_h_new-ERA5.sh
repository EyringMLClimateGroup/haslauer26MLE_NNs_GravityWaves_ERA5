# Script to prepare ERA5 data (horizontal regridding part 1)


simulation=$1
simulation2d=$2
files=$3
prefix=$4
suffix=$5
hgridres=$6
vc=$7
preprocessed=$8


cd ${preprocessed}
mkdir -p temp

cdo -P 12 setmisstonn -remapcon,${hgridres} -selname,var131 -seldate,${suffix} ${simulation} ${preprocessed}temp/_ua_${suffix}_${hgridres}.grb
cdo -P 12 setmisstonn -remapcon,${hgridres} -selname,var132 -seldate,${suffix} ${simulation} ${preprocessed}temp/_va_${suffix}_${hgridres}.grb
cdo -P 12 setmisstonn -remapcon,${hgridres} -selname,var130 -seldate,${suffix} ${simulation} ${preprocessed}temp/_ta_${suffix}_${hgridres}.grb
cdo -P 12 setmisstonn -remapcon,${hgridres} -selname,var133 -seldate,${suffix} ${simulation} ${preprocessed}temp/_hus_${suffix}_${hgridres}.grb
cdo -P 12 merge ${preprocessed}temp/_ua_${suffix}_${hgridres}.grb ${preprocessed}temp/_va_${suffix}_${hgridres}.grb ${preprocessed}temp/_ta_${suffix}_${hgridres}.grb ${preprocessed}temp/_hus_${suffix}_${hgridres}.grb ${preprocessed}temp/_uvtq_${suffix}_${hgridres}.grb
cdo -P 12 setmisstonn -remapcon,${hgridres} -selname,var129 -seltimestep,1 ${simulation2d} ${preprocessed}temp/_zs_${hgridres}.grb
cdo -P 12 setmisstonn -remapcon,${hgridres} -selname,var134 -seldate,${suffix} ${simulation2d} ${preprocessed}temp/_ps_${suffix}_${hgridres}.grb