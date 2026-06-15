# Script to prepare ERA5 data


simulation=$1
simulation2d=$2
fluxes=$3
igtype=$4
suffix=$5
hgridres=$6
preprocessed=$7
timestep=$8


cd ${preprocessed}
mkdir -p temp

cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var131 -seldate,${suffix} ${simulation} ${preprocessed}ua_${timestep}_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var132 -seldate,${suffix} ${simulation} ${preprocessed}va_${timestep}_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var135 -seldate,${suffix} ${simulation} ${preprocessed}wa_${timestep}_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var130 -seldate,${suffix} ${simulation} ${preprocessed}ta_${timestep}_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var133 -seldate,${suffix} ${simulation} ${preprocessed}hus_${timestep}_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var129 -seldate,${suffix} ${simulation} ${preprocessed}geop_${timestep}_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var228 -seldate,${suffix} ${simulation2d} ${preprocessed}precip_${timestep}_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} -selname,var59 -seldate,${suffix} ${simulation2d} ${preprocessed}cape_${timestep}_${hgridres}.grb

cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} ${fluxes}MFx_${igtype}_${timestep}_${hgridres}.grb ${preprocessed}MFx_${igtype}_${timestep}_${hgridres}.grb
cdo -P 12 setmisstonn -sellonlatbox,-180,180,-70,70 -remapcon,${hgridres} ${fluxes}MFy_${igtype}_${timestep}_${hgridres}.grb ${preprocessed}MFy_${igtype}_${timestep}_${hgridres}.grb














# #cdo -P 12 setmisstonn -remapcon,${hgridres} -seldate,${suffix} ${simulation}${prefix}_extra_3d_pfull_${suffix}.grb ${preprocessed}temp/_pfull_${suffix}_${hgridres}.grb

# #cdo -P 12 setmisstonn -remapcon,${hgridres} -sellevel,${vc}/191 /p/scratch/icon-a-ml/haslauer1/_data/_LONGRUN/qubicc_long_run_zfull_ml.grb ${preprocessed}height_files/_zfull_${hgridres}.grb

# cdo -P 12 merge ${preprocessed}temp/_ua_${suffix}_${hgridres}.grb ${preprocessed}temp/_va_${suffix}_${hgridres}.grb ${preprocessed}temp/_ta_${suffix}_${hgridres}.grb ${preprocessed}temp/_hus_${suffix}_${hgridres}.grb ${preprocessed}temp/_uvtq_${suffix}_${hgridres}.grb

# cdo -P 12 setmisstonn -remapcon,${hgridres} -selname,var129 -seltimestep,1 ${simulation2d} ${preprocessed}temp/_zs_${hgridres}.grb

# cdo -P 12 setmisstonn -remapcon,${hgridres} -selname,var134 -seldate,${suffix} ${simulation2d} ${preprocessed}temp/_ps_${suffix}_${hgridres}.grb

# cdo div ${preprocessed}p_levels.nc -enlarge,${preprocessed}p_levels.nc ${preprocessed}temp/_ps_${suffix}_${hgridres}.grb ${preprocessed}height_files/_pdivps_${suffix}_${hgridres}.grb
# cdo -O ln ${preprocessed}temp/_ps_${suffix}_${hgridres}.grb ${preprocessed}temp/_ln_${suffix}_${hgridres}.grb