# Script to prepare ERA5 data (vertical regridding)


simulation=$1
files=$2
prefix=$3
suffix=$4
hgridres=$5
vgridres=$6
vgridfile=$7
preprocessed=$8


cdo intlevelx3d,${vgridfile} -setzaxis,/p/project1/icon-a-ml/haslauer1/gravity_waves/__git/haslauer23_gravitywaves_parameterization/__ERA5/MODES/_utils/zaxis.txt ${preprocessed}temp/_uvtq_${suffix}_${hgridres}.grb ${preprocessed}height_files/_pdivps_${suffix}_${hgridres}.grb ${preprocessed}temp/_uvtqx_${suffix}_${hgridres}${vgridres}.grb
cdo delete,name=var5 ${preprocessed}temp/_uvtqx_${suffix}_${hgridres}${vgridres}.grb ${preprocessed}temp/_uvtq_${suffix}_${hgridres}${vgridres}.grb
rm ${preprocessed}temp/_uvtqx_${suffix}_${hgridres}${vgridres}.grb