# Script to prepare ERA5 data (inverse vertical regridding)

source /p/home/jusers/haslauer1/juwels/.bash_profile
conda activate /p/project1/icon-a-ml/mambaforge/envs/haslauer1_modes1

infile=$1
outfile=$2
intermediate=$3
preprocessed=$4
suffix=$5
vgridfile=$6



hgridres=n360

cdo intlevelx3d,${preprocessed}height_files/_pdivps_${suffix}_${hgridres}.grb -selname,u ${infile} ${vgridfile} ${intermediate}_u.grb
cdo intlevelx3d,${preprocessed}height_files/_pdivps_${suffix}_${hgridres}.grb -selname,v ${infile} ${vgridfile} ${intermediate}_v.grb

cdo setzaxis,/p/project1/icon-a-ml/haslauer1/gravity_waves/__git/haslauer23_gravitywaves_parameterization/__ERA5/MODES/_utils/zaxis.txt -delete,name=var131 ${intermediate}_u.grb ${outfile}_u.grb
cdo setzaxis,/p/project1/icon-a-ml/haslauer1/gravity_waves/__git/haslauer23_gravitywaves_parameterization/__ERA5/MODES/_utils/zaxis.txt -delete,name=var131 ${intermediate}_v.grb ${outfile}_v.grb

rm ${intermediate}_u.grb
rm ${intermediate}_v.grb

