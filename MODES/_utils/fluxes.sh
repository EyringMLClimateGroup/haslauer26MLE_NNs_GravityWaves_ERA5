# Calculate momentum fluxes
# MF_x, MF_y = -1/g \overline{u * \omega}, -1/g \overline{v * \omega}


infile_1=$1
infile_2=$2
infile_w=$3
target_1=$4
target_2=$5
targetgrid=$6

g=-9.81



cdo setgrid,n360 ${infile_1} ${infile_1}_temp
cdo setgrid,n360 ${infile_2} ${infile_2}_temp
cdo setgrid,n360 ${infile_w} ${infile_w}_temp

cdo divc,${g} -mul ${infile_1}_temp ${infile_w}_temp ${target_1}_temp
cdo divc,${g} -mul ${infile_2}_temp ${infile_w}_temp ${target_2}_temp

cdo remapcon,${targetgrid} ${target_1}_temp ${target_1}
cdo remapcon,${targetgrid} ${target_2}_temp ${target_2}

rm ${infile_1}_temp
rm ${infile_2}_temp
rm ${infile_w}_temp
rm ${target_1}_temp
rm ${target_2}_temp

