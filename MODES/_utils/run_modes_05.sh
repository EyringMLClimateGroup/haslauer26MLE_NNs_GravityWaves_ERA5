#module load netCDF-Fortran
#module load Stages/2025

cd $1

ln -sf /p/project1/icon-a-ml/haslauer1/gravity_waves/src/MODES_gcc-11.20/NMF_RUN/Inv/inversion.run inversion.run
cp normal_inverse.cnf_ERA5 normal_inverse.cnf
./inversion.run