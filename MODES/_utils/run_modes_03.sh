#module load netCDF-Fortran
#module load Stages/2025

cd $1

ln -sf /p/project1/icon-a-ml/haslauer1/gravity_waves/src/MODES_gcc-11.20/NMF_RUN/Hort/hort_struct.run hort_struct.run
cp houghcalc.cnf_ERA5 houghcalc.cnf
./hort_struct.run