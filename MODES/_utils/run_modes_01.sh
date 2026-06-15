#module load netCDF-Fortran
#module load Stages/2025


cd $1

ln -sf /p/project1/icon-a-ml/haslauer1/gravity_waves/src/MODES_gcc-11.20/NMF_RUN/Grid/gauss.run gauss.run
cp gauss.cnf_ERA5 gauss.cnf
./gauss.run

ln -sf /p/project1/icon-a-ml/haslauer1/gravity_waves/src/MODES_gcc-11.20/NMF_RUN/Vert/vsf.run vsf.run
cp vsfcalc.cnf_ERA5 vsfcalc.cnf
./vsf.run