# Interpretable Neural Networks to Predict Momentum Fluxes of Orographic Gravity Waves

This repository provides the code corresponding to the study

> Haslauer, E., Schwabe, M., Dörnbrack, A., Gerber, E., Rapp, M., &#381;agar, N., Eyring, V.: Interpretable Neural Networks to Predict Momentum Fluxes of Orographic Gravity Waves

submitted to *Machine Learning: Earth*. 

A preprint can be found on arXiv:

> [*https://arxiv.org/abs/2605.05052*](https://arxiv.org/abs/2605.05052)

In this study, we train neural networks on ERA5 reanalysis data to predict momentum fluxes of orographic gravity waves. This repository contains the code for filtering ERA5 data for gravity waves using the software MODES, calculating corresponding momentum fluxes, training neural networks, and evaluation of the results.

---

### Repository Content

The directory **MODES** contains scripts and notebooks for

- downloading ERA5 reanalysis data: `_00_download_ERA5.py`,
- preprocessing data for MODES, including regridding: `01_preprocessing_ERA5.py`,
- applying filtering for gravity waves using MODES: `_02_modes_analysis_new_ERA5.py`,
- regridding of data back to ERA5 grid: `_03_regrid_ERA5.py`,
- calculation of gravity wave momentum fluxes: `_04_fluxes_ERA5.py`,
- visualisation of gravity wave momentum fluxes: `_05_fluxes_visualisation.ipynb`,
- preparing data for machine learning: `_06_ML_data_preparation.py`.

The directory **ML** contains scripts and notebooks for

- assembling training and test datasets: `_01_datasets_script_2024_train.py`, `_01_datasets_script_2022_test.py`,
- training neural networks: `_02_training_script.py`,
- evaluation and visualisation of results: `_03_evaluation_sets.py`, `_03_inference_plot.ipynb`, `_03_R2_calculation.py`,
- analysis of the neural networks using SHAP values: `_04_SHAP_calculation.py`, `_04_SHAP_details.ipynb`,
- comparison of neural networks with conventional parameterisation scheme: `_05_Lott_comparison.ipynb`.

---

### Data and Software

**ERA5 reanalysis data** is described in [*Hersbach et al. (2020)*](https://doi.org/10.1002/qj.3803) and can be obtained at https://cds.climate.copernicus.eu/. For our study, we used the datasets *ERA5 hourly data on pressure levels from 1940 to present* and *ERA5 hourly data on single levels form 1940 to present*.

**MODES** is a software applying three-dimensional linear wave theory for the decomposition of global circulation in terms of normal-mode functions. It is described in [*&#381;agar et al. (2015)*](https://doi.org/10.5194/gmd-8-1169-2015) and can be obtained upon request at https://modes.cen.uni-hamburg.de/.

---

### Dependencies

Most of the code is written in **Python**, except for some shell scripts. `env.yml` contains all dependencies used, and they can be installed with the command `conda env create --name <your_env_name> --file=env.yml`. Main dependencies are:

- CDO
- cdsapi
- Matplotlib
- NumPy
- psyplot
- PyTorch
- Xarray