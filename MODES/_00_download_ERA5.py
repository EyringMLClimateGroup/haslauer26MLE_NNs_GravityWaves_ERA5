# Script for downloading ERA5 data at surface and on 37 pressure levels

import cdsapi


# Select desired date

year='2024' #2024-02-29 & 2020-12-21
month='07'
day='01'


# Select paths for 2D and 3D files
# (If only one is needed, set the other to 'False')

file_2d = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/__raw/__'+year+'-'+month+'/'+year+'-'+month+'-'+day+'_2d.grb'
file_3d = '/p/scratch/icon-a-ml/haslauer1/_data/_ERA5/__raw/__'+year+'-'+month+'/'+year+'-'+month+'-'+day+'_3d.grb'


#############################################
# Download ERA5 data

# 2D variables

if file_2d:

    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
            "sea_surface_temperature",
            "surface_pressure",
            "total_precipitation",
            "mean_eastward_gravity_wave_surface_stress",
            "mean_gravity_wave_dissipation",
            "mean_northward_gravity_wave_surface_stress",
            "mean_total_precipitation_rate",
            "total_cloud_cover",
            "total_column_cloud_ice_water",
            "total_column_cloud_liquid_water",
            "convective_precipitation",
            "large_scale_precipitation",
            "angle_of_sub_gridscale_orography",
            "anisotropy_of_sub_gridscale_orography",
            "convective_available_potential_energy",
            "eastward_gravity_wave_surface_stress",
            "eastward_turbulent_surface_stress",
            "geopotential",
            "gravity_wave_dissipation",
            "instantaneous_northward_turbulent_surface_stress",
            "land_sea_mask"
            "northward_gravity_wave_surface_stress",
            "slope_of_sub_gridscale_orography",
            "standard_deviation_of_filtered_subgrid_orography",
            "standard_deviation_of_orography",
            "total_column_water_vapour"],
        "year": [year],
        "month": [month],
        "day": [day],
        "time": [
            "00:00"#, "01:00", "02:00",
            # "03:00", "04:00", "05:00",
            # "06:00", "07:00", "08:00",
            # "09:00", "10:00", "11:00",
            # "12:00", "13:00", "14:00",
            # "15:00", "16:00", "17:00",
            # "18:00", "19:00", "20:00",
            # "21:00", "22:00", "23:00"
        ],
        "data_format": "grib",
        "download_format": "unarchived"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request, file_2d)


# 3D variables

if file_3d:
    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
        ],
        "year": [year],
        "month": [month],
        "day": [day],
        "time": [
            "00:00"#, "01:00", "02:00",
            # "03:00", "04:00", "05:00",
            # "06:00", "07:00", "08:00",
            # "09:00", "10:00", "11:00",
            # "12:00", "13:00", "14:00",
            # "15:00", "16:00", "17:00",
            # "18:00", "19:00", "20:00",
            # "21:00", "22:00", "23:00"
        ],
        "pressure_level": [
            "1", "2", "3",
            "5", "7", "10",
            "20", "30", "50",
            "70", "100", "125",
            "150", "175", "200",
            "225", "250", "300",
            "350", "400", "450",
            "500", "550", "600",
            "650", "700", "750",
            "775", "800", "825",
            "850", "875", "900",
            "925", "950", "975",
            "1000"
        ],
        "data_format": "grib",
        "download_format": "unarchived"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request, file_3d)