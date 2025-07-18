"""
Calculates Annual Maximum Series (AMS) values for user specified durations.

Author: Ryan Noe
"""
import os
import xarray as xr
import xclim as xc
import numpy as np
from datetime import datetime
import pint_xarray


xc.set_options(data_validation='log')
xc.set_options(cf_compliance="log")


def remove_edge_cells(dataset, left=5, right=5, top=5, bottom=5):
    """
    Remove the 5 cells around the edge of a dataset. These cells are used for calibraion and do not contain valid data.
    Parameters:
    dataset (xarray.Dataset): The input dataset to be masked.
    left (int, optional): Number of columns to mask on the left edge. Default is 5.
    right (int, optional): Number of columns to mask on the right edge. Default is 5.
    top (int, optional): Number of rows to mask on the top edge. Default is 5.
    bottom (int, optional): Number of rows to mask on the bottom edge. Default is 5.
    Returns:
    xarray.Dataset: The masked dataset with the specified edges removed.
    """

    mask_array = np.ones((dataset['lon'].size, dataset['lat'].size))
    mask_array[0:top, :] = 0  # First 5 rows
    mask_array[-bottom:, :] = 0  # Last 5 rows
    mask_array[:, 0:left] = 0  # First 5 cols
    mask_array[:, -right:] = 0  # Last 5 cols

    dataset.coords['mask'] = (('lon', 'lat'), mask_array)
    return dataset.where(dataset.mask == 1).drop_vars('mask')


def clean_attrs(orig_ds):
    """
    Cleans and adjusts attributes of the input dataset.

    This function performs the following operations on the input dataset:
    1. Converts 3-hourly precipitation data to daily data by multiplying by 8.
    2. Updates the units attribute of the 'PREC_biasadju' and 'PREC' variables to 'mm d-1'.
    3. Sorts the dataset by the 'time' coordinate.

    Parameters:
    orig_ds (xarray.Dataset): The original dataset containing precipitation data.

    Returns:
    xarray.Dataset: The modified dataset with cleaned and adjusted attributes.
    """
    orig_ds['PREC_biasadju'] *= 8 # converting 3-hourly data to daily
    orig_ds['PREC_biasadju'].attrs['units'] = 'mm d-1'
    orig_ds['PREC'] *= 8 # converting 3-hourly data to daily
    orig_ds['PREC'].attrs['units'] = 'mm d-1'
    orig_ds = orig_ds.sortby('time')
    return orig_ds


def add_var(ds, var):
    """
    Uses xclim to calculate precipitation Annual Maximum Series (AMS) for a specific number of days.
    Parameters:
    ds (xarray.Dataset): The input dataset containing precipitation data.
    var (str): The variable name, either 'yearly_precip-inches-max_{n}-day' or 
    'yearly_unadjustedPrecip-inches-max_{n}-day', where {n} is the duration in days of the AMS.
    Returns:
    xarray.DataArray: The resulting DataArray AMS for the specified number of days.
    """
    v = var.split('_')[1] # adjusted or unadjusted precip data
    n = var.split('_')[2].split('-')[0] # number of days to calculate AMS for

    if v == 'adjustedLiessPrecip-inches-max':
        da = xc.atmos.max_n_day_precipitation_amount(ds['PREC_biasadju'], window=int(n), freq='YS').rename(
            {'time': 'yearly'}).pint.quantify().pint.to('inches')
            
    if v == 'unadjustedLiessPrecip-inches-max':
        da = xc.atmos.max_n_day_precipitation_amount(ds['PREC'], window=int(n), freq='YS').rename(
            {'time': 'yearly'}).pint.quantify().pint.to('inches')
    
    if v == 'qmappPrismPrecip-inches-max':
        da = xc.atmos.max_n_day_precipitation_amount(ds['PREC'], window=int(n), freq='YS').rename(
            {'time': 'yearly'}).pint.quantify().pint.to('inches')
            
    if v == 'qmappHuidobroPrecip-inches-max':
        da = xc.atmos.max_n_day_precipitation_amount(ds['PREC'], window=int(n), freq='YS').rename(
            {'time': 'yearly'}).pint.quantify().pint.to('inches')

    # Ensure the dimensions are always in the same order
    da = da.transpose('yearly', 'lat', 'lon')

    return da


if __name__ == '__main__':
    individual_ams_files = os.getenv('AMS_PATH')
    liess_data = '/home/hroop/public/CMIP6/'
    prism_data = '/scratch.global/rrnoe/prism_bias_corrected/'
    huidobro_data = '/scratch.global/rrnoe/huidobro_bias_corrected/'

    precip_variants = [
        'adjustedLiessPrecip',
        'unadjustedLiessPrecip',
        'qmappPrismPrecip',
        'qmappHuidobroPrecip',
    ]

    durations = [1, 2, 3, 4, 5, 7, 10, 20, 30, 45, 60]

    scenarios = [
        'historical_1995-2014',
        'ssp245_2040-2059',
        'ssp245_2060-2079',
        'ssp245_2080-2099',
        'ssp370_2040-2059',
        'ssp370_2060-2079',
        'ssp370_2080-2099',
        'ssp585_2040-2059',
        'ssp585_2060-2079',
        'ssp585_2080-2099',
        ]

    models = ['BCC-CSM2-MR', 'CESM2', 'CMCC-ESM2', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L']

    # Create a list of all possible combinations of scenarios and models (60 total) to send to the cluster
    runs = [] 
    for scenario in scenarios:
        for model in models:
            run_tup = (scenario, model)
            runs.append(run_tup)

    # Each task array gets a different scenario/model combination
    run = runs[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    scenario = run[0]
    model = run[1]

    # individual_ams_files folder must exist already. Creating it in this script causes write conflicts.
    out_path = f'{individual_ams_files}/{scenario}_{model}_AMS.nc'

    # Construct path to source data
    liess_in_path = f'{liess_data}/{scenario}/{scenario}_{model}.nc'
    prism_in_path = f'{prism_data}/{scenario}_{model}_PRISM_bias_corrected.nc'
    huidobro_in_path = f'{huidobro_data}/{scenario}_{model}_Huidobro_bias_corrected.nc'

    print(f'Starting {scenario} {model}')
    start_time = datetime.now()
    liess_ds = xr.open_dataset(liess_in_path, chunks='auto')
    prism_ds = xr.open_dataset(prism_in_path, chunks='auto')
    huidobro_ds = xr.open_dataset(huidobro_in_path, chunks='auto')

    # Remove edge cells
    liess_ds = remove_edge_cells(liess_ds)
    prism_ds = remove_edge_cells(prism_ds)
    huidobro_ds = remove_edge_cells(huidobro_ds)

    # clean attributes
    liess_ds = clean_attrs(liess_ds)
    prism_ds['PREC'].attrs['units'] = 'mm d-1'
    huidobro_ds['PREC'].attrs['units'] = 'mm d-1'

    # Create an empty dataset to store the output
    out_ds = xr.Dataset()

    print('Finished preparing input. Time elapsed:', datetime.now() - start_time)
    # Add variables to the output dataset
    for precip_variant in precip_variants:
        for dur in durations:
            var_str = f'yearly_{precip_variant}-inches-max_{dur}-day'
            if precip_variant == 'qmappPrismPrecip':
                in_ds = prism_ds
            if precip_variant == 'qmappHuidobroPrecip':
                in_ds = huidobro_ds
            if precip_variant == 'adjustedLiessPrecip':
                in_ds = liess_ds
            if precip_variant == 'unadjustedLiessPrecip':
                in_ds = liess_ds
            out_ds[var_str] = add_var(in_ds, var_str)

    # Save the output dataset to a netCDF file
    out_ds = out_ds.pint.dequantify()

    for var_name in out_ds.data_vars:
        out_ds[var_name] = out_ds[var_name].astype('float32')

    encoding = {var: {'dtype': 'float32', 'zlib': True, '_FillValue': np.float32(np.nan)} 
            for var in out_ds.data_vars}

    out_ds.to_netcdf(out_path, mode='w', encoding=encoding)

    print('Finished AMS calc. Time elapsed:', datetime.now() - start_time)
