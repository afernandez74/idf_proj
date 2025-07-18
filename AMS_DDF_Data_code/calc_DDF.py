"""
Calculates Depth-Duration-Frequency values from AMS values with many user defined parameters.

Author: Ryan Noe
"""
import os
import xarray as xr
import numpy as np
import datetime
import rasterio as rio
from multiprocessing import Pool
from lmoments3 import distr
import pandas as pd
from scipy import interpolate


def calc_DDF(ams, duration, aris, correct_ams, series_type, apply_smoothing):
    """
    Calculate Depth-Duration-Frequency (DDF) values.
    Parameters:
    ams (array-like): Annual maximum series values.
    duration (int): Duration for which DDF is calculated (e.g., 1, 2, 3, 4, 7 days).
    aris (array-like): Array of Average Recurrence Intervals (ARIs).
    correct_ams (bool): Whether to apply correction factors to AMS values for short durations.
    series_type (str): Type of series ('ams' for annual maximum series, 'pds' for partial duration series).
    apply_smoothing (bool): Whether to apply smoothing to the calculated precipitation depths.

    Returns:
    array-like: Calculated precipitation depths for each return interval.
    """

    # Optionally correct AMS values for short durations
    if correct_ams and duration in [1, 2, 3, 4, 7]:
        correction_factors = {1: 1.12, 2: 1.04, 3: 1.03, 4: 1.02, 7: 1.01}    
        correction_factor = correction_factors[duration]
        ams = np.multiply(ams, correction_factor)

    # Calculate AEP values depending on the series type
    # We use AMS, but PDS is worth exploring, see this for more info:
    # https://tonyladson.wordpress.com/2017/07/04/converting-between-ey-aep-and-ari/
    if series_type == 'ams':
        aep = 1 / np.array(aris)
    elif series_type == 'pds':
        aep = 1 - np.exp(-1 / np.array(aris))
    else:
        raise ValueError('Invalid series type')
    aep = 1 - aep

    # Fit GEV distribution to AMS values and calculate precip depths for each return interval
    fit_params = distr.gev.lmom_fit(ams)
    precip_depths = distr.gev(**fit_params).ppf(aep)

    if apply_smoothing: # Requires at least 4 ARIs
        spline_func = interpolate.UnivariateSpline(aris, precip_depths)
        precip_depths = spline_func(aris)

    return precip_depths
    
def process_duration(ds, dur, params):
    """
    Processes the duration AMS data to calculate the Depth-Duration-Frequency (DDF) values.
    Parameters:
    ds (xarray.Dataset): The dataset containing precipitation data.
    dur (int): The duration in days for which the DDF is to be calculated.
    params (dict): A dictionary of parameters defining the calculation.

    Returns:
    xarray.DataArray: A DataArray containing the calculated DDF values with dimensions ['recurrence', 'lat', 'lon', 'duration'].
    """
    
    # Construct an AMS for a model or models
    ams = ds[f"yearly_{params['adjustment']}-inches-max_{dur}-day"]
    models_list = params['models']
    models_data = []

    if isinstance(models_list, str):
        ams = ams.sel(model=models_list)
    else:
        for model in models_list:
            model_data = ams.sel(model=model)
            models_data.append(model_data)
        ams = xr.concat(models_data, dim='yearly').values
    
    # Calculate DDF values
    precip_depths = np.apply_along_axis(calc_DDF, 0, ams, dur, params['aris'], params['correct_ams'], params['series_type'], params['apply_smoothing'])

    ddf = xr.DataArray(precip_depths, dims=['recurrence', 'lat', 'lon'], coords={'recurrence': params['aris'], 'lat': ds.coords['lat'], 'lon': ds.coords['lon']})
    return ddf.expand_dims(dim={'duration': [dur]})

def make_ddf_point(ds, lat, lon, params):
    """
    Get precipitation depths for a specific latitude and longitude.
    
    Parameters:
    ds (xarray.Dataset): The dataset containing precipitation data.
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.
    params (dict): A dictionary of parameters defining the calculation.
    
    Returns:
    pd.DataFrame: A DataFrame containing the calculated precipitation depths for each duration and ARI.
    """
    precip_depths = {}

    for dur in params['durs']:
        models_list = params['models']
        models_data = []

        if isinstance(models_list, str):
            ams = ds[f"yearly_{params['adjustment']}-inches-max_{dur}-day"].sel(model=models_list).sel(lat=lat, lon=lon, method='nearest').values

        else:
            for model in models_list:
                model_data = ds[f"yearly_{params['adjustment']}-inches-max_{dur}-day"].sel(model=model).sel(lat=lat, lon=lon, method='nearest').values
                models_data.append(model_data)
        
            ams = np.array(models_data).flatten()

    
        depths = calc_DDF(ams, dur, params['aris'], params['correct_ams'], params['series_type'], params['apply_smoothing'])
        precip_depths[dur] = depths
    

    df = pd.DataFrame(precip_depths, index=params['aris'])
    df.index.name = 'Recurrence'
    df.columns.name = 'Duration (days)'
    return df

def make_ddf_array(ds, params, nc_profile='netcdf', tif_profile='tif'):
    """
    A multi-processing wrapper to generate an xarray ds containing Depth-Duration-Frequency (DDF) data.
    Sends each duration to a seperate process for calculation.

    Parameters:
    ds (xarray.Dataset): The input dataset containing precipitation data.
    params (dict): A dictionary containing parameters for the DDF calculation.
    nc_profile (str): The naming profile for NetCDF files. Default is 'netcdf'.
    tif_profile (str): The naming profile for GeoTIFF files. Default is 'tif'.

    Returns:
    xarray.Dataset: A combined dataset with DDF data concatenated along the 'duration' dimension.
    """

    # num_processes = max(1, int(os.cpu_count() * 0.75))
    num_processes = 12
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_duration, [(ds, dur, params) for dur in params['durs']])
    ddf_combined = xr.concat(results, dim='duration')

    ddf_combined = ddf_combined.to_dataset(name='depths')

    if params['output_nc_folder']:
        out_file_path = create_filename(params, profile=nc_profile)
        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        for var_name in ddf_combined.data_vars:
            ddf_combined[var_name] = ddf_combined[var_name].astype('float32')
            encoding = {var: {'dtype': 'float32', 'zlib': True, '_FillValue': np.float32(np.nan)} 
                    for var in ddf_combined.data_vars}
        ddf_combined.to_netcdf(out_file_path, encoding=encoding, mode='w')


    if params['output_tif_folder']:
        for duration in params['durs']:
            for ari in params['aris']:
                ddf = ddf_combined['depths'].sel(duration=duration).sel(recurrence=ari)
                out_file_path = create_filename(params, profile=tif_profile, duration=duration, ari=ari)  
                write_tif(ddf, out_file_path)

def write_tif(da, out_path):
    """
    Writes a DataArray to a GeoTIFF file with predefined spatial resolution and extent.

    Parameters:
    da (xarray.DataArray): The input data array to be written to a GeoTIFF file.
    out_path (str): The output file path where the GeoTIFF will be saved.

    Notes:
    - The function interpolates the data array to a consistent resolution and extent.
    - The latitude range is interpolated from 42.2795 to 50.8486 with a step of 0.036 because source data are irregular.
    - The data array is flipped along the latitude axis becuase of differing conventions.
    - Missing values are filled with a specific nodata value (-3.4028234663852886e+38), but np.nan is also an option.
    - The spatial dimensions are set to 'lon' and 'lat', and the CRS is set to EPSG:4326.
    - The output GeoTIFF file is saved with a data type of float32.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_lats = np.arange(42.2795, 50.8486, 0.036)
    da = da.interp(lat=new_lats, method='nearest') 
    da = da.isel(lat=slice(None, None, -1))
    da = da.fillna(-3.4028234663852886e+38)
    da.rio.write_nodata(-3.4028234663852886e+38, inplace=True)
    da.rio.set_spatial_dims('lon', 'lat').rio.write_crs("epsg:4326").rio.to_raster(out_path, dtype=np.float32, compress='LZW')

def create_filename(params, profile='default', duration=None, ari=None, lat=None, lon=None):
    """
    Generates a filename based on the provided DDF parameters and naming profile. Can be used to construct a complete or partial path.
    
    Parameters:
    params (dict): A dictionary containing parameters for the DDF calculation.
    profile (str): The naming profile to use. Options are 'default', 'detailed', 'tif', 'csv', 'nested-tifs', or 'netcdf'. Default is 'default'.
    duration (int, optional): The duration in days, required for 'tif' profile.
    ari (int, optional): The Average Recurrence Interval, required for 'tif' profile.
    lat (float, optional): The latitude, required for 'csv' profile.
    lon (float, optional): The longitude, required for 'csv' profile.

    Returns:
    str: The generated filename based on the provided parameters and profile.

    Raises:
    ValueError: If an invalid profile is provided.
    """
    scenario = params['scenario']
    start = params['start']
    end = params['end']
    ams_length = (int(end) - int(start)) + 1
    adjustment = params['adjustment']
    correction = 'corrected' if params['correct_ams'] else 'uncorrected'
    smoothing = 'smoothed' if params['apply_smoothing'] else 'unsmoothed'
    nc_dir = params['output_nc_folder']
    tif_dir = params['output_tif_folder']
    csv_dir = params['output_csv_folder']
    model = params['models']
    if isinstance(model, list) and len(model) == 1:
        model = model[0]
    if isinstance(model, list) and len(model) > 1:
        model = f'separate{len(model)*ams_length}'

    # Add whatever naming pofiles you want here
    if profile == 'default':
        return f'{scenario}_{start}-{end}_{model}'
    elif profile == 'detailed':
        return f'{scenario}_{start}-{end}_{model}_{adjustment}_{correction}_{smoothing}'
    elif profile == 'tif':
        return f'{tif_dir}{scenario}_{start}-{end}_{model}_{adjustment}_{ari}yr{duration:>02}da.tif'
    elif profile == 'nested-tifs':
        return f'{tif_dir}/{adustment}/{scenario}_{start}-{end}/{model}/{scenario}_{start}-{end}_{model}'
    elif profile == 'netcdf':
        return f'{nc_dir}{scenario}_{start}-{end}_{model}.nc'
    elif profile == 'csv':
        return f'{csv_dir}{scenario}_{start}-{end}_{model}_{lat}_{lon}.csv'
    else:
        raise ValueError('Invalid profile')


if __name__ == '__main__':
    local_ams_files = '/users/7/rrnoe/precip-intensity/' # Set this path
    output_nc_folder = '/users/7/rrnoe/precip-intensity/ddf_nc/' # Set this path
    output_tifs_folder = os.getenv('OUT_TIF_PATH')

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

    adjustments = ['adjustedLiessPrecip', 'unadjustedLiessPrecip', 'qmappHuidobroPrecip', 'qmappPrismPrecip']

    """
    in_params (dict): A dictionary containing the following keys:
        durs (list of ints): Duration for which DDF is calculated (e.g., 1, 2, 3, 4, 7, 10 days).
        aris (list of ints): Average Recurrence Intervals (ARIs), (e.g., 2, 5, 10, 25, 50, 100, 200 years).
        models (str or list): 'BCC-CSM2-MR', 'CESM2', 'CMCC-ESM2', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', or a list of any of these.
            Passing a list will treat the models as seperate observations, thus 6 models with 20 years of data each will result in 120 AMS values.
        scenario (str): The scenario name. Valid options are ssp245, ssp585, ssp370, and historical.
        start (str): The start year. Recommended options are 1995, 2040, 2060, and 2080, but can be any year in the dataset.
        end (str): The end year. Recommended options are 2014, 2059, 2079, and 2099, but can be any year in the dataset.
        adjustment (str): The adjustment type ('adjustedLiessPrecip', 'unadjustedLiessPrecip', 'qmappPrismPrecip', 'qmappHuidobroPrecip').
        correct_ams (bool): Whether to apply correction factors to AMS values for short durations. Recommended True. 
        series_type (str): Type of series ('ams' for annual maximum series, 'pds' for partial duration series). Typically ams.
        apply_smoothing (bool): Whether to apply smoothing to the calculated precipitation depths. Recommended True, requires at least 4 ARIs.
        output_nc_folder (str): The output folder for NetCDF files. Leave blank to skip saving NetCDF files.
        output_tif_folder (str): The output folder for GeoTIFF files. Leave blank to skip saving GeoTIFF files.
    """

    in_params = {
        'durs': [1, 2, 3, 4, 7, 10],
        'aris': [2, 5, 10, 25, 50, 100, 200], 
        'models': 'BCC-CSM2-MR', # Use a list to treat multiple models as one time series
        'scenario': 'ssp245',
        'start': '2040',        
        'end': '2059',       
        'adjustment': 'qmappPrismPrecip', 
        'correct_ams': True,
        'series_type': 'ams',
        'apply_smoothing': True,
        'output_nc_folder': output_nc_folder, # Set to None to skip saving NetCDF files
        'output_tif_folder': output_tifs_folder, # Set to None to skip saving GeoTIFF files
        'output_csv_folder': None, # Set to None to skip saving CSV files
    }

    # Define your looping logic here to construct the parameters you want to use to create DDFs
    runs = []
    for in_adjustment in adjustments:
        for in_scenario in scenarios:
            runtup = (in_adjustment, in_scenario)
            runs.append(runtup)

    run = runs[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    in_adjustment = run[0]
    in_scenario = run[1]

    for in_model in models:
        in_ssp = in_scenario.split('_')[0]
        in_period = in_scenario.split('_')[1].split('-')
        in_params['adjustment'] = in_adjustment
        in_params['scenario'] = in_ssp
        in_params['models'] = in_model
        in_params['start'] = in_period[0]
        in_params['end'] = in_period[1]

        # Main execution area
        start_time = datetime.datetime.now()             
        in_ds = xr.open_dataset(f"{local_ams_files}{in_params['scenario']}_AMS.nc").sel(yearly=slice(in_params['start'], in_params['end']))

        # To get DDF values for a specific location(s), create a lat/lon list and use this function
        # The create_filename function could be modified to include station name as an option argument
        if in_params['output_csv_folder']:
            coords = [(44.2394, -95.6308), (44.8831, -93.2289), (48.8947, -95.33)] # Tracy, MSP, Warroad
            for coord in coords:                
                depths = make_ddf_point(in_ds, coord[0], coord[1], in_params)
                file_name = create_filename(in_params, profile='csv', lat=coord[0], lon=coord[1])
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                depths.to_csv(file_name)

        # To get DDF values for the entire dataset, and save as netcdf or tif use this function
        # Modify or define alternative naming profiles in create_filename function
        make_ddf_array(in_ds, in_params, nc_profile='netcdf', tif_profile='tif')

        print(f"{in_ssp}_{in_period[0]}-{in_period[1]}_{in_model} execution time: {(datetime.datetime.now() - start_time).total_seconds()} seconds")
