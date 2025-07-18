"""
Align PRISM and Huidobro datasets with CMIP6 data so bias correction can be applied.

Author: Ryan Noe
"""
import xarray as xr
import os


def align_netcdf(reference_file, target_file, output_file):
    """
    Aligns the target NetCDF file to the reference NetCDF file and saves the result.

    This function reads a reference NetCDF file and a target NetCDF file, aligns the target
    file to the reference file using nearest neighbor interpolation, and saves the aligned
    dataset to a specified output file.

    Parameters:
    reference_file (str): Path to the reference NetCDF file.
    target_file (str): Path to the target NetCDF file that needs to be aligned.
    output_file (str): Path where the aligned NetCDF file will be saved.

    Returns:
    None
    """
    ref_ds = xr.open_dataset(reference_file)
    target_ds = xr.open_dataset(target_file)
    aligned_ds = target_ds.interp_like(ref_ds, method='nearest')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    aligned_ds.to_netcdf(output_file)
    

if __name__ == "__main__":
    # Any historical file will do, but I chose CESM2
    reference_file = '/home/hroop/public/CMIP6/historical_1995-2014/historical_1995-2014_CESM2.nc'

    # obs data provided by Alejandro
    target_file_folder = '~/precip-intensity/obs_data/'
    output_file_folder = '~/precip-intensity/aligned_obs_data/'

    target_files = ['Huidobro_PREC.nc', 'PRISM_PREC.nc']

    for target_file in target_files:
        target_file_path = f'{target_file_folder}{target_file}'
        output_file_path = f'{output_file_folder}{target_file}'
        align_netcdf(reference_file, target_file_path, output_file_path)
