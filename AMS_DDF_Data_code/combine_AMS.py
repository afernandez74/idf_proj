"""
Combines AMS files into a single file for each time period.

Author: Ryan Noe
"""
import xarray as xr
import glob


def assign_coords(ds):
    """
    Assigns coordinates to an xarray Dataset based on the filename.
    This function extracts the scenario (ssp) and model (gcm) information from the
    filename stored in the dataset's encoding and assigns them as coordinates to the dataset.
    Parameters:
    ds (xarray.Dataset): The input dataset with encoding containing the source filename.
    Returns:
    xarray.Dataset: The dataset with the new coordinates 'scenario' and 'model' assigned.
    """

    fname = ds.encoding["source"].split('/')[-1]
    ssp = fname.split('_')[0] + '_' + fname.split('_')[1]
    gcm = fname.split('_')[2]
    ds = ds.assign_coords(scenario=ssp)
    ds = ds.assign_coords(model=gcm)
    return ds


if __name__ == '__main__':
    ssps = [
        'historical',
        'ssp245',
        'ssp370',
        'ssp585',
        ]

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

    individual_ams_files = '~/precip-intensity/individual_AMS_files/'

    # Creates a nested list structure needed to combined the files into a single dataset
    for ssp in ssps:
        scenarios_subset = [scenario for scenario in scenarios if ssp in scenario]
        nested_ams_files, ams_file_subset = [], []
        for scenario in scenarios_subset:
            for ams_file in glob.glob(f'{individual_ams_files}{scenario}*'):
                ams_file_subset.append(ams_file)
            ams_file_subset.sort()
            nested_ams_files.append(ams_file_subset)
            ams_file_subset = []
        nested_ams_files.sort()

        out_ds = xr.open_mfdataset(nested_ams_files, combine='nested', concat_dim=['scenario', 'model'], preprocess=assign_coords)

        # Remove empty observations and drop the 'scenario' variable
        if ssp == 'historical':
            new_ds = out_ds.sel(scenario=f'{ssp}_1995-2014').dropna(dim='yearly', how='all')
            new_ds = new_ds.drop_vars('scenario')

        else:
            sub1 = out_ds.sel(scenario=f'{ssp}_2040-2059').dropna(dim='yearly', how='all')
            sub2 = out_ds.sel(scenario=f'{ssp}_2060-2079').dropna(dim='yearly', how='all')
            sub3 = out_ds.sel(scenario=f'{ssp}_2080-2099').dropna(dim='yearly', how='all')

            new_ds = xr.concat([sub1, sub2, sub3], dim='yearly')
            new_ds = new_ds.drop_vars('scenario')
        
        
        new_ds.to_netcdf(f'~/precip-intensity/{ssp}_AMS.nc')
