"""
Uses quantile mapping found in `QMAPP_extrap.py` to bias correct the downscaled CMIP6 data.

Author: Ryan Noe
"""
import os
import xarray as xr
import numpy as np
from datetime import datetime
from QMAPP_extrap import QMAPP_extrap
from dask.distributed import Client, LocalCluster


def qmapp_wrapper(obs, hist, futu):
    """Wrapper for QMAPP_extrap to work with apply_ufunc"""
    if np.isnan(futu).all() or np.isnan(hist).all() or np.isnan(obs).all():
        return np.full_like(futu, np.nan)
    return QMAPP_extrap(obs, hist, futu)


if __name__ == '__main__':
    source_data = '/home/hroop/public/CMIP6/'

    in_obs_path = os.getenv('OBS_DATA_PATH')

    obs_source = in_obs_path.split('/')[-1].split('_')[0]

    output_data = os.getenv('BIAS_CORRECTED_PATH')

    chunks = {'time': -1, 'lat': 16, 'lon': 15}

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

    cluster = LocalCluster(
        n_workers=30,
        threads_per_worker=1,
        memory_limit='1GB',
        processes=True,
        scheduler_port=0,
        dashboard_address=None
    )
    client = Client(cluster)

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

    # Construct path to source data
    in_cmip6_future_path = f'{source_data}/{scenario}/{scenario}_{model}.nc'
    in_cmip6_hist_path = f'{source_data}/historical_1995-2014/historical_1995-2014_{model}.nc'

    in_future_ds = xr.open_dataset(in_cmip6_future_path)[['PREC']].chunk(chunks)
    in_hist_ds = xr.open_dataset(in_cmip6_hist_path)[['PREC']].chunk(chunks)
    in_obs_ds = xr.open_dataset(in_obs_path)[['PREC']].chunk(chunks)

    print("Starting processing...")
    start_time = datetime.now()

    result = xr.apply_ufunc(
        qmapp_wrapper,
        in_obs_ds,
        in_hist_ds,
        in_future_ds,
        input_core_dims=[['time'], ['time'], ['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        join='override',
        output_dtypes=[float],
    )
    
    result = result.assign_coords(time=in_future_ds.time)
    result.to_netcdf(f'{output_data}/{scenario}_{model}_{obs_source}_bias_corrected.nc', mode='w')
    client.close()
    print(f'Time elapsed: {datetime.now() - start_time}')
    cluster.close()
