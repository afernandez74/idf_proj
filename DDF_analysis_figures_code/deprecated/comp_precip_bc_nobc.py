#sanity check of bias corrected precipitation

import re
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.features import geometry_mask
import pyproj 
import cartopy.feature as cfeature
import os

from funcs import reproject_raster, init_lambert_proj, load_minnesota_reproj, create_dataarray, read_asc_to_geotiff
plt.rcdefaults()
plt.style.use('seaborn-v0_8-poster')
#%%
# load minnesota outline and projection for maps
lambert_proj = init_lambert_proj()
minnesota = load_minnesota_reproj(lambert_proj)

futu_AF = xr.open_dataset('/Users/afer/idf_cmip6_local/ddf_repo_USpatial/corrected_precip/ssp585_2080-2099_CESM2_bc_AF.nc')
futu_Ry = xr.open_dataset('/Users/afer/idf_cmip6_local/ddf_repo_USpatial/corrected_precip/ssp585_2080-2099_CESM2_PRISM_bias_corrected.nc')

futu_AF_box = futu_AF.PREC#.sel(lat=slice(45,46)).sel(lon=slice(-94,-93))
futu_Ry_box = futu_Ry.PREC#.sel(lat=slice(45,46)).sel(lon=slice(-94,-93))
#%%
af_ts = futu_AF_box.mean(dim=['lat','lon'])
ry_ts = futu_Ry_box.mean(dim=['lat','lon'])
#%%
fig,ax = plt.subplots()

ax.plot(af_ts.time.values,af_ts.values, color='r',label = 'af')
ax.plot(af_ts.time.values,ry_ts.values,color='b',label = 'ry')
ax.legend()
#%% total annual precip

af_ts_annual = af_ts.resample(time='YE').sum()
ry_ts_annual = ry_ts.resample(time='YE').sum()
fig,ax = plt.subplots()

ax.plot(af_ts_annual.time.values,af_ts_annual.values, color='r',label = 'AF')
ax.plot(af_ts_annual.time.values,ry_ts_annual.values,color='b',label = 'RY')
ax.legend()

#%%
af_snip = futu_AF.PREC.isel(time = 3000)
ry_snip = futu_Ry.PREC.isel(time = 3000)

(af_snip-ry_snip).plot()
