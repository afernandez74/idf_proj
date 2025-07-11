import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import rasterio
import re
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
import pyproj 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
plt.rcdefaults()
plt.style.use('seaborn-v0_8-poster')

#%%
path = '/Users/afer/idf_cmip6_local/ddf_repo_USpatial/individual_AMS_files/'
path_2 = '/Users/afer/idf_cmip6_local/idf_repo/Data/AMS_modeled/'

AMS_reg = xr.open_dataset(path+'ssp585_2080-2099_CESM2_AMS.nc')
AMS_bc = xr.open_dataset(path+'ssp585_2080-2099_CESM2_bc_AMS.nc')
AMS_lin_bc = xr.open_dataset(path_2+'ssp585_AMS.nc')

var = 'yearly_unadjustedPrecip-inches-max_1-day'
var2 = 'yearly_precip-inches-max_1-day'


lat = 46
lon = -96

AMS_spot_reg = AMS_reg[var].sel(lat = lat,lon = lon, method = 'nearest')
AMS_spot_bc = AMS_bc[var].sel(lat = lat,lon = lon, method = 'nearest')
AMS_spot_lin_bc = AMS_lin_bc[var2].sel(lat = lat,lon = lon, method = 'nearest').sel(model='CESM2').isel(yearly = range(-20,-1))
#%%
fig, ax = plt.subplots()

ax.plot(AMS_spot_reg.yearly.values, AMS_spot_reg.values, 
            label = 'reg', 
            linewidth = 3,
            )
ax.plot(AMS_spot_bc.yearly.values, AMS_spot_bc.values, 
            label = 'bc', 
            linewidth = 3,
            )
ax.plot(AMS_spot_lin_bc.yearly.values, AMS_spot_lin_bc.values, 
            label = 'lin_bc', 
            linewidth = 3,
            )
ax.legend()
