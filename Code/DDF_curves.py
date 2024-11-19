# script for plotting DDF curves

# input value for coordinates and script finds necessary gridcells from which 
# to obtain the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer
import os
import seaborn as sns
import xarray as xr
from funcs import reproject_raster, init_lambert_proj, load_minnesota_reproj, create_dataarray
import cartopy.feature as cfeature
#%% specify stuff

#specify SSP and period
scenario = 'ssp585'
period = '2080-2099'

# # point coordinates
# lat = 44.8851 #N
# lon = -93.2144 #E
lat = 43.6064
lon = -93.3019

#%% paths
base_path = '../Data/DDF_tif/'

# =============================================================================
# Pick data from options
# =============================================================================

ensemble = False # ensemble vs separate
smooth = True # smooth vs unsmooth
adjusted = False # adjusted vs unadjusted

clip_MN = True #if true, raster data clipped to MN 

#save path results
save_path = '../Figures/DDF_viols/'
save_path = save_path + 'clip_MN/' if clip_MN else save_path + 'whole/'

# =============================================================================
# Set paths
# =============================================================================
path = base_path +'ensemble/' if ensemble else base_path + 'separate/'
path = path +'smooth/' if smooth else path + 'unsmooth/'
path = path +'adjusted/' if adjusted else path + 'unadjusted/'

# path to historical data 
path_hist = path + 'historical/1995-2014/'

files_hist_all = [path_hist + file for file in os.listdir(path_hist) if not file.startswith('.')]

# all modeling SSPs
scenarios = [file for file in os.listdir(path) if not file.startswith('.') and not file.startswith('historical')]

# list of paths for all scenarios
paths_scenarios = [path + scenario for scenario in scenarios]

# files in all paths
paths_all = []
for path in paths_scenarios:
    files = [file for file in os.listdir(path) if not file.startswith('.')]
    paths_temp = [path + '/' + file for file in files]
    paths_all.append(paths_temp)

paths_all = [item for row in paths_all for item in row]

files_all = []
for path in paths_all:
    files = [file for file in os.listdir(path) if not file.startswith('.')]
    files_temp = [path + '/' + file for file in files]
    files_all.append(files_temp)

# all paths for all files in all SSPs and all periods 
files_all = [item for row in files_all for item in row]

del files_temp, paths_temp, files, paths_all, paths_scenarios, path

#%% filter paths for desired files

# filter paths for those with specified return interval and duration
files = [s for s in files_all if scenario in s and period in s]

#%% load minnesota outline and projection for maps
lambert_proj = init_lambert_proj()
minnesota = load_minnesota_reproj(lambert_proj)

#%% open raster files and reproject

# =============================================================================
# read projection rasters
# =============================================================================
data_futu=[]
for file in files:
    name = file[file.find('adjusted_') + 9 : file.find('.tif')]
    data_futu_temp, metadata_temp = reproject_raster(file,lambert_proj.proj4_init)
    data_futu_da = create_dataarray(data_futu_temp, metadata_temp, name)
    data_futu.append(data_futu_da)

data_futu = xr.concat(data_futu,dim='source')

# =============================================================================
#  read historical period raster
# =============================================================================
data_hist=[]
metadata_hist=[]
for file in files_hist_all:
    name = file[file.find('adjusted_') + 9 : file.find('.tif')]
    data_hist_temp, metadata_hist = reproject_raster(file,lambert_proj.proj4_init)
    data_hist = create_dataarray(data_hist_temp,metadata_temp,name)

# =============================================================================
# clip data to minnesota shape 
# =============================================================================
metadata = metadata_hist
# define mask 
mask = geometry_mask(
    minnesota.geometry,
    transform = metadata['transform'],
    invert = True,
    out_shape = (metadata['height'],metadata['width'])
    )

# perform the clip
if clip_MN:
    for name, da in data_futu.groupby('source'):
        data_futu.loc[dict(source=name)] = np.where(mask, da.squeeze(), np.nan)
        
    data_hist = data_hist.where(mask)
#%% define transformer to locate lat, lon point in crs

data_crs = data_futu.crs
wgs84_crs = "EPSG:4326"

transformer = Transformer.from_crs(wgs84_crs, data_crs,always_xy=True)

x,y = transformer.transform(lon,lat)

#%% data arrays for hist and projections
data_loc = data_futu.sel(x=x,y=y,method='nearest')
data_loc_hist = data_hist.sel(x=x,y=y,method='nearest')
#%% plot DDF lines (Depth vs RI for varying durations)

plt.rcParams.update({'font.size': 14})

data_loc = data_futu.sel(x=x,y=y,method='nearest')

plt.figure(figsize=(10, 6))

xticks = np.unique(data_loc.RI.values)

for duration, da in data_loc.groupby('D'):
    da = da.sortby('RI')
    dat_x = da.RI.values
    dat_y = da.values
    plt.plot(dat_x, dat_y, label = f'{duration} day')

plt.xscale('log')
plt.xlabel("RI")
plt.xticks(xticks, labels=[str(int(tick)) for tick in xticks])
plt.title(f'Depth vs Return Interval curves for varying durations\n Location: {lat},{lon}\n {scenario} {period}')
plt.ylabel("Precipitation Depth")
plt.legend()
plt.grid()
plt.show()

#%% plot DDF lines (Depth vs Duration for varying RI)

plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(10, 6))

xticks = np.unique(data_loc.D.values)

for RI, da in data_loc.groupby('RI'):
    da = da.sortby('D')
    dat_x = da.D.values
    dat_y = da.values
    plt.plot(dat_x, dat_y, label = f'{RI} years')

plt.xscale('log')
plt.xlabel("Duration")
plt.xticks(xticks, labels=[str(int(tick)) for tick in xticks])
plt.title(f'Depth vs Return Interval curves for varying durations\n Location: {lat},{lon}\n {scenario} {period}')
plt.ylabel("Precipitation Depth")
plt.legend()
plt.grid()
plt.show()


#%%# visualize location
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': lambert_proj})

# Set titles and labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Draw political boundaries and other features, matching the Lambert Conformal projection
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
ax.add_feature(cfeature.RIVERS, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
ax.add_feature(cfeature.LAKES, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)

# Plot the Minnesota boundary on the Lambert Conformal map
minnesota.boundary.plot(ax=ax, color='black', linewidth=1.5)

ax.plot(x,y,'ro',markersize = 10,label='location')

# latitudes and longitudes for map extent
min_lon, min_lat = -97.94, 42.54
max_lon, max_lat = -88.69, 49.97
ax.set_extent([min_lon,max_lon, min_lat, max_lat])
# Add gridlines
ax.gridlines(draw_labels=True, x_inline=False,y_inline=False)
