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
plt.rcdefaults()
plt.style.use('seaborn-v0_8-poster')

#%% specify stuff

#specify SSP and period
scenario = 'ssp585'
period = '2080-2099'

# # point coordinates
lat = 48.8947
lon = -95.33

D_max = 10
RI_max = 100
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
    data_hist_da = create_dataarray(data_hist_temp,metadata_temp,name)
    data_hist.append(data_hist_da)
data_hist = xr.concat(data_hist,dim = 'source')

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

plt.figure()

for duration, da in data_loc.groupby('D'):
    da = da.sortby('RI')
    da = da.where(da['RI'] <= RI_max, drop = True)
    dat_x = da.RI.values
    dat_y = da.values
    if duration <= D_max:
        plt.plot(dat_x, dat_y, label = f'{duration} day')

xticks = np.unique(dat_x)

plt.xscale('log')
plt.xlabel("RI")
plt.xticks(xticks, labels=[str(int(tick)) for tick in xticks])
plt.title(f'Depth vs Return Interval curves for varying durations\n Location: {lat},{lon}\n {scenario} {period}')
plt.ylabel("Precipitation Depth")
plt.legend()
plt.grid()
plt.show()

#%% plot DDF lines (Depth vs Duration for varying RI)
plt.figure()

for RI, da in data_loc.groupby('RI'):
    da = da.sortby('D')
    da = da.where(da['D'] <= D_max, drop = True)
    dat_x = da.D.values
    dat_y = da.values
    if RI <= RI_max:
        plt.plot(dat_x, dat_y, label = f'{RI} years')

xticks = np.unique(dat_x)

plt.xscale('log')
plt.xlabel("Duration")
plt.xticks(xticks, labels=[str(int(tick)) for tick in xticks])
plt.title(f'Depth vs Return Interval curves\n Location: {lat},{lon}\n {scenario} {period}')
plt.ylabel("Precipitation Depth")
plt.legend()
plt.grid()
plt.show()

#%% plot DDF lines (Depth vs RI for 1 day) - proj vs hist comparison
dur = 1

stn = 'warroad'
a14= pd.read_csv('../Data/A14/' + stn + '.csv').to_numpy()

#filter data for the required duration
da = data_loc.sel(source = data_loc.coords['D'] == dur)
da_hist = data_loc_hist.sel(source = data_loc_hist.coords['D'] == dur)


fig, ax = plt.subplots()

da = da.sortby('RI')
da = da.where(da['RI'] <= RI_max, drop = True)
dat_x = da.RI.values
dat_y = da.values
ax.plot(dat_x,dat_y, label = 'proj', color = 'blue')

da = da_hist.sortby('RI')
da = da.where(da['RI'] <= RI_max, drop = True)
dat_x = da.RI.values
dat_y = da.values
ax.plot(dat_x,dat_y, label = 'hist', color = 'red')

ax.plot(dat_x,a14.T, label = 'A14', color = 'black')

xticks = np.unique(dat_x)

plt.xscale('log')
plt.xlabel("RI")
plt.xticks(xticks, labels=[str(int(tick)) for tick in xticks])
title = f'Depth vs Return Interval curves for varying durations\n Location: {lat},{lon} {stn}\n {scenario} {period}\n Duration {dur:02} days'
title = title + '\n Adjusted' if adjusted else title + '\n Unadjusted'
plt.title(title)
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
