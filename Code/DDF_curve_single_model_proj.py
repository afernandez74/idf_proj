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


# # point coordinates
lat = 44.5 #N
lon = -93.5 #E

D_max = 10
RI_max = 100
#%% Choose rasters to plot
base_path = '../Data/DDF_individual_model_tif/'

# =============================================================================
# Pick data from options
# =============================================================================

clip_MN = True #if true, raster data clipped to MN 

#save path results
save_path = '../Figures/DDF_curves_single_model_proj/'
save_path = save_path + 'clip_MN/' if clip_MN else save_path + 'whole/'

# =============================================================================
# Set paths
# =============================================================================

# path to historical data 
path_hist = base_path + 'hist/'

# ==============================================================================
# Pick future scenario 
# =============================================================================

scenarios = [file for file in os.listdir(base_path) if not file.startswith('.') and not file.startswith('hist')]

print('Scenarios:')
for i, item in enumerate (scenarios):
    print(f'{i}:    {item}')

while True:
    try:
        ix = int(input("Enter the number of your future scenario choice: "))
        if 0 <= ix <= len(scenarios):
            break
        else:
            print(f"Please enter a number between 0 and {len(scenarios)-1}.")
    except ValueError:
        print("Please enter a valid number.")

path = base_path + scenarios[ix] + '/'

# =============================================================================
# Get all paths to future files
# =============================================================================

files = os.listdir(path)

# list of files in selexted paths - projections
models = [file for file in os.listdir(path) if not file.startswith('.')]
path_durations = [path + file for file in os.listdir(path) if not file.startswith('.')]
path_hist_models = [path_hist + file for file in os.listdir(path_hist) if not file.startswith('.')]

# =============================================================================
# Get all paths to files
# =============================================================================

# files in all paths future projections
paths_models = []
for pathy in path_durations:
    model = pathy[pathy.rfind('/')+1:]
    files = [file for file in os.listdir(pathy) if not file.startswith('.')]
    paths_temp = [path + model + '/' + file for file in files]
    paths_models.append(paths_temp)

paths_models = [item for row in paths_models for item in row]

paths_all = []
for pathy in paths_models:
    files = [file for file in os.listdir(pathy) if not file.startswith('.')]
    paths_temp = [pathy +'/' + file for file in files]
    paths_all.append(paths_temp)

paths_all = [item for row in paths_all for item in row]


# files in all paths historical
# paths_models_hist = []
# for pathy in path_durations:
#     model = pathy[pathy.rfind('/')+1:]
#     files = [file for file in os.listdir(pathy) if not file.startswith('.')]
#     paths_temp = [path + model + '/' + file for file in files]
#     paths_models_hist.append(paths_temp)

# paths_models_hist = [item for row in paths_models_hist for item in row]

paths_all_hist = []
for pathy in path_hist_models:
    files = [file for file in os.listdir(pathy) if not file.startswith('.')]
    paths_temp = [pathy + '/' + file for file in files]
    paths_all_hist.append(paths_temp)

paths_all_hist = [item for row in paths_all_hist for item in row]

#%%#%% load minnesota outline and projection for maps
lambert_proj = init_lambert_proj()
minnesota = load_minnesota_reproj(lambert_proj)
#%% open raster files and reproject

files = paths_all
# =============================================================================
# read projection rasters
# =============================================================================
data_futu=[]
for file in files:
    name = file[file.rfind('unadjusted') + len('unadjusted_') : file.find('.tif')]
    model = name.split('_')[-2] 
    data_futu_temp, metadata_temp = reproject_raster(file,lambert_proj.proj4_init)
    data_futu_da = create_dataarray(data_futu_temp, metadata_temp, name)
    data_futu_da = data_futu_da.assign_coords(model=model)
    data_futu.append(data_futu_da)

data_futu = xr.concat(data_futu,dim='source')

data_hist=[]
for file in paths_all_hist:
    name = 'historical_' + file[file.rfind('/') + 1 : file.find('.tif')]
    model = name.split('_')[-2]
    data_hist_temp, metadata_temp = reproject_raster(file,lambert_proj.proj4_init)
    data_hist_da = create_dataarray(data_hist_temp, metadata_temp, name)
    data_hist_da = data_hist_da.assign_coords(model=model)
    data_hist.append(data_hist_da)

data_hist = xr.concat(data_hist,dim='source')

# =============================================================================
# clip data to minnesota shape 
# =============================================================================
metadata = metadata_temp

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
if clip_MN:
    for name, da in data_hist.groupby('source'):
        data_hist.loc[dict(source=name)] = np.where(mask, da.squeeze(), np.nan)

del data_hist_temp,data_futu_temp, data_futu_da, data_hist_da, metadata_temp
#%% define transformer to locate lat, lon point in crs

data_crs = data_futu.crs
wgs84_crs = "EPSG:4326"

transformer = Transformer.from_crs(wgs84_crs, data_crs,always_xy=True)

x,y = transformer.transform(lon,lat)


#%% data arrays for hist and projections
data_loc = data_futu.sel(x=x,y=y,method='nearest')
data_loc_hist = data_hist.sel(x=x,y=y,method='nearest')

# discard unnecessary data
data_loc = data_loc.where((data_loc["RI"] <= RI_max) & (data_loc["D"] <= D_max), drop=True)
data_loc_hist = data_loc_hist.where((data_loc_hist["RI"] <= RI_max) & (data_loc_hist["D"] <= D_max), drop=True)
#%% spread of DDF lines through different periods - absolute values

cmap = plt.get_cmap('jet', 4)

colors = [cmap(i) for i in range(0,4)]

alpha_fill = 0.1
fig,ax = plt.subplots()

i=1
dur = 1
mean = []
q25 =[]
q75 = []

for period, da in data_loc.groupby('period'):
    da = da.sel(source = da.coords['D'] == dur)
    da = da.sortby('RI')
    for RI_i, da_2 in da.groupby('RI'):
        mean.append(da_2.mean(dim = 'source').values)
        q25.append(da_2.quantile(q = 0.25).values)
        q75.append(da_2.quantile(q = 0.75).values)
    dat_x = np.unique(da.RI.values)
    dat_y_1 = mean
    dat_y_2 = q25
    dat_y_3 = q75
    plt.plot(dat_x, dat_y_1, label = period, linewidth=1.5, color = colors[i])
    plt.plot(dat_x, dat_y_2, linewidth=.5, color = colors[i], linestyle = '--')
    plt.plot(dat_x, dat_y_3, linewidth=.5, color = colors[i], linestyle = '--')  
    ax.fill_between(dat_x, dat_y_2, dat_y_3, color=colors[i], alpha=alpha_fill)
    mean = []
    q25 =[]
    q75 = []
    i=i+1

mean = []
q25 =[]
q75 = []

da = data_loc_hist
da = da.sel(source = da.coords['D'] == dur)
da = da.sortby('RI')
for RI_i, da_2 in da.groupby('RI'):
    mean.append(da_2.mean(dim = 'source').values)
    q25.append(da_2.quantile(q = 0.25).values)
    q75.append(da_2.quantile(q = 0.75).values)
dat_x = np.unique(da.RI.values)
dat_y_1 = mean
dat_y_2 = q25
dat_y_3 = q75
plt.plot(dat_x, dat_y_1, label = '1995-2014', linewidth=1.5, color = colors[0])
plt.plot(dat_x, dat_y_2, linewidth=.5, color = colors[0], linestyle = '--')
plt.plot(dat_x, dat_y_3, linewidth=.5, color = colors[0], linestyle = '--')  
ax.fill_between(dat_x, dat_y_2, dat_y_3, color=colors[0], alpha=alpha_fill)


xticks = np.unique(dat_x)

plt.xscale('log')
plt.xlabel("RI")
plt.xticks(xticks, labels=[str(int(tick)) for tick in xticks])
title = f'Depth vs Return Interval spread\n Location: {lat},{lon} \n duration {dur:02}days \n Scenario {scenarios[ix]}'
plt.title(title)
plt.ylabel("Precipitation Depth")
plt.legend()
plt.grid()

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path+'DDF_curve_'+f'{scenarios[ix]}_{dur}da_loc{lat}_{lon}'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
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

