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

D = 1 # duration 
RI_max = 100 # return iterval in years

sites = ['MSP',
         'CLOQUET',
         'MANKATO',
         'BRAINERD',
         'BEMIDJI',
         'ROCHESTER_INTL_AP',
         'CROOKSTON_NW',
         'ST_CLOUD',
         'MORRIS',
         'GRAND_RPDS'
         ]

lat_lons = [
            [44.8831, -93.2289],
            [46.8369, -92.1833],
            [44.1542, -94.0211],
            [46.3433, -94.2100],
            [47.5369, -94.8297],
            [43.9042, -92.4917],
            [47.8014, -96.6028],
            [45.5433, -94.0514],
            [45.5903, -95.8747],
            [47.2436, -93.4975]
            ]

print('Sites:')
for i, item in enumerate (sites):
    print(f'{i}:    {item}')

while True:
    try:
        ix = int(input("Enter the number of the site: "))
        if 0 <= ix <= len(sites):
            break
        else:
            print(f"Please enter a number between 0 and {len(sites)-1}.")
    except ValueError:
        print("Please enter a valid number.")

site = sites[ix]
lat,lon = lat_lons[ix]

del ix
#%% Choose rasters to plot
base_path = '../Data/DDF_tif_bc_25/'

# =============================================================================
# Pick data from options
# =============================================================================
clip_MN = True #if true, raster data clipped to MN 

# =============================================================================
# Set paths
# =============================================================================
filenames = []
# Read filenames
for file in os.listdir(base_path):
    if os.path.isfile(os.path.join(base_path, file)) and not file.startswith('.'):
        filenames.append(file)

# Extract unique values for each position
split_filenames = [filename.split('_') for filename in filenames]

# Transpose the list to group by position
unique_values = [set(column) for column in zip(*split_filenames)]

# Convert sets to sorted lists
unique_values = [sorted(values) for values in unique_values]

# strings to identify list of scenarios, models, periods and bias correction schemes
scenario_i = 'historical'
model_i = 'CESM2'
period_i = '1995-2014'
bc_i = 'adjustedLiessPrecip'

# find the lists for scenarios, models, periods and bias corrections
for values in unique_values:
    if scenario_i in values:
        scenarios = values
    if model_i in values:
        models = values
    if period_i in values:
        periods = values
    if bc_i in values:
        bc_sources = values

# =============================================f================================
# Pick scenario 
# =============================================================================

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

scenario = scenarios[ix]
# =============================================================================
# Pick bc_source
# =============================================================================
print('Bias correction sources:')

for i, item in enumerate (bc_sources):
    print(f'{i}:    {item}')

while True:
    try:
        ix3 = int(input("Enter the number of your choice: "))
        if 0 <= ix3 <= len(bc_sources):
            break
        else:
            print(f"Please enter a number between 0 and {len(bc_sources)-1}.")
    except ValueError:
        print("Please enter a valid number.")

bc_source = bc_sources[ix3]
# =============================================================================
# Get all paths to future files
# =============================================================================
strings = [scenario, f'_{bc_source}', f'{D:02}da']
strings_hist = ['historical', f'_{bc_source}', f'{D:02}da']

files = [file for file in filenames if all(s in file for s in strings)]
files_hist = [file for file in filenames if all(s in file for s in strings_hist)]

# =============================================================================
# Get all paths to files
# =============================================================================

paths_all = [base_path + file for file in files]
paths_all_hist = [base_path + file for file in files_hist]

# =============================================================================
# load minnesota outline and projection for maps
# =============================================================================

lambert_proj = init_lambert_proj()
shape_path = "/Users/afer/idf_cmip6_local/idf_repo/Data/tl_2022_us_state.zip"
minnesota = load_minnesota_reproj(lambert_proj,shape_path)
#%% open raster files and reproject

files = paths_all
# =============================================================================
# read projection rasters
# =============================================================================

data_futu=[]
for file in paths_all:
    name = file.split('/')[-1]
    model = name.split('_')[-3]
    bc_source = name.split('_')[-2]
    data_futu_temp, metadata_temp = reproject_raster(file,lambert_proj.proj4_init)
    data_futu_da = create_dataarray(data_futu_temp, metadata_temp, name)
    data_futu_da = data_futu_da.assign_coords(model=model)
    data_futu_da = data_futu_da.assign_coords(bc_source=bc_source)
    data_futu.append(data_futu_da)

data_futu = xr.concat(data_futu,dim='source')

# =============================================================================
# read historic rasters
# =============================================================================
data_hist=[]
for file in paths_all_hist:
    name = file.split('/')[-1]
    model = name.split('_')[-3]
    bc_source = name.split('_')[-2]
    data_hist_temp, metadata_temp = reproject_raster(file,lambert_proj.proj4_init)
    data_hist_da = create_dataarray(data_hist_temp, metadata_temp, name)
    data_hist_da = data_hist_da.assign_coords(model=model)
    data_hist_da = data_hist_da.assign_coords(bc_source=bc_source)
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
data_loc = data_loc.where((data_loc["RI"] <= RI_max), drop=True)
data_loc_hist = data_loc_hist.where((data_loc_hist["RI"] <= RI_max), drop=True)

# convert to mm
data_loc = data_loc * 25.4 # change to mm
data_loc_hist = data_loc_hist * 25.4 # change to mm

# load NA14 data
a14= pd.read_csv('../Data/A14/'+site+'.csv').to_numpy() * 25.4 #and change to mm
#%%hist vs NA14 compares
cmap = plt.get_cmap('jet', 4)

colors = [cmap(i) for i in range(0,4)]

alpha_fill = 0.1
fig,ax = plt.subplots()


mean = []
q25 =[]
q75 = []

da = data_loc_hist
da = da.sortby('RI')
for RI_i, da_2 in da.groupby('RI'):
    mean.append(da_2.mean(dim = 'source').values)
    q25.append(da_2.quantile(q = 0.1).values)
    q75.append(da_2.quantile(q = 0.9).values)
dat_x = np.unique(da.RI.values)
dat_y_1 = mean
dat_y_2 = q25
dat_y_3 = q75
plt.plot(dat_x, dat_y_1, label = '1995-2014', linewidth=1.5, color = 'red')
plt.plot(dat_x, dat_y_2, linewidth=.5, color = 'red', linestyle = '--')
plt.plot(dat_x, dat_y_3, linewidth=.5, color = 'red', linestyle = '--')  
ax.fill_between(dat_x, dat_y_2, dat_y_3, color='red', alpha=alpha_fill)

ax.plot(dat_x,a14[0], label = 'A14', color = 'black', linewidth = 3)
ax.plot(dat_x,a14[1], color = 'black', linewidth = 0.5)
ax.plot(dat_x,a14[2], color = 'black', linewidth = 0.5)
ax.fill_between(dat_x, a14[1], a14[2], color='black', alpha=0.1)

xticks = np.unique(dat_x)

plt.xscale('log')
plt.xlabel("RI")
plt.xticks(xticks, labels=[str(int(tick)) for tick in xticks])
title = f'Intensity vs Return Interval Projections\n Site: {site} \n duration {D:02}days \n Scenario {scenarios[ix]} \n Bias Correction {bc_source}'
plt.title(title)
plt.ylabel("Precipitation Depth")
plt.legend()
plt.grid()

save_option = input("Save figure? (y/n): ").lower()

#save path results
save_path = '../Figures/abs_proj/mean_model_abs_IDF_curves/'

if save_option == 'y':
    save_path_name = save_path+'DDF_curve_'+f'{scenarios[ix]}_{D}da_loc{lat}_{lon}'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
plt.show()

#%% spread of DDF lines through different periods - absolute values

cmap = plt.get_cmap('jet', 4)

colors = [cmap(i) for i in range(0,4)]

alpha_fill = 0.1
fig,ax = plt.subplots()

i=1
mean = []
q25 =[]
q75 = []

for period, da in data_loc.groupby('period'):
    da = da.sortby('RI')
    for RI_i, da_2 in da.groupby('RI'):
        mean.append(da_2.mean(dim = 'source').values)
        q25.append(da_2.quantile(q = 0.2).values)
        q75.append(da_2.quantile(q = 0.8).values)
    dat_x = np.unique(da.RI.values)
    dat_y_1 = mean
    dat_y_2 = q25
    dat_y_3 = q75
    plt.plot(dat_x, dat_y_1, label = period, linewidth=1.5, color = colors[i])
    # plt.plot(dat_x, dat_y_2, linewidth=.5, color = colors[i], linestyle = '--')
    # plt.plot(dat_x, dat_y_3, linewidth=.5, color = colors[i], linestyle = '--')  
    # ax.fill_between(dat_x, dat_y_2, dat_y_3, color=colors[i], alpha=alpha_fill)
    mean = []
    q25 =[]
    q75 = []
    i=i+1

mean = []
q25 =[]
q75 = []

da = data_loc_hist
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

ax.plot(dat_x,a14[0], label = 'A14', color = 'black', linewidth = 3)
ax.plot(dat_x,a14[1], color = 'black', linewidth = 0.5)
ax.plot(dat_x,a14[2], color = 'black', linewidth = 0.5)
ax.fill_between(dat_x, a14[1], a14[2], color='black', alpha=0.1)

xticks = np.unique(dat_x)

plt.xscale('log')
plt.xlabel("RI")
plt.xticks(xticks, labels=[str(int(tick)) for tick in xticks])
title = f'Intensity vs Return Interval Projections\n Site: {site} \n duration {D:02}days \n Scenario {scenarios[ix]} \n Bias Correction {bc_source}'
plt.title(title)
plt.ylabel("Precipitation Depth")
plt.legend()
plt.grid()

save_option = input("Save figure? (y/n): ").lower()

#save path results
save_path = '../Figures/abs_proj/mean_model_abs_IDF_curves/'

if save_option == 'y':
    save_path_name = save_path+'DDF_curve_'+f'{scenarios[ix]}_{D}da_loc{lat}_{lon}'
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

