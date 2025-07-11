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
from funcs import reproject_raster, init_lambert_proj, load_minnesota_reproj, create_dataarray_temp
import cartopy.feature as cfeature
plt.rcdefaults()
plt.style.use('seaborn-v0_8-poster')

#%% specify stuff

sites = ['MSP',
         'CLOQUET',
         'MANKATO',
         'BRAINERD',
         'BEMIDJI',
         'ROCHESTER_INTL_AP',
         'CROOKSTON_NW',
         'ST_CLOUD'
         ]

lat_lons = [
            [44.8831, -93.2289],
            [46.8369, -92.1833],
            [44.1542, -94.0211],
            [46.3433, -94.2100],
            [47.5369, -94.8297],
            [43.9042, -92.4917],
            [47.8014, -96.6028],
            [45.5433, -94.0514]
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

D_max = 10
RI_max = 100
#%% paths
base_path = '../Data/DDF_individual_model_tif/hist/'

clip_MN = True #if true, raster data clipped to MN 

#save path results
save_path = '../Figures/DDF_curves_single_model_hist_a14/'
save_path = save_path + 'clip_MN/' if clip_MN else save_path + 'whole/'

models = [file for file in os.listdir(base_path) if not file.startswith('.')]

path_models = [base_path + file for file in os.listdir(base_path) if not file.startswith('.')]

# files in all paths
paths_all = []
for path in path_models:
    files = [file for file in os.listdir(path) if not file.startswith('.')]
    paths_temp = [path + '/' + file for file in files]
    paths_all.append(paths_temp)
    
paths_all = [item for row in paths_all for item in row]

del path,path_models

lambert_proj = init_lambert_proj()
shape_path = "/Users/afer/idf_cmip6_local/idf_repo/Data/tl_2022_us_state.zip"
minnesota = load_minnesota_reproj(lambert_proj,shape_path)

#%% open raster files and reproject

files = paths_all
# =============================================================================
# read projection rasters
# =============================================================================
data_futu=[]
for file in files:
    name = file[file.rfind('_') + 1 : file.find('.tif')]
    model = file[file.rfind('/') + 1 : file.rfind('_')]
    data_futu_temp, metadata_temp = reproject_raster(file,lambert_proj.proj4_init)
    data_futu_da = create_dataarray_temp(data_futu_temp, metadata_temp, name, model)
    data_futu.append(data_futu_da)

data_futu = xr.concat(data_futu,dim='source')

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
        
#%% define transformer to locate lat, lon point in crs

data_crs = data_futu.crs
wgs84_crs = "EPSG:4326"

transformer = Transformer.from_crs(wgs84_crs, data_crs,always_xy=True)

x,y = transformer.transform(lon,lat)

#%% data arrays for hist and projections
data_loc = data_futu.sel(x=x,y=y,method='nearest') * 25.4 # change to mm

#%% plot DDF lines (Depth vs RI for varying durations)

cmap = plt.get_cmap('jet', len(models)+1)

colors = [cmap(i) for i in range(len(models))]
dur = 1
stn = 'AGU'
a14= pd.read_csv('../Data/A14/'+stn+'.csv').to_numpy() * 25.4

fig,ax = plt.subplots()

i=0
for model, da in data_loc.groupby('model'):
    da = da.where(da['RI'] <= RI_max, drop = True)
    da = da.sel(source = da.coords['D'] == dur)
    da = da.sortby('RI')
    dat_x = da.RI.values
    dat_y = da.values
    plt.plot(dat_x, dat_y, color = colors[i],label = model, linewidth=1.5)
    i=i+1

ax.plot(dat_x,a14[0].T, label = 'A14', color = 'black', linewidth = 3)


xticks = np.unique(dat_x)

plt.xscale('log')
plt.xlabel("RI")
plt.xticks(xticks, labels=[str(int(tick)) for tick in xticks])
title = f'Depth vs Return Interval curves individual models\n Location: {lat},{lon} {stn}\n historical 1995-2014\n duration {dur:02}days'
plt.title(title)
plt.ylabel("Precipitation Depth")
plt.legend()
plt.grid()
plt.show()
#%% plot DDF lines (Depth vs RI for varying durations) - spread TODO

dur = 1
stn = 'AGU'
a14= pd.read_csv('../Data/A14/'+stn+'.csv').to_numpy()* 25.4

fig,ax = plt.subplots()

means = []
q25 = []
q75 = []
for RIi, da in data_loc.groupby('RI'):
    da = da.where(da['RI'] <= RI_max, drop = True)
    da = da.sel(source = da.coords['D'] == dur)
    if RIi>RI_max:
        break
    means.append(da.mean().values)
    q25.append(da.quantile(0.1).values)
    q75.append(da.quantile(0.9).values)
dat_x = [2,5,10,25,50,100]

ax.plot(dat_x,a14[0], label = 'A14', color = 'black', linewidth = 3)
ax.plot(dat_x,a14[1], color = 'black', linewidth = 0.5)
ax.plot(dat_x,a14[2], color = 'black', linewidth = 0.5)
ax.fill_between(dat_x, a14[1], a14[2], color='black', alpha=0.1)


ax.plot(dat_x,means, linewidth = 3, color = 'red', label = 'mean')
ax.plot(dat_x,q25, linewidth = 0.5, color = 'red')
ax.plot(dat_x,q75, linewidth = 0.5, color = 'red')
ax.fill_between(dat_x, q25, q75, color='red', alpha=0.1)

xticks = np.unique(dat_x)

plt.xscale('log')
plt.xlabel("RI")
plt.xticks(xticks, labels=[str(int(tick)) for tick in xticks])
title = f'Depth vs Return Interval curves individual models\n Location: {lat},{lon} {stn}\n historical 1995-2014\n duration {dur:02}days'
plt.title(title)
plt.ylabel("Precipitation Depth")
plt.legend()
plt.grid()

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path+'DDF_curve_'+f'stn_{stn}da_loc_{lat}_{lon}_dur{dur:02}da)'
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

