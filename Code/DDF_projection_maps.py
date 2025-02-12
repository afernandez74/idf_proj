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
from funcs import reproject_raster, init_lambert_proj, load_minnesota_reproj, create_dataarray, read_asc_to_geotiff
plt.rcdefaults()
plt.style.use('seaborn-v0_8-poster')

#%% Set parameters of analysis
# =============================================================================
# Pick data from options
# =============================================================================

clip_MN = True #if true, raster data clipped to MN 

# return interval [years]
RI = 10

#duration [days]
D = 1

#save path results
save_path = '../Figures/DDF_projs/'
save_path = save_path + 'clip_MN/' if clip_MN else save_path + 'whole/'

# load minnesota outline and projection for maps
lambert_proj = init_lambert_proj()
minnesota = load_minnesota_reproj(lambert_proj)

#%% paths for historical and projection rasters 

# =============================================================================
# Set paths
# =============================================================================
#path of DDF projection rasters
base_path = '../Data/DDF_individual_model_tif/'


# path of DDF values for historical simulation
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
# Pick future period
# =============================================================================

periods = [file for file in os.listdir(path) if not file.startswith('.')]
print('Periods:')

for i, item in enumerate (periods):
    print(f'{i}:    {item}')

while True:
    try:
        ix2 = int(input("Enter the number of your choice: "))
        if 0 <= ix2 <= len(periods):
            break
        else:
            print(f"Please enter a number between 0 and {len(periods)-1}.")
    except ValueError:
        print("Please enter a valid number.")

path = path + periods[ix2] + '/'

# =============================================================================
# Get all paths to future files
# =============================================================================

files = os.listdir(path)

# list of files in selexted paths - projections
models = [file for file in os.listdir(path) if not file.startswith('.')]
path_models = [path + file for file in os.listdir(path) if not file.startswith('.')]
path_hist_models = [path_hist + file for file in os.listdir(path_hist) if not file.startswith('.')]

# =============================================================================
# Get all paths to files
# =============================================================================

# files in all paths future projections
paths_all = []
for pathy in path_models:
    model = pathy[pathy.rfind('/')+1:]
    files = [file for file in os.listdir(pathy) if not file.startswith('.')]
    paths_temp = [path + model + '/' + file for file in files]
    paths_all.append(paths_temp)

paths_all = [item for row in paths_all for item in row]

#filter for desired duration and RI
paths_all = [file for file in paths_all if f'{D:02}da' in file and f'_{RI}yr' in file]

# files in all paths historical
paths_all_hist = []
for pathy in path_hist_models:
    model = pathy[pathy.rfind('/')+1:]
    files = [file for file in os.listdir(pathy) if not file.startswith('.')]
    paths_temp = [path_hist + model + '/' + file for file in files]
    paths_all_hist.append(paths_temp)

paths_all_hist = [item for row in paths_all_hist for item in row]

#filter for desired duration and RI
paths_all_hist = [file for file in paths_all_hist if f'{D:02}da' in file and f'{RI}yr' in file]

del path_models, pathy, paths_temp, files, i
#%% open raster files and reproject

# =============================================================================
# read projection rasters
# =============================================================================
data_futu=[]
for file in paths_all:
    name = file[file.rfind('unadjusted') + len('unadjusted_') : file.find('.tif')]
    model = name.split('_')[-2] 
    data_futu_temp, metadata_temp = reproject_raster(file,lambert_proj.proj4_init)
    data_futu_da = create_dataarray(data_futu_temp, metadata_temp, name)
    data_futu_da = data_futu_da.assign_coords(model=model)
    data_futu.append(data_futu_da)

data_futu = xr.concat(data_futu,dim='source')

# =============================================================================
# read historic rasters
# =============================================================================
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
#%% Calculate the change between future and historical data
change = data_futu.copy()

for idx in range(len(data_futu['source'])):
    # Get the model corresponding to the current index
    model_str = str(data_futu['model'].values[idx])
    
    # Select the data for the current model from both DataArrays using integer indexing on 'source'
    da1_model = data_hist.isel(source=idx)
    da2_model = data_futu.isel(source=idx)
    
    # Perform the percent change calculation (avoid division by zero by masking NaNs)
    percent_change = ((da2_model / da1_model)-1)*100
    
    # Assign the calculated percent change to the corresponding location in the 'change' DataArray
    change.loc[dict(source=data_futu.source[idx])] = percent_change

change_mean = change.mean(dim = 'source',skipna=True)
change_lo = change.quantile(q=.25,dim = 'source',skipna=True)
change_hi = change.quantile(q=.75,dim = 'source',skipna=True)

#%% Locate raster file from NA14 data and convert .asc to .tif

base_path = '../Data/A14_raster/'
    
# obtain path to NA14 file
files = [file for file in os.listdir(base_path) if not file.startswith('.')]

file = [file for file in files if f'mw{RI}yr' in file and f'{D:02}da' in file]

file = [file for file in file if file.endswith('.asc')]

name = file[0]
file_path = base_path + name

# Convert .asc file to geotiff
read_asc_to_geotiff(file_path, file_path.replace('.asc','.tif'))

#%% open raster files and reproject (NA14)

# Reproject the NA14 GeoTIFF
NA14_reproj, NA14_metadata = reproject_raster(file_path, lambert_proj.proj4_init)

# create data array from reprojected NA14 raster

transform = NA14_metadata['transform']
width = NA14_metadata['width']
height = NA14_metadata['height']
crs = NA14_metadata['crs']

# Calculate the x and y coordinates using the affine transform
x_coords = np.arange(width) * transform[0] + transform[2]
y_coords = np.arange(height) * transform[4] + transform[5]
    
RI_da, D_da = map(int, re.search(r"(\d+)yr(\d+)da", name).groups())

# Create DataArray
data_NA14 = xr.DataArray(
    NA14_reproj,
    dims=["y", "x"],
    coords={
        "x": ("x", x_coords),
        "y": ("y", y_coords),
        "RI":RI_da,
        "D":D_da,
        "source":f'NA14_{RI_da}yr{D_da}da'

    },
    attrs={"crs": str(crs),
           "transform": transform,
           "name": name
            }
)
#resample NA14 data to same resolution as CMIP6 data
data_NA14_resamp = data_NA14.interp_like(data_hist.isel(source = 0), method = 'linear')

# apply minnesota mask to resampled data
NA14_clip = data_NA14_resamp.where(mask, np.nan)

# divide by 1000 to obtain [in] dimensions
NA14_clip = NA14_clip/1000

NA14 = NA14_clip

# get rid of negative values and outliers
NA14.values[NA14.values < 0.0] = np.nan
threshold = 3
cond = (NA14 < NA14.mean() - threshold * NA14.std()) | (NA14 > NA14.mean() + threshold * NA14.std())

NA14 = NA14.where(~cond,np.nan)

del NA14_clip,data_NA14_resamp,data_NA14, transform, width, height, crs, NA14_reproj, RI_da, D_da
#%% Calculate bias (NA14 / CMIP6) for each model

bias = data_hist.copy()

for idx in range(len(data_hist['source'])):
    # Get the model corresponding to the current index
    model_str = str(data_hist['model'].values[idx])
    
    # Select the data for the current model from both DataArrays using integer indexing on 'source'
    da_i = data_hist.isel(source=idx)

    # Perform the percent change calculation (avoid division by zero by masking NaNs)
    bias_i = (da_i / NA14.values)
    
    # Assign the calculated percent change to the corresponding location in the 'change' DataArray
    bias.loc[dict(source=data_hist.source[idx])] = bias_i
    
    

#%% plot maps of bias corrected historical IDF vals



# Set up the figure and axis with Lambert Conformal projection
fig, ax = plt.subplots(subplot_kw={'projection': lambert_proj})

# Set titles and labels
ax.set_title(f'Event depth change from historical to {periods[ix2]}\n{RI}-year event, {D}-day duration, {scenarios[ix]} Scenario')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Draw political boundaries and other features, matching the Lambert Conformal projection
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
ax.add_feature(cfeature.RIVERS, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
ax.add_feature(cfeature.LAKES, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)

# Plot the Minnesota boundary on the Lambert Conformal map
minnesota.boundary.plot(ax=ax, color='black', linewidth=1.5)

extent = [
    metadata['transform'][2],
    metadata['transform'][2] + metadata['transform'][0] * metadata['width'],
    metadata['transform'][5] + metadata['transform'][4] * metadata['height'],
    metadata['transform'][5]
]

#transformer for custom extent values
transformer = pyproj.Transformer.from_crs(metadata['crs'], lambert_proj.proj4_init, always_xy=True)

# latitudes and longitudes for map extent
min_lon, min_lat = -97.94, 42.54
max_lon, max_lat = -88.69, 49.97

# =============================================================================
# Contour map
# =============================================================================

# levels for contours:
if np.nanmax(change_mean) >100:
    levels = np.concatenate([
        np.linspace(-50, 0, 2, endpoint = False),
        np.linspace(0, 100, 4, endpoint = False),
        np.linspace(100, np.nanmax(change_mean), 2)
        ])
else:
    levels = np.concatenate([
        np.linspace(-50, 0, 2, endpoint = False),
        np.linspace(0, 100, 4, endpoint = False)])

#normalize values 
norm = plt.Normalize(vmin=-75, vmax = 75)

cax = ax.contourf(change_mean, transform = lambert_proj,extent = extent,
                cmap='RdYlBu', origin='upper',
                levels = levels,
                norm=norm,
                )

ax.set_extent([min_lon,max_lon, min_lat, max_lat])

# Add a colorbar with label
plt.colorbar(cax, ax=ax, orientation='vertical', label='% Change')

# # Calculate statistics for display once
# mean_change= np.nanmean(change)
# max_change = np.nanmax(change)
# min_change= np.nanmin(change)

# # Add statistical information as text in a white box
# stats_text = f'Mean: {mean_change:.2f}x\nMax: {max_change:.2f}x\nMin: {min_change:.2f}x'
# props = dict(boxstyle='round', facecolor='white', alpha=0.8)
# ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=props)

# Add gridlines
ax.gridlines(draw_labels=True, x_inline=False,y_inline=False)

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path+'DDF_change_'+f'{scenarios[ix]}_{periods[ix2]}_{RI}yr_{D}da'
    save_path_name = save_path_name + '_MNclip' if clip_MN else save_path_name
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
    
#%% plot change of futu/hist for chosen data / duration / return interval combo (single model grid)

# Set up the figure and axis with Lambert Conformal projection
fig, ax = plt.subplots(nrows=2, ncols=3,subplot_kw={'projection': lambert_proj})
ax = ax.flatten()

# Add a title for the entire figure
fig.suptitle(f'Projceted change from historical to {periods[ix2]}\n{RI}-year event, {D}-day duration, {scenarios[ix]} Scenario',fontsize = 20)

for idx, source in enumerate(change['source']):
    ax_j = ax[idx]
    
    change_j = change.isel(source = idx)
    
    # Set titles and labels
    ax_j.set_title(source.model.values)

    # Draw political boundaries and other features, matching the Lambert Conformal projection
    ax_j.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax_j.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    ax_j.add_feature(cfeature.RIVERS, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
    ax_j.add_feature(cfeature.LAKES, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
    
    # Plot the Minnesota boundary on the Lambert Conformal map
    minnesota.boundary.plot(ax=ax_j, color='black', linewidth=1.5)
    
    extent = [
        metadata['transform'][2],
        metadata['transform'][2] + metadata['transform'][0] * metadata['width'],
        metadata['transform'][5] + metadata['transform'][4] * metadata['height'],
        metadata['transform'][5]
    ]
    
    #transformer for custom extent values
    transformer = pyproj.Transformer.from_crs(metadata['crs'], lambert_proj.proj4_init, always_xy=True)
    
    # latitudes and longitudes for map extent
    min_lon, min_lat = -97.94, 42.54
    max_lon, max_lat = -88.69, 49.97
    
    # =============================================================================
    # Contour map
    # =============================================================================
    
    # levels for contours:
    if np.nanmax(change_j) >100:
        levels = np.concatenate([
            np.linspace(-50, 0, 2, endpoint = False),
            np.linspace(0, 100, 4, endpoint = False),
            np.linspace(100, np.nanmax(change_j), 2)
            ])
    else:
        levels = np.concatenate([
            np.linspace(-50, 0, 2, endpoint = False),
            np.linspace(0, 100, 4, endpoint = False)])
    
    #normalize values 
    norm = plt.Normalize(vmin=-75, vmax = 75)
    
    cax = ax_j.contourf(change_j, transform = lambert_proj,extent = extent,
                    cmap='RdYlBu', origin='upper',
                    levels = levels,
                    norm=norm,
                    )
    
    ax_j.set_extent([min_lon,max_lon, min_lat, max_lat])
    
    # Add a colorbar with label
    plt.colorbar(cax, ax=ax_j, orientation='vertical', label='% Change')
    
    # Calculate statistics for display once
    mean_change= np.nanmean(change_j)
    max_change = np.nanmax(change_j)
    min_change= np.nanmin(change_j)
    
    # Add statistical information as text in a white box
    stats_text = f'Mean: {mean_change:.2f}x\nMax: {max_change:.2f}x\nMin: {min_change:.2f}x'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax_j.text(0.02, 0.98, stats_text, transform=ax_j.transAxes, verticalalignment='top', bbox=props)
    
    # Add gridlines
    # ax_j.gridlines(draw_labels=True, x_inline=False,y_inline=False)

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path+'DDF_change_'+f'{scenarios[ix]}_{periods[ix2]}_{RI}yr_{D}da_indiv_mods'
    save_path_name = save_path_name + '_MNclip' if clip_MN else save_path_name
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
