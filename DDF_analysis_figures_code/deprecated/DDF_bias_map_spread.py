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

#%% specify return interval and duration for analysis

clip_MN = True #if true, raster data clipped to MN 

# return interval (years)
RI = 2

# Duration (days)
D = 1

#save path results
save_path = '../Figures/Bias/'
save_path = save_path + 'clip_MN/' if clip_MN else save_path + 'whole/'

# load minnesota outline and projection for maps
lambert_proj = init_lambert_proj()
minnesota = load_minnesota_reproj(lambert_proj)


#%% paths for CMIP6 projections
base_path = '../Data/DDF_individual_model_tif/'

# =============================================================================
# Set paths
# =============================================================================

# path to historical data 
path_hist = base_path + 'hist/'

path_hist_models = [path_hist + file for file in os.listdir(path_hist) if not file.startswith('.')]

# files in all paths historical
paths_all_hist = []
for pathy in path_hist_models:
    model = pathy[pathy.rfind('/')+1:]
    files = [file for file in os.listdir(pathy) if not file.startswith('.')]
    paths_temp = [path_hist + model + '/' + file for file in files]
    paths_all_hist.append(paths_temp)

paths_all_hist = [item for row in paths_all_hist for item in row]

# filter for desired duration and RI
paths_all_hist = [file for file in paths_all_hist if f'{D:02}da' in file and f'_{RI}yr' in file]

#%% open raster files and reproject (CMIP6)

data_hist=[]
for file_i in paths_all_hist:
    name = 'historical_' + file_i[file_i.rfind('/') + 1 : file_i.find('.tif')]
    model = name.split('_')[-2]
    data_hist_temp, metadata_temp = reproject_raster(file_i,lambert_proj.proj4_init)
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
    for name, da in data_hist.groupby('source'):
        data_hist.loc[dict(source=name)] = np.where(mask, da.squeeze(), np.nan)

del data_hist_temp, data_hist_da, metadata_temp


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

#%% quick plot of NA14 data
# fig, ax = plt.subplots(subplot_kw={'projection': lambert_proj})
# # Set titles and labels
# ax.set_title(f'NA14 Depth for {RI}yr {D}da')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')

# # latitudes and longitudes for map extent
# min_lon, min_lat = -97.94, 42.54
# max_lon, max_lat = -88.69, 49.97


# # Draw political boundaries and other features, matching the Lambert Conformal projection
# ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
# ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
# ax.add_feature(cfeature.RIVERS, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
# ax.add_feature(cfeature.LAKES, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)

# # Plot the Minnesota boundary on the Lambert Conformal map
# minnesota.boundary.plot(ax=ax, color='black', linewidth=1.5)

# extent = [
#     metadata['transform'][2],
#     metadata['transform'][2] + metadata['transform'][0] * metadata['width'],
#     metadata['transform'][5] + metadata['transform'][4] * metadata['height'],
#     metadata['transform'][5]
# ]

# cax = ax.contourf(NA14, transform = lambert_proj,extent = extent,
#                 cmap='autumn_r', 
#                 origin='upper',
#                 )

# ax.set_extent([min_lon,max_lon, min_lat, max_lat])

# # Add a colorbar with label
# plt.colorbar(cax, ax=ax, orientation='vertical', label='Depth [in]')

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
#%% plot map of bias per model

# Define shared normalization and levels for the colorbar
vmin, vmax = 0, 2  # Set minimum and maximum values for normalization
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter = 1.0, vmax=vmax)
levels = np.linspace(vmin, vmax, 10)  # Define 11 levels between vmin and vmax

fig, ax = plt.subplots(nrows=2, ncols=3,subplot_kw={'projection': lambert_proj})
ax = ax.flatten()

# Add a title for the entire figure
fig.suptitle(f'Model Bias for {RI}yr {D:02}day Event',fontsize = 20)

for idx, source in enumerate(bias['source']):
    
    ax_i = ax[idx]
    bias_i = bias.isel(source = idx)

    # Set titles and labels
    ax_i.set_title(source.model.values)
    ax_i.set_xlabel('Longitude')
    ax_i.set_ylabel('Latitude')
    
    # Draw political boundaries and other features, matching the Lambert Conformal projection
    ax_i.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax_i.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    ax_i.add_feature(cfeature.RIVERS, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
    ax_i.add_feature(cfeature.LAKES, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
    
    # Plot the Minnesota boundary on the Lambert Conformal map
    minnesota.boundary.plot(ax=ax_i, color='black', linewidth=1.5)
    
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
    
    
    cax = ax_i.contourf(bias_i, 
                    transform = lambert_proj,
                    extent = extent,
                    cmap='bwr_r', 
                    origin='upper',
                    levels = levels,
                    norm=norm,
                    )
    
    ax_i.set_extent([min_lon,max_lon, min_lat, max_lat])
    
# Add a colorbar with label
cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', fraction = 0.05, pad = 0.1, 
             label='Bias')
cbar.set_ticks(np.linspace(vmin, vmax, 5)) 
    
    # Add gridlines
    # ax_i.gridlines(draw_labels=True, x_inline=False,y_inline=False)

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'Bias_map_'+f'{RI}yr_{D}da'
    save_path_name = save_path_name + '_MNclip' if clip_MN else save_path_name
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
#%% plot map of mean bias across all models

bias_mean = bias.mean(dim = 'source')


# Define shared normalization and levels for the colorbar
vmin, vmax = 0, 2  # Set minimum and maximum values for normalization
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter = 1.0, vmax=vmax)
levels = np.linspace(vmin, vmax, 10)  # Define 11 levels between vmin and vmax

fig, ax = plt.subplots(subplot_kw={'projection': lambert_proj})


# Set titles and labels
ax.set_title(f'Mean bias across all models\nfor {RI}yr {D:02}day Event')
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


cax = ax.contourf(bias_mean, 
                transform = lambert_proj,
                extent = extent,
                cmap='bwr_r', 
                origin='upper',
                levels = levels,
                norm=norm,
                )

ax.set_extent([min_lon,max_lon, min_lat, max_lat])
    
# Add a colorbar with label
cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', fraction = 0.05, pad = 0.1, 
             label='Bias')
cbar.set_ticks(np.linspace(vmin, vmax, 5)) 
    
    # Add gridlines
    # ax_i.gridlines(draw_labels=True, x_inline=False,y_inline=False)

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'Mean_bias_map_'+f'{RI}yr_{D}da'
    save_path_name = save_path_name + '_MNclip' if clip_MN else save_path_name
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
#%% Plot spread of bias 
# Create an empty list to store the data for plotting
bias_spreads = []
for model,bias_i in bias.groupby('model'):
    vals = bias_i.values.flatten()
    dat = pd.DataFrame({
        model: vals[~pd.isna(vals)]# Use the period as the column name
    })
    bias_spreads.append(dat)
bias_spreads = pd.concat(bias_spreads, axis=1)

# %% Plot the violin plot
plt.figure()
# Create the violin plot using seaborn
sns.boxplot(data=bias_spreads, 
               # inner='quart', 
               palette="bright", 
               # scale = 'width',
               linewidth = 2.0,
               showfliers = False
               )

plt.title(f'Bias spread per model {RI}yr {D}da')
plt.xticks(rotation=45)
plt.ylim(-0,2
         )
plt.grid(linewidth = 0.5,linestyle = '--')
plt.axhline(1.0, color = 'black', linewidth = 1.0, linestyle = '--')


# Add plot labels and title
plt.xlabel('Period')
plt.ylabel('Bias (mod/NA14)')

# Show the plot
plt.tight_layout()

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'Bias_spread_model_'+f'{RI}yr_{D}da' 
    save_path_name = save_path_name + '_MNclip' if clip_MN else save_path_name
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()