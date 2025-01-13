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

D = 1
RI = 100
#%% Choose rasters to plot
base_path = '../Data/DDF_individual_model_tif/'

# =============================================================================
# Pick data from options
# =============================================================================

clip_MN = True #if true, raster data clipped to MN 

#save path results
save_path = '../Figures/DDF_change_distrib_single_model/'
save_path = save_path + 'clip_MN/' if clip_MN else save_path + 'whole/'

# =============================================================================
# Set path
# =============================================================================

# path to historical data 
path_hist = base_path + 'hist/'

# =============================================================================
# Get all path to files
# =============================================================================

#future files
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

durations = [file for file in os.listdir(path) if not file.startswith('.')]
path_durations = [path + duration for duration in durations]

path_models = []
for pathy in path_durations:
    files = [file for file in os.listdir(pathy) if not file.startswith('.')]
    path_temp = [pathy +'/' + file for file in files]
    path_models.append(path_temp)
path_models = [item for row in path_models for item in row]


path_all = []
for pathy in path_models:
    files = [file for file in os.listdir(pathy) if not file.startswith('.')]
    path_temp = [pathy +'/' + file for file in files]
    path_all.append(path_temp)

path_all = [item for row in path_all for item in row]

#historical files 
models_hist = [file for file in os.listdir(path_hist) if not file.startswith('.')]
path_models_hist = [path_hist + model for model in models_hist]

path_all_hist = []
for pathy in path_models_hist:
    files = [file for file in os.listdir(pathy) if not file.startswith('.')]
    path_temp = [pathy +'/' + file for file in files]
    path_all_hist.append(path_temp)

path_all_hist = [item for row in path_all_hist for item in row]

#%%#%% load minnesota outline and projection for maps
lambert_proj = init_lambert_proj()
minnesota = load_minnesota_reproj(lambert_proj)
#%% open raster files and reproject

files = path_all
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
for file in path_all_hist:
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

#%% filter data for duration and RI
data_futu_sel = data_futu.sel(source=data_futu.source['RI']==RI)
data_futu_sel = data_futu_sel.sel(source=data_futu_sel.source['D']==D)
data_futu_sel = data_futu_sel.sortby('period')
data_futu_sel = data_futu_sel.sortby('model')

data_hist_sel = data_hist.sel(source=data_hist.source['RI']==RI)
data_hist_sel = data_hist_sel.sel(source=data_hist_sel.source['D']==D)
data_hist_sel = data_hist_sel.sortby('model')
#%% calculate change percentages
change = data_futu_sel.copy()

for period, futu_data in data_futu_sel.groupby('period'):
    # Get the corresponding historical data based on the model coordinate
    for model in futu_data['model'].values:
        futu_source = futu_data['source'].sel(source=futu_data['model'] == model)
        hist_source = data_hist_sel['source'].sel(source=data_hist_sel['model'] == model)
        
        # Select the future and historical data for the same model
        da2 = futu_data.sel(source=futu_data['model'] == model).values
        da1 = data_hist_sel.sel(source=data_hist_sel['model'] == model).values

        # Calculate percent change
        percent_change = ((da2 / da1) - 1) * 100
        percent_change = np.squeeze(percent_change)

        change.loc[{'source': futu_source.values[0]}] = percent_change
# %% Calculate mean percent change per period
# Create an empty list to store the data for plotting
plot_data = []

# Loop over the 'period' dimension in the change DataArray
for period, period_data in change.groupby('period'):
    # Calculate the mean percent change across the 'model' dimension for the current period
    mean_percent_change = period_data.mean(dim='source',skipna = True)
    non_nan_values = mean_percent_change.values.flatten()
    non_nan_values = non_nan_values[~pd.isna(non_nan_values)]  # Remove NaN values

    # Create a DataFrame for the current period with non-NaN values
    period_df = pd.DataFrame({
        period: non_nan_values  # Use the period as the column name
    })
    
    # Append the period_df to the plot_data list
    plot_data.append(period_df)
df = pd.concat(plot_data, axis=1)

# %% Plot the violin plot
plt.figure()

# Create the violin plot using seaborn
sns.violinplot(data=df, inner="quart", palette="muted", scale = 'width')

plt.title(f'Mean percent change per period of simulation for {scenarios[ix]}')
plt.xticks(rotation=45)
# plt.legend(loc='upper left')
plt.ylim(-50,150)
plt.grid(zorder = 1)


# Add plot labels and title
plt.title(f'Distribution of Percent Change by Period\n {scenarios[ix]}_{RI}yr_{D}da')
plt.xlabel('Period')
plt.ylabel('Percent Change')

# Show the plot
plt.tight_layout()

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path+'DDF_distrib_'+f'{scenarios[ix]}_{RI}yr_{D}da'    
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()