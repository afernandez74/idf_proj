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

#%% Choose rasters to plot
base_path = '../Data/DDF_tif_bc_25/'
save_path = '../Figures/RC_proj/model_period_boxplots/clip_MN/'
# =============================================================================
# Pick data from options
# =============================================================================
clip_MN = True #if true, raster data clipped to MN 

# return interval [years]
RI = 50

#duration [days]
D = 7

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
strings = [scenario, f'_{bc_source}' ,f'_{RI}yr',f'{D:02}da']
strings_hist = ['historical',f'_{bc_source}' ,f'_{RI}yr',f'{D:02}da']

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

#%% calculate change percentages
change = data_futu.copy()

for period, futu_data in data_futu.groupby('period'):
    # Get the corresponding historical data based on the model coordinate
    for model in futu_data['model'].values:
        futu_source = futu_data['source'].sel(source=futu_data['model'] == model)
        hist_source = data_hist['source'].sel(source=data_hist['model'] == model)
        
        # Select the future and historical data for the same model
        da2 = futu_data.sel(source=futu_data['model'] == model).values
        da1 = data_hist.sel(source=data_hist['model'] == model).values

        # Calculate percent change
        percent_change = ((da2 / da1) - 1) * 100
        percent_change = np.squeeze(percent_change)

        change.loc[{'source': futu_source.values[0]}] = percent_change
        
# %% Calculate mean percent change per period
# Create an empty list to store the data for plotting
model_mean = change.groupby('period').mean()

#%% generate a grid of boxplots for each model and period
df = change.to_dataframe(name='percent_change').reset_index()
df = df.dropna(subset=["percent_change"])

df_mean = model_mean.to_dataframe(name='percent_change').reset_index()
df_mean = df_mean.dropna(subset=["percent_change"])
df_mean["model"] = "Model Mean"

df_combined = pd.concat([df, df_mean], ignore_index=True)

model_order = sorted(df["model"].unique())

if 'Model Mean' in df_combined['model'].unique() and 'Model Mean' not in model_order:
    model_order.append('Model Mean')

# Convert the 'model' column to a categorical type with the specified order
df_combined['model'] = pd.Categorical(df_combined['model'], categories=model_order, ordered=True)

# Sort the DataFrame by the ordered 'model' column and then by 'period'
df_combined = df_combined.sort_values(['model', 'period'])

# plot boxes

plt.figure(figsize=(12, 8))  # Set a larger figure size for better readability
# Create grouped boxplot
sns.boxplot(
    data=df_combined, 
    x="model", 
    y="percent_change", 
    hue="period",  # Group by model within each period
    showfliers=False, 
    palette="bright"
)

plt.title(f"Comparison of Bias Across Models and Periods for {D}day, {RI}yr events\nScenario {scenarios[ix]}", fontsize=16)
plt.xlabel("Model")
plt.ylabel("% Change")
plt.ylim(-100, 200)
plt.xticks(rotation=45, ha='right')  # Rotate and align labels for readability
plt.legend(title="Period", bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside plot
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Show the plot
plt.tight_layout()

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path+f'{scenarios[ix]}_{RI}yr_{D}da'    
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

