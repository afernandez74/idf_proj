
import numpy as np
import rasterio
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.features import geometry_mask
import pyproj 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import seaborn as sns
from scipy.stats import mannwhitneyu
from funcs import reproject_raster, init_lambert_proj, load_minnesota_reproj, create_dataarray
plt.rcdefaults()
plt.style.use('seaborn-v0_8-poster')

#%% Choose rasters to plot
base_path = '../Data/DDF_tif_bc_25/'

# =============================================================================
# Pick data from options
# =============================================================================
clip_MN = True #if true, raster data clipped to MN 

# return interval [years]
RI = 100

#duration [days]
D = 10

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
# Pick future period
# =============================================================================
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

period = periods[ix2]

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
strings = [scenario, period, f'_{bc_source}' ,f'_{RI}yr',f'{D:02}da']

files = [file for file in filenames if all(s in file for s in strings)]

files_hist = [file.replace(scenario,'historical') for file in files]
files_hist = [file.replace(period,'1995-2014') for file in files_hist]

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

#%% get rid of negative values and outliers
for source_i in data_hist.source.values:
   
    subset = data_hist.sel(source=source_i)
    
    # Compute median and MAD (Median Absolute Deviation)
    median = subset.median(skipna=True)
    mad = (np.abs(subset - median)).median(skipna=True)
    
    # Define robust threshold
    threshold_upper = median + 10 * mad
    threshold_lower = median - 10 * mad
    
    # Apply filtering
    masky = (subset < threshold_lower) | (subset > threshold_upper)
    data_hist.loc[dict(source=source_i)] = subset.where(~masky, np.nan)

del masky,threshold_upper,threshold_lower,mad,median,subset
for source_i in data_futu.source.values:
   
    subset = data_futu.sel(source=source_i)
    
    # Compute median and MAD (Median Absolute Deviation)
    median = subset.median(skipna=True)
    mad = (np.abs(subset - median)).median(skipna=True)
    
    # Define robust threshold
    threshold_upper = median + 10 * mad
    threshold_lower = median - 10 * mad
    
    # Apply filtering
    masky = (subset < threshold_lower) | (subset > threshold_upper)
    data_futu.loc[dict(source=source_i)] = subset.where(~masky, np.nan)

del masky,threshold_upper,threshold_lower,mad,median,subset
#%% Calculate the change between future and historical data 
change = data_futu.copy()
abs_change = data_futu.copy()
abs_proj = data_futu.copy()
for idx in range(len(data_futu['source'])):
    # Get the model corresponding to the current index
    model_str = str(data_futu['model'].values[idx])
    
    # Select the data for the current model from both DataArrays using integer indexing on 'source'
    da1_model = data_hist.isel(source=idx)
    da2_model = data_futu.isel(source=idx)
    
    # Perform the percent change calculation (avoid division by zero by masking NaNs)
    percent_change = ((da2_model / da1_model)-1)*100
    subtraction = da2_model - da1_model
    tot_proj = da2_model
    # Assign the calculated percent change to the corresponding location in the 'change' DataArray
    change.loc[dict(source=data_futu.source[idx])] = percent_change
    # Assign the calculated difference to the corresponding location in the 'change' DataArray
    abs_change.loc[dict(source=data_futu.source[idx])] = subtraction
    abs_proj.loc[dict(source=data_futu.source[idx])] = tot_proj

# calculate multi-model mean of relative change percent
change_mean = change.mean(dim = 'source',skipna=True)
abs_change_mean = abs_change.mean(dim = 'source', skipna = True)
abs_change_mean = abs_change_mean * 25.4 # change to mm

#%% Calculate projected means and historical means and return values as MMI and MAC
abs_proj_mean = abs_proj.mean(dim = 'source',skipna=True)
abs_proj_mean = abs_proj_mean * 25.4 / D # change to mm/da

abs_hist_mean = data_hist.mean(dim='source',skipna=True)
abs_hist_mean = abs_hist_mean * 25.4 / D#change to mm/da

# Calculate statistics for display once
mean_change= np.nanmean(abs_hist_mean)
std_change = np.std(abs_hist_mean)
max_change = np.nanmax(abs_hist_mean)
min_change= np.nanmin(abs_hist_mean)

# Flatten and clean the data
hist_vals = abs_hist_mean.values.flatten()
proj_vals = abs_proj_mean.values.flatten()

# Mask invalid or non-positive values
mask = ~np.isnan(hist_vals) & ~np.isnan(proj_vals) & (hist_vals > 0)
hist_vals = hist_vals[mask]
proj_vals = proj_vals[mask]

# Mann–Whitney U test
u_stat, p_value = mannwhitneyu(proj_vals, hist_vals, alternative='greater')
# print(f"M-U p-val: {p_value}")

# Relative (multiplicative) change
log_ratio = np.log(proj_vals / hist_vals)
log_ratio = log_ratio[np.isfinite(log_ratio)]
mult_increase = np.exp(log_ratio)

median_mult = np.median(mult_increase)
percent_increase = (median_mult - 1) * 100
lower_q_mult = np.percentile(mult_increase, 25)
upper_q_mult = np.percentile(mult_increase, 75)
iqr_percent = ((lower_q_mult - 1) * 100, (upper_q_mult - 1) * 100)

# Absolute change
abs_change = proj_vals - hist_vals
median_abs = np.median(abs_change)
lower_q_abs = np.percentile(abs_change, 25)
upper_q_abs = np.percentile(abs_change, 75)

# Console output
print(f"\n\nChange from historical to {period} in {RI}-year, {D}-day events for {scenario}:")
print(f"\nMMI (%):  \n{percent_increase:.1f} ({iqr_percent[0]:.1f} - {iqr_percent[1]:.1f})\n")
# print(f"IQR (percent): {iqr_percent[0]:.1f}% – {iqr_percent[1]:.1f}%\n")

print(f"\nMAC (mm/day): \n{median_abs:.1f} ({lower_q_abs:.1f} - {upper_q_abs:.1f})")
# print(f"IQR (absolute): {lower_q_abs:.1f} – {upper_q_abs:.1f} mm")

