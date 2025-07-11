
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
RI = 25

#duration [days]
D = 1

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
# # get rid of BCC model if ssp370
# if scenarios[ix] == 'ssp370':
#     change = change.where(change['model'] != 'BCC-CSM2-MR', drop = True)
#     abs_change = abs_change.where(abs_change['model'] != 'BCC-CSM2-MR', drop = True)

# calculate multi-model mean of relative change percent
change_mean = change.mean(dim = 'source',skipna=True)
abs_change_mean = abs_change.mean(dim = 'source', skipna = True)
abs_change_mean = abs_change_mean * 25.4 # change to mm
#%% plot change of futu/hist for chosen data / duration / return interval combo
# (mean of all models)

# Set up the figure and axis with Lambert Conformal projection
fig, ax = plt.subplots(subplot_kw={'projection': lambert_proj})

# Set titles and labels
ax.set_title(f'Event depth change from historical to {periods[ix2]}\n{RI}-year event, {D}-day duration, {scenarios[ix]} Scenario\n{bc_source}')
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
vmin, vmax = -40, 160

# levels for contours:
if np.nanmax(abs_change_mean) > vmax:
    levels = np.concatenate([
        np.linspace(-40, 0, 2, endpoint = False),
        np.linspace(0, vmax, 8, endpoint = False),
        np.linspace(vmax, np.nanmax(abs_change_mean), 2)
        ])
else:
    levels = np.concatenate([
        np.linspace(-40, 0, 2, endpoint = False),
        np.linspace(0, vmax, 8, endpoint = False)])

# vmin = -40 if vmin > 0 else vmin

#normalize values 
norm = mcolors.TwoSlopeNorm(vmin = vmin, vcenter=-10, vmax = vmax)

cmap = plt.cm.RdYlBu

cax = ax.contourf(abs_change_mean, transform = lambert_proj,extent = extent,
                cmap=cmap, 
                origin='upper',
                levels = levels,
                norm=norm,
                )

# Overlay values greater than vmax in a separate colo
over_vmax = np.ma.masked_less_equal(abs_change_mean, vmax)  # Mask everything ≤ vmax
if over_vmax.any():
    cax2 = ax.contourf(over_vmax, transform=lambert_proj, extent=extent,
                   colors=['midnightblue'], origin='upper', 
                   # levels=[vmax, np.nanmax(abs_change_mean)]
                   )

ax.set_extent([min_lon,max_lon, min_lat, max_lat])

# Add a colorbar with label
plt.colorbar(cax, ax=ax, orientation='vertical', label='Precip depth change mm/24hr')

# Calculate statistics for display once
mean_change= np.nanmean(abs_change_mean)
std_change = np.std(abs_change_mean)
max_change = np.nanmax(abs_change_mean)
min_change= np.nanmin(abs_change_mean)

# Add statistical information as text in a white box
stats_text = f'Mean: {mean_change:.2f}mm\nStd: {std_change:.2f}mm'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=props)

# Add gridlines
ax.gridlines(draw_labels=True, x_inline=False,y_inline=False)

#save path results
save_path = '../Figures/abs_proj/mean_model_abs_change_map/'

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'map_RC_bc_'+bc_source+f'_{scenarios[ix]}_{periods[ix2]}_{RI}yr_{D}da'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

#%% histogram
# Flatten data and remove NaNs for histogram
valid_data = abs_change_mean.values.flatten()
valid_data = valid_data[~np.isnan(valid_data)]

# Create histogram figure
fig_hist, ax_hist = plt.subplots()

# Plot histogram
n, bins, patches = ax_hist.hist(valid_data, 
                                bins=levels, 
                                edgecolor='black', 
                                align='mid')

# Apply colors to bars
for patch, level in zip(patches, levels[:]):
    patch.set_facecolor(cmap(norm(level)))

# Set labels
ax_hist.set_xlabel('mm/day difference')
ax_hist.set_ylabel('Frequency')
ax_hist.set_title('Histogram of Event Depth Change')

save_option = input("Save histogram? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'hist_RC_bc_'+bc_source+f'_{scenarios[ix]}_{periods[ix2]}_{RI}yr_{D}da'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
#%%
# #%% plot change of futu/hist for chosen data / duration / return interval combo 
# #(single model grid)

# # Set up the figure and axis with Lambert Conformal projection
# fig, ax = plt.subplots(nrows=2, ncols=3,subplot_kw={'projection': lambert_proj})
# ax = ax.flatten()

# # Add a title for the entire figure
# fig.suptitle(f'Projceted change from historical to {periods[ix2]}\n{RI}-year event, {D}-day duration, {scenarios[ix]} Scenario',fontsize = 20)

# abs_change = abs_change.sortby('model')

# for idx, source in enumerate(abs_change['source']):
#     ax_j = ax[idx]
    
#     change_j = abs_change.isel(source = idx)
    
#     # Set titles and labels
#     ax_j.set_title(source.model.values)

#     # Draw political boundaries and other features, matching the Lambert Conformal projection
#     ax_j.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
#     ax_j.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
#     ax_j.add_feature(cfeature.RIVERS, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
#     ax_j.add_feature(cfeature.LAKES, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
    
#     # Plot the Minnesota boundary on the Lambert Conformal map
#     minnesota.boundary.plot(ax=ax_j, color='black', linewidth=1.5)
    
#     extent = [
#         metadata['transform'][2],
#         metadata['transform'][2] + metadata['transform'][0] * metadata['width'],
#         metadata['transform'][5] + metadata['transform'][4] * metadata['height'],
#         metadata['transform'][5]
#     ]
    
#     #transformer for custom extent values
#     transformer = pyproj.Transformer.from_crs(metadata['crs'], lambert_proj.proj4_init, always_xy=True)
    
#     # latitudes and longitudes for map extent
#     min_lon, min_lat = -97.94, 42.54
#     max_lon, max_lat = -88.69, 49.97
    
#     # =============================================================================
#     # Contour map
#     # =============================================================================
    
#     # levels for contours:
#     if np.nanmax(change_j) >100:
#         levels = np.concatenate([
#             np.linspace(-60, 0, 3, endpoint = False),
#             np.linspace(0, 100, 5, endpoint = False),
#             np.linspace(100, np.nanmax(change_j), 2)
#             ])
#     else:
#         levels = np.concatenate([
#             np.linspace(-60, 0, 3, endpoint = False),
#             np.linspace(0, 100, 5, endpoint = False)])
            
#     #normalize values 
#     norm = plt.Normalize(vmin=-75, vmax = 75)
    
#     cax = ax_j.contourf(change_j, transform = lambert_proj,extent = extent,
#                     cmap='RdYlBu', origin='upper',
#                     levels = levels,
#                     norm=norm,
#                     )
    
#     ax_j.set_extent([min_lon,max_lon, min_lat, max_lat])
    
#     # Add a colorbar with label
#     plt.colorbar(cax, ax=ax_j, orientation='vertical', label='% Change')
    
#     # Calculate statistics for display once
#     mean_change= np.nanmean(change_j)
#     max_change = np.nanmax(change_j)
#     min_change= np.nanmin(change_j)
    
#     # Add statistical information as text in a white box
#     stats_text = f'Mean: {mean_change:.2f}%\nStd: {std_change:.2f}%'
#     props = dict(boxstyle='round', facecolor='white', alpha=0.8)
#     ax_j.text(0.02, 0.98, stats_text, transform=ax_j.transAxes, verticalalignment='top', bbox=props)
    
#     # Add gridlines
#     # ax_j.gridlines(draw_labels=True, x_inline=False,y_inline=False)

# save_option = input("Save figure? (y/n): ").lower()
# save_path = '../Figures/RC_proj/single_model_RC_map/'
# save_path = save_path + 'clip_MN/' if clip_MN else save_path + 'whole/'

# if save_option == 'y':
#     save_path_name = save_path + 'map_RC_bc_'+bc_source+f'_{scenarios[ix]}_{periods[ix2]}_{RI}yr_{D}da'
#     # Save as SVG
#     plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
#     # Save as PNG
#     plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
# else:
#     plt.show()

#%% plot absolute value for projection (not subtraction or relative change)
abs_proj_mean = abs_proj.mean(dim = 'source',skipna=True)
abs_proj_mean = abs_proj_mean * 25.4 # change to mm

abs_hist_mean = data_hist.mean(dim='source',skipna=True)
abs_hist_mean = abs_hist_mean * 25.4 #change to mm
# Set up the figure and axis with Lambert Conformal projection

# latitudes and longitudes for map extent
min_lon, min_lat = -97.94, 42.54
max_lon, max_lat = -88.69, 49.97

# vmin, vmax = abs_hist_mean.mean().values - abs_hist_mean.std().values, abs_proj_mean.mean().values + abs_proj_mean.mean().values
# vmin, vmax = 60, 400
vmin, vmax = 60, abs_proj_mean.mean().values + abs_proj_mean.mean().values
# levels for contours:
# levels = np.concatenate([
#     np.linspace(vmin, vmax, 5, endpoint = False)])


#normalize values 
norm = mcolors.Normalize(vmin = vmin, 
                            # vcenter=0, 
                            vmax = vmax
                            )

cmap = plt.cm.magma_r

fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2,subplot_kw={'projection': lambert_proj},
                              figsize = (14,6))
plt.subplots_adjust(wspace=0.5)
# =============================================================================
# 1st Contour map
# =============================================================================


# Set titles and labels
ax1.set_title(f'Event depth change from historical to {periods[ix2]}\n{RI}-year event, {D}-day duration, {scenarios[ix]} Scenario\n{bc_source}')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')

# Draw political boundaries and other features, matching the Lambert Conformal projection
ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
ax1.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
ax1.add_feature(cfeature.RIVERS, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
ax1.add_feature(cfeature.LAKES, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)

# Plot the Minnesota boundary on the Lambert Conformal map
minnesota.boundary.plot(ax=ax1, color='black', linewidth=1.5)

extent = [
    metadata['transform'][2],
    metadata['transform'][2] + metadata['transform'][0] * metadata['width'],
    metadata['transform'][5] + metadata['transform'][4] * metadata['height'],
    metadata['transform'][5]
]

#transformer for custom extent values
transformer = pyproj.Transformer.from_crs(metadata['crs'], lambert_proj.proj4_init, always_xy=True)

cax = ax1.imshow(abs_hist_mean, transform = lambert_proj,extent = extent,
                cmap=cmap, 
                origin='upper',
                # levels = levels,
                norm=norm,
                )

# # Overlay values greater than vmax in a separate colo
# over_vmax = np.ma.masked_less_equal(abs_change_mean, vmax)  # Mask everything ≤ vmax
# if over_vmax.any():
#     cax2 = ax.contourf(over_vmax, transform=lambert_proj, extent=extent,
#                    colors=['midnightblue'], origin='upper', 
#                    # levels=[vmax, np.nanmax(abs_change_mean)]
#                    )

ax1.set_extent([min_lon,max_lon, min_lat, max_lat])

# Add a colorbar with label
plt.colorbar(cax, ax=ax1, orientation='vertical', label='Precip depth change mm/24hr')

# Calculate statistics for display once
mean_change= np.nanmean(abs_hist_mean)
std_change = np.std(abs_hist_mean)
max_change = np.nanmax(abs_hist_mean)
min_change= np.nanmin(abs_hist_mean)

# Add statistical information as text in a white box
stats_text = f'Mean: {mean_change:.2f}mm\nStd: {std_change:.2f}mm'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top', bbox=props)

# =============================================================================
# 2nd Contour map
# =============================================================================


# Set titles and labels
ax2.set_title(f'Event depth change from historical to {periods[ix2]}\n{RI}-year event, {D}-day duration, {scenarios[ix]} Scenario\n{bc_source}')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
# Draw political boundaries and other features, matching the Lambert Conformal projection
ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
ax2.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
ax2.add_feature(cfeature.RIVERS, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
ax2.add_feature(cfeature.LAKES, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)

# Plot the Minnesota boundary on the Lambert Conformal map
minnesota.boundary.plot(ax=ax2, color='black', linewidth=1.5)

extent = [
    metadata['transform'][2],
    metadata['transform'][2] + metadata['transform'][0] * metadata['width'],
    metadata['transform'][5] + metadata['transform'][4] * metadata['height'],
    metadata['transform'][5]
]

#transformer for custom extent values
transformer = pyproj.Transformer.from_crs(metadata['crs'], lambert_proj.proj4_init, always_xy=True)



cax = ax2.imshow(abs_proj_mean, transform = lambert_proj,extent = extent,
                cmap=cmap, 
                origin='upper',
                # levels = levels,
                norm=norm,
                )

# # Overlay values greater than vmax in a separate colo
# over_vmax = np.ma.masked_less_equal(abs_change_mean, vmax)  # Mask everything ≤ vmax
# if over_vmax.any():
#     cax2 = ax.contourf(over_vmax, transform=lambert_proj, extent=extent,
#                    colors=['midnightblue'], origin='upper', 
#                    # levels=[vmax, np.nanmax(abs_change_mean)]
#                    )

ax2.set_extent([min_lon,max_lon, min_lat, max_lat])

# Add a colorbar with label
plt.colorbar(cax, ax=ax2, orientation='vertical', label='Precip depth change mm/24hr')

# Calculate statistics for display once
mean_change= np.nanmean(abs_proj_mean)
std_change = np.std(abs_proj_mean)
max_change = np.nanmax(abs_proj_mean)
min_change= np.nanmin(abs_proj_mean)

# Add statistical information as text in a white box
stats_text = f'Mean: {mean_change:.2f}mm\nStd: {std_change:.2f}mm'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top', bbox=props)


#save path results
save_path = '../Figures/abs_proj/mean_model_abs_value_map/'

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'map_RC_bc_'+bc_source+f'_{scenarios[ix]}_{periods[ix2]}_{RI}yr_{D}da'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

#%%

# Flatten and clean the data
hist_vals = abs_hist_mean.values.flatten()
proj_vals = abs_proj_mean.values.flatten()

# Mask invalid or non-positive values
mask = ~np.isnan(hist_vals) & ~np.isnan(proj_vals) & (hist_vals > 0)
hist_vals = hist_vals[mask]
proj_vals = proj_vals[mask]

# Mann–Whitney U test
u_stat, p_value = mannwhitneyu(proj_vals, hist_vals, alternative='greater')

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
print(f"Median multiplicative increase:  {percent_increase:.1f}%")
print(f"IQR (percent): {iqr_percent[0]:.1f}% – {iqr_percent[1]:.1f}%\n")

print(f"Median absolute change: {median_abs:.1f} mm")
print(f"IQR (absolute): {lower_q_abs:.1f} – {upper_q_abs:.1f} mm")

# Plot KDEs
plt.figure(figsize=(10, 6))
sns.kdeplot(hist_vals, label='Historical', color='darkorange', fill=False, linewidth=2.5)
sns.kdeplot(proj_vals, label='Projected', color='mediumblue', fill=False, linewidth=2.5)

plt.title('Empirical Distributions of IDF Values')
plt.xlabel('Precipitation Depth (mm)')
plt.ylabel('Density')
plt.legend()

# Annotate results on the plot
textstr = '\n'.join((
    f'Mann–Whitney U p-value: {p_value:.2e}',
    f'Median increase: {percent_increase:.1f}%',
    f'IQR (percent): {iqr_percent[0]:.1f}–{iqr_percent[1]:.1f}%',
    f'Median abs. change: {median_abs:.1f} mm',
    f'IQR (abs): {lower_q_abs:.1f}–{upper_q_abs:.1f} mm'))

props = dict(boxstyle='round', facecolor='white', alpha=0.85)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
               verticalalignment='top', bbox=props)

plt.grid(linestyle='--', linewidth=1)
plt.tight_layout()
save_path = '../Figures/abs_proj/mean_model_abs_distribs/'

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'distribs_bc_'+bc_source+f'_{scenarios[ix]}_{periods[ix2]}_{RI}yr_{D}da'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
