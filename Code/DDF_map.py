import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.mask import mask
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
#%% Choose rasters to plot
base_path = '../Data/DDF_tif/'

# =============================================================================
# Pick data from options
# =============================================================================

ensemble = False # ensemble vs separate
smooth = True # smooth vs unsmooth
adjusted = False # adjusted vs unadjusted

clip_MN = True #if true, raster data clipped to MN 

# return interval [years]
RI = 100

#duration [days]
D = 1

# =============================================================================
# Set paths
# =============================================================================
path = base_path +'ensemble/' if ensemble else base_path + 'separate/'
path = path +'smooth/' if smooth else path + 'unsmooth/'
path = path +'adjusted/' if adjusted else path + 'unadjusted/'

# path to historical data 
path_hist = path + 'historical/1995-2014/'

# =============================================f================================
# Pick future scenario 
# =============================================================================

scenarios = [file for file in os.listdir(path) if not file.startswith('.')]

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

path = path + scenarios[ix] + '/'

# =============================================================================
# Pick future period
# =============================================================================

periods = [file for file in os.listdir(path) if not file.startswith('.')]
print('Periods:')

for i, item in enumerate (periods):
    print(f'{i}:    {item}')

while True:
    try:
        ix = int(input("Enter the number of your choice: "))
        if 0 <= ix <= len(periods):
            break
        else:
            print(f"Please enter a number between 0 and {len(periods)-1}.")
    except ValueError:
        print("Please enter a valid number.")

path = path + periods[ix] + '/'

files = os.listdir(path)

#%% load minnesota outline for map 
url = "https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip"
usa = gpd.read_file(url)
minnesota = usa[usa['NAME'] == 'Minnesota']
# lambert_projection = ccrs.LambertConformal()
# minnesota = minnesota.to_crs(lambert_projection.proj4_init)
# minnesota_bounds = minnesota.total_bounds

# Lambert Conformal Conic projection 
lambert_proj = ccrs.LambertConformal(
    central_longitude = -94.37,
    central_latitude = 46.03,
    standard_parallels = (33,45)
    )

#%% load data with rasterio as a dictionary

tif_files_futu = [file for file in os.listdir(path) if not file.startswith('.')]
pref_futu = os.path.commonprefix(tif_files_futu)

tif_files_hist = [file for file in os.listdir(path_hist) if not file.startswith('.')]
pref_hist = os.path.commonprefix(tif_files_hist)

# =============================================================================
#  read projection raster 
# =============================================================================
with rasterio.open(path + pref_futu + f'{RI}yr{D:02}da.tif') as src:
    # print(src.crs)  # Coordinate Reference System
    if clip_MN:    
        out_image, out_transform = mask(src, minnesota.geometry, crop=True)
        data_futu = out_image[0]  # Read the first band
    else:
        data_futu = src.read(1)
    data_futu[data_futu < 0] = 0
    transform_futu = src.transform
    crs_futu = src.crs
    
# =============================================================================
#  read historical period raster
# =============================================================================
with rasterio.open(path_hist + pref_hist + f'{RI}yr{D:02}da.tif') as src:
    # print(src.crs)  # Coordinate Reference System
    if clip_MN:
        out_image, out_transform = mask(src, minnesota.geometry, crop=True)
        data_hist = out_image[0]  # Read the first band
    else:
        data_hist = src.read(1)
        
    data_hist[data_hist<0] = 0    
    transform_hist = src.transform
    crs_futu = src.crs

#%% plot ratio of futu/hist for chosen data / duration / return interval combo

plt.rcParams.update({'font.size': 14})

# Calculate the ratio between future and historical data
ratio = data_futu / data_hist

# Set up the figure and axis with Lambert Conformal projection
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.LambertConformal()})

# Set titles and labels
ax.set_title(f'Event depth change from historical to {periods[ix]}\n100-year event, 1-day duration')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Draw political boundaries and other features, matching the Lambert Conformal projection
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
ax.add_feature(cfeature.RIVERS, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)
ax.add_feature(cfeature.LAKES, linestyle='--', color='lightblue', linewidth=0.4, zorder=1)

# Plot the Minnesota boundary on the Lambert Conformal map
minnesota = minnesota.to_crs(lambert_proj.proj4_init)
minnesota.boundary.plot(ax=ax, color='black', linewidth=1.5)

# Display the raster data using imshow, with extent matching Minnesota's bounds
cax = ax.imshow(ratio, cmap='YlGnBu', origin='upper', vmin=1, vmax=2)

# Set extent (adjust these values based on your data)
ax.set_extent([-97.94, -88.69, 
               42.54, 49.97], crs=ccrs.PlateCarree())


# Add a colorbar with label
plt.colorbar(cax, ax=ax, orientation='vertical', label='Ratio')

# Calculate statistics for display once
mean_ratio = np.nanmean(ratio)
max_ratio = np.nanmax(ratio)
min_ratio = np.nanmin(ratio)

# Add statistical information as text in a white box
stats_text = f'Mean: {mean_ratio:.2f}x\nMax: {max_ratio:.2f}x\nMin: {min_ratio:.2f}x'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=props)

# Show the plot
plt.show()