import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
import pyproj 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from funcs import reproject_raster, init_lambert_proj, load_minnesota_reproj
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

#save path results
save_path = '../Figures/DDF_maps/'
save_path = save_path + 'clip_MN/' if clip_MN else save_path

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
        ix2 = int(input("Enter the number of your choice: "))
        if 0 <= ix2 <= len(periods):
            break
        else:
            print(f"Please enter a number between 0 and {len(periods)-1}.")
    except ValueError:
        print("Please enter a valid number.")

path = path + periods[ix2] + '/'

files = os.listdir(path)

# list of files in selexted paths - projections
tif_files_futu = [file for file in os.listdir(path) if not file.startswith('.')]
#prefix shared by all files
pref_futu = os.path.commonprefix(tif_files_futu)

# list of files in selexted paths - historical 
tif_files_hist = [file for file in os.listdir(path_hist) if not file.startswith('.')]
#prefix shared by all files
pref_hist = os.path.commonprefix(tif_files_hist)

#%% load minnesota outline and projection for maps
lambert_proj = init_lambert_proj()
minnesota = load_minnesota_reproj(lambert_proj)

#%% open raster files and reproject

# =============================================================================
#  read projection raster 
# =============================================================================

data_futu, metadata = reproject_raster(path+pref_futu+f'{RI}yr{D:02}da.tif',
                                                 lambert_proj.proj4_init)
# =============================================================================
#  read historical period raster
# =============================================================================
data_hist, metadata = reproject_raster(path_hist+pref_hist+f'{RI}yr{D:02}da.tif',
                                                 lambert_proj.proj4_init)

# =============================================================================
# clip data to minnesota shape 
# =============================================================================

# define mask 
mask = geometry_mask(
    minnesota.geometry,
    transform = metadata['transform'],
    invert = True,
    out_shape = (metadata['height'],metadata['width'])
    )

if clip_MN:
    data_futu = np.where(mask, data_futu, np.nan)
    data_hist = np.where(mask, data_hist, np.nan)
#%% plot ratio of futu/hist for chosen data / duration / return interval combo

plt.rcParams.update({'font.size': 14})

# Calculate the ratio between future and historical data
ratio = data_futu / data_hist

# Set up the figure and axis with Lambert Conformal projection
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': lambert_proj})

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

#levels for contours:
levels = np.concatenate([
    np.linspace(0.5, 1.0, 2, endpoint = False),
    np.linspace(1.0, 2.0, 3, endpoint = False),
    np.linspace(2.0, np.nanmax(ratio), 3)
    ])

#normalize values over 1.0 for extra contrast
norm = plt.Normalize(vmin=0, vmax = 2)

cax = ax.contourf(ratio, transform = lambert_proj,extent = extent,
                cmap='RdYlBu', origin='upper',
                levels = levels,
                norm=norm,
                )

# cax2 = ax.contourf(ratio, transform = lambert_proj,extent = extent,
#                 colors='gold', origin='upper',
#                 levels = [np.min(ratio),1.0],
#                 norm=norm,
#                 )

ax.set_extent([min_lon,max_lon, min_lat, max_lat])

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

# Add gridlines
ax.gridlines(draw_labels=True, x_inline=False,y_inline=False)

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path+'DDF_ratio_'+f'{scenarios[ix]}_{periods[ix2]}_{RI}yr_{D}da'
    save_path_name = save_path_name + '_MNclip' if clip_MN else save_path_name
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

