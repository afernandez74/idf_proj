import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import seaborn as sns
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

files_hist_all = [file for file in os.listdir(path_hist) if not file.startswith('.')]

# ==============================================================================
# Pick future scenario  - instead of choosing, calculate them all and plot violins of each
# so 9 violins per plot
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

# list of files in selexted paths - projections
periods_futu = [file for file in os.listdir(path) if not file.startswith('.')]

#%% filter paths for correct files

files_all = []

# find all paths in the specified scenario
for period in periods_futu:
    pathy = path + period + '/'
    filys = [file for file in os.listdir(pathy) if not file.startswith('.')]
    for file in filys:
        files_all.append(period+'/'+file)
        
#sufix for specified return interval and duration
suf = f'{RI}yr{D:02}da'

#filter paths for those with specified return interval and duration
files = [s for s in files_all if suf in s]
#paths for desired files
paths = [path + name for name in files]

files_hist = [s for s in files_hist_all if suf in s]
paths_hist = [path_hist + file for file in files_hist]

#%% load minnesota outline and projection for maps
lambert_proj = init_lambert_proj()
minnesota = load_minnesota_reproj(lambert_proj)
#%% open raster files and reproject

# =============================================================================
#  read projection raster 
# =============================================================================
data_futu=[]
metadata_futu=[]
for file in paths:
    data_futu_temp, metadata_temp = reproject_raster(file,lambert_proj.proj4_init)
    data_futu.append(data_futu_temp)
    metadata_futu.append(metadata_temp)

# =============================================================================
#  read historical period raster
# =============================================================================
data_hist=[]
metadata_hist=[]
for file in paths_hist:
    data_hist_temp, metadata_temp = reproject_raster(file,lambert_proj.proj4_init)
    data_hist.append(data_hist_temp)
    metadata_hist.append(metadata_temp)

# =============================================================================
# clip data to minnesota shape 
# =============================================================================
metadata = metadata_hist[0]
# define mask 
mask = geometry_mask(
    minnesota.geometry,
    transform = metadata['transform'],
    invert = True,
    out_shape = (metadata['height'],metadata['width'])
    )
# perform the clip

if clip_MN:
    for i in range(len(data_futu)):
        data_futu[i] = np.where(mask, data_futu[i], np.nan)
    data_hist[0] = np.where(mask, data_hist[0], np.nan)
#%% calculate ratios
ratios = []
for i in range(len(data_futu)):
    ratio = data_futu[i] / data_hist[0]
    ratios.append(ratio)

ratios_df = pd.DataFrame({
    periods_futu[0]:ratios[0].flatten(),
    periods_futu[1]:ratios[1].flatten(),
    periods_futu[2]:ratios[2].flatten()
})
ratios_df = ratios_df.sort_index(axis=1)
#%% plot distribution of ratios

for ratio in ratios:
    sns.kdeplot(ratio.flatten(), shade=True)
plt.title("Kernel Density Estimate of Raster Values")
plt.xlabel("Pixel Value")
plt.ylabel("Density")
plt.show()

#%% violin
for ratio in ratios:    
    sns.violinplot(data=ratios_df,fill=True)
plt.title("Violin Plot of Raster Values")
plt.xlabel("Pixel Value")
plt.show()
