import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
import seaborn as sns
import xarray as xr
from funcs import reproject_raster, init_lambert_proj, load_minnesota_reproj, create_dataarray

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
D = 10

#save path results
save_path = '../Figures/DDF_viols/'
save_path = save_path + 'clip_MN/' if clip_MN else save_path + 'whole/'

# =============================================================================
# Set paths
# =============================================================================
path = base_path +'ensemble/' if ensemble else base_path + 'separate/'
path = path +'smooth/' if smooth else path + 'unsmooth/'
path = path +'adjusted/' if adjusted else path + 'unadjusted/'

# path to historical data 
path_hist = path + 'historical/1995-2014/'

files_hist_all = [path_hist + file for file in os.listdir(path_hist) if not file.startswith('.')]

#%% paths for all files 

scenarios = [file for file in os.listdir(path) if not file.startswith('.') and not file.startswith('historical')]

paths_scenarios = [path + scenario for scenario in scenarios]

paths_all = []
for path in paths_scenarios:
    files = [file for file in os.listdir(path) if not file.startswith('.')]
    paths_temp = [path + '/' + file for file in files]
    paths_all.append(paths_temp)

paths_all = [item for row in paths_all for item in row]

files_all = []
for path in paths_all:
    files = [file for file in os.listdir(path) if not file.startswith('.')]
    files_temp = [path + '/' + file for file in files]
    files_all.append(files_temp)

files_all = [item for row in files_all for item in row]

del files_temp, paths_temp, files, paths_all, paths_scenarios, path

#%% filter paths for correct files
        
#sufix for specified return interval and duration
suf = f'{RI}yr{D:02}da'

#filter paths for those with specified return interval and duration
files = [s for s in files_all if suf in s]

# final lists of files to be loaded and analyzed
files_hist = [s for s in files_hist_all if suf in s]

#%% load minnesota outline and projection for maps
lambert_proj = init_lambert_proj()
minnesota = load_minnesota_reproj(lambert_proj)
#%% open raster files and reproject

# =============================================================================
#  read projection rasters
# =============================================================================
data_futu=[]
for file in files:
    name = file[file.find('unadjusted_') + 11 : file.find('.tif')]
    data_futu_temp, metadata_temp = reproject_raster(file,lambert_proj.proj4_init)
    data_futu_da = create_dataarray(data_futu_temp, metadata_temp, name)
    data_futu.append(data_futu_da)

data_futu = xr.concat(data_futu,dim='source')
# =============================================================================
#  read historical period raster
# =============================================================================
data_hist=[]
metadata_hist=[]
for file in files_hist:
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
    for name, da in data_futu.groupby('source'):
        data_futu.loc[dict(source=name)] = np.where(mask, da.squeeze(), np.nan)
    data_hist[0] = np.where(mask, data_hist[0], np.nan)

#%% calculate ratios
ratios = data_futu.copy()

for name, da in data_futu.groupby('source'):
    ratios.loc[dict(source=name)] = da.squeeze().values/data_hist[0]
ratios.name = 'ratios'
ratios_df = ratios.to_dataframe().reset_index()
#%% violin

ratios_df = ratios_df.sort_values(by = ['scenario', 'period'])
plt.figure(figsize=(10, 6))
sns.violinplot(x='source', y='ratios', hue='period', data=ratios_df, inner="quart", palette="muted", scale="width")
plt.title(f'Distribution of ratios for {suf}')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.ylim(0.5,2.5)
plt.grid(zorder = 0.5)

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'DDF_viol_' + suf

    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

