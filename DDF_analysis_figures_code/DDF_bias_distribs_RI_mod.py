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

# calculates model-specific bias ratio compared to NA14

#%% specify return interval and duration for analysis

clip_MN = True #if true, raster data clipped to MN 

# Duration (days)
D = 4

#save path results
save_path = '../Figures/Bias/'
save_path = save_path + 'clip_MN/' if clip_MN else save_path + 'whole/'

# =============================================================================
# load minnesota outline and projection for maps
# =============================================================================

lambert_proj = init_lambert_proj()
shape_path = "/Users/afer/idf_cmip6_local/idf_repo/Data/tl_2022_us_state.zip"
minnesota = load_minnesota_reproj(lambert_proj,shape_path)

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
paths_all_hist = [file for file in paths_all_hist if f'{D:02}da' in file]

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

file = [file for file in files if f'{D:02}da' in file]

file = [file for file in file if file.endswith('.asc')]

NA14_paths = []
for file_i in file:
    file_i_path = base_path + file_i
    file_i_path_new = file_i_path.replace('.asc','.tif')
    NA14_paths.append(file_i_path_new)
    read_asc_to_geotiff(file_i_path, file_i_path_new)
    
del file_i_path,file_i_path_new,file, files

#%% open raster files and reproject (NA14)
# Reproject the NA14 GeoTIFF

NA14=[]
for file_i in NA14_paths:
    
    NA14_reproj_i, NA14_metadata_i = reproject_raster(file_i, lambert_proj.proj4_init)
    
    # create data array from reprojected NA14 raster
    
    transform = NA14_metadata_i['transform']
    width = NA14_metadata_i['width']
    height = NA14_metadata_i['height']
    crs = NA14_metadata_i['crs']
    
    # Calculate the x and y coordinates using the affine transform
    x_coords = np.arange(width) * transform[0] + transform[2]
    y_coords = np.arange(height) * transform[4] + transform[5]
        
    RI_da = re.search(r'mw(\d{1,3})yr', file_i).group(1)
    
    # Create DataArray
    data_NA14 = xr.DataArray(
        NA14_reproj_i,
        dims=["y", "x"],
        coords={
            "x": ("x", x_coords),
            "y": ("y", y_coords),
            "RI":int(RI_da),
            "D":D,
            "source":f'NA14_{RI_da}yr{D:02}da'
    
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
    NA14_i = NA14_clip
    
    # get rid of negative values and outliers
    NA14_i.values[NA14_i.values < 0.0] = np.nan
    threshold = 3
    cond = (NA14_i < NA14_i.mean() - threshold * NA14_i.std()) | (NA14_i > NA14_i.mean() + threshold * NA14_i.std())
    
    NA14_i = NA14_i.where(~cond,np.nan)
    
    #add dataarray to data_NA14 list
    NA14.append(NA14_i)

NA14 = xr.concat(NA14, dim='source')

del NA14_clip,data_NA14_resamp,data_NA14, transform, width, height, crs, NA14_reproj_i, RI_da,NA14_i

#%% Calculate bias (NA14 / CMIP6) for each model

bias = data_hist.copy()

for idx in range(len(data_hist['source'])):
    # Select the data for the current model from both DataArrays using integer indexing on 'source'
    da_i = data_hist.isel(source=idx)

    # find NA14 entry with same RI 
    NA14_i = NA14.where(NA14.coords['RI'] == int(da_i.RI), drop = True).squeeze()
    
    # Perform the percent change calculation (avoid division by zero by masking NaNs)
    bias_i = (da_i / NA14_i.values)
    
    # Assign the calculated percent change to the corresponding location in the 'change' DataArray
    bias.loc[dict(source=da_i.source)] = bias_i

bias = bias.where(bias.coords['RI'] != 200, drop = True)
#%% plot spread of bias as box plots separated by RI and model

df = bias.to_dataframe(name="bias").reset_index()
df = df.dropna(subset=["bias"])
df = df.sort_values('model')
df = df.sort_values("RI")

# Create a faceted box plot grid
g = sns.FacetGrid(df, 
                  col="model", 
                  col_wrap=3, 
                  sharey=True, 
                  height=4,
                  col_order = sorted(df["model"].unique()))

g.map(sns.boxplot, "RI", "bias", 
      order=sorted(df["RI"].unique()), 
      showfliers = False,
      palette="bright",
      # inner='quart',
      # scale = 'width'
      )

for ax in g.axes.flat:
    ax.grid(axis='y', which='major', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)  # Horizontal gridlines
    ax.grid(axis='x', which='major', linestyle='', linewidth=0)  # No vertical gridlines
    ax.set_axisbelow(True)  # Ensure gridlines are drawn below the plots
    
g.fig.suptitle(f"Comparison of Bias Across Models and RI for {D}day events", 
               fontsize=16, 
               y=1.02)  # Adjust 'y' to control the vertical position of the title
g.set(ylim=(0.0, 2.0))
# Adjust plot aesthetics
g.set_titles("{col_name}")
g.set_axis_labels("RI [yr]", "Bias")
g.set(xticks=range(len(df["RI"].unique())), xticklabels=sorted(df["RI"].unique()))
g.tight_layout()

# Show the plot
plt.tight_layout()

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'Bias_spreads_all_mods_'+f'{D}da_box' 
    save_path_name = save_path_name + '_MNclip' if clip_MN else save_path_name
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()