#%%
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import matplotlib.path as mpath
from scipy.interpolate import griddata, NearestNDInterpolator
from geodatasets import get_path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from funcs import mk_test
plt.rcdefaults()
plt.style.use('seaborn-v0_8-poster')

#%%
# path to data
path = '../Data/AMS_NOAA/'

#save path
save_path = '../Figures/MK_trend_maps/'

# name of file to read
name = '07d_AMS_NOAA_Stations.csv'

# minimum length of data series to analyze
min_years = 100


# read in AMS data
df = pd.read_csv(path + name)
df2 = pd.read_csv(path+name,header = None) # for year values

# First 4 columns contain metadata
metadata_columns = df.columns[0:4].tolist()
metadata = df[metadata_columns]

# Dataframe without metadata
df_dat = df.drop(columns=metadata_columns).T
dat = df_dat.values

years = df2.iloc[0,4:].values.astype(int)

AMS = xr.DataArray(
    data = dat,
    dims = ['year','id'],
    coords ={
        'year' : years,
        'id' : df_dat.columns,
        'lat' : ('id', metadata['lat'].values),
        'lon' : ('id', metadata['lon'].values),
        'name' : ('id', metadata['station name']),
        'code' : ('id', metadata['station code']),
        },
    name = "AMS"
)

# keep only the data in time range
AMS = AMS.sel(year = slice(1900,2010))

#%% Filter AMS series by minimum year and perform MK trend test

# filter to series that meet minimum length requirement
AMS_lon_ix = AMS.count(dim='year') >= min_years
AMS_lon = AMS.where(AMS_lon_ix, drop = True)

# drop NANs from the leftover series and pick the last min_years in each for analysis
AMS_filt =[]

for i in range(AMS_lon.shape[1]):
    temp = AMS_lon[:,i].dropna(dim='year')
    temp = temp.isel(year=slice(-min_years,None))
    AMS_filt.append(temp)
    
AMS_filt = xr.concat(AMS_filt,dim='id')#.sel(year = slice(1900,2000))

mk_results = []

for i in range(AMS_filt.shape[0]):
    result = mk_test(AMS_filt.isel(id=i))
    mk_results.append(result)

# create dataset of results
mk_results_ds = xr.Dataset(
    {key: ('id', [res[key] for res in mk_results]) for key in mk_results[0].keys()},
    coords = {'id': AMS_filt.coords['id']}
    )

# merge dataset of results with AMS array
AMS_filt_mk = AMS_filt.to_dataset(name = 'AMS')
AMS_filt_mk = AMS_filt_mk.merge(mk_results_ds)

res_pd = AMS_filt_mk.mean(dim = 'year').to_pandas()

#%% load minnesota outline for map 

url = "https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip"
usa = gpd.read_file(url)
minnesota = usa[usa['NAME'] == 'Minnesota']

# Lambert Conformal Conic projection 
lambert_proj = ccrs.LambertConformal(
    central_longitude = AMS_filt_mk.lon.mean(dim='id').item(),
    central_latitude = AMS_filt_mk.lat.mean(dim='id').item(),
    standard_parallels = (33,45)
    )

#%% map trend test results 

#dataset to plot
AMS_mk = AMS_filt_mk

# var to plot = normalized slope
var = (AMS_mk.slope) /(AMS_mk.slope.max()) 


# latitudes and longitudes for map extent
min_lon, min_lat = -97.94, 42.54
max_lon, max_lat = -88.69, 49.97

#significance value for plot
sig = 0.05

# Create figure and axis
fig, ax = plt.subplots(subplot_kw={'projection': lambert_proj})

plt.title(name[:-4]+f' MK trend test min {min_years}yrs')

# draw political boundaries 
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.RIVERS, linestyle = '--', 
               color = 'lightblue', linewidth = 0.4,zorder=2)
ax.add_feature(cfeature.LAKES, linestyle = '--', 
               color = 'lightblue', linewidth = 0.4,zorder=2)
minnesota = minnesota.to_crs(lambert_proj.proj4_init)
minnesota.boundary.plot(ax=ax, color='black', linewidth=1.5,zorder = 9)

#colormap
cmap = plt.cm.bwr_r
norm = colors.Normalize(vmin=-var.max(), vmax=var.max())

# =============================================================================
# points plot 
# =============================================================================
# Plot the points
scatter = ax.scatter(
    AMS_mk.lon, AMS_mk.lat,
    c='none',
    s=80,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    edgecolor=plt.cm.bwr_r(norm(var)),
    linewidth=2,
    alpha=1,
    zorder = 10
)

# plot black outline to all points
scatter_outlines = ax.scatter(
    AMS_mk.lon, AMS_mk.lat,
    c='none',
    s=140,
    transform=ccrs.PlateCarree(),
    # cmap=cmap,
    # norm=norm,
    edgecolor='black',
    linewidth=1,
    alpha=1,
    zorder = 9
)

# Create a mask for significant p-values
significant = AMS_mk.p < sig  # Adjust threshold as needed

# Plot filled circles for significant points
scatter2 = ax.scatter(
    AMS_mk.lon.where(significant), AMS_mk.lat.where(significant),
    c=var.where(significant),
    s=80,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    alpha=1.0,
    zorder = 11
)

# =============================================================================
# contour plot in background
# =============================================================================
#grid for contouf plot
xx,yy = np.meshgrid(np.linspace(min_lon,max_lon,100),
                    np.linspace(min_lat,max_lat,100))
#interp z data
zz = griddata((AMS_mk.lon.values, AMS_mk.lat.values), 
                       var.values, (xx, yy), method='cubic')

# Transform the grid to Lambert projection
lambert_transformer = lambert_proj.transform_points(ccrs.PlateCarree(), xx, yy)
xx_lambert, yy_lambert = lambert_transformer[..., 0], lambert_transformer[..., 1]

# Create a nearest-neighbor interpolator
nn_interpolator = NearestNDInterpolator(
    list(zip(AMS_mk.lon, AMS_mk.lat)), 
    var
)

# Fill NaN values (outside data bounds) with nearest-neighbor interpolation
mask = np.isnan(zz)
zz[mask] = nn_interpolator(xx[mask], yy[mask])

# Convert minnesota geometry to a path
minnesota_path = mpath.Path(minnesota.geometry.iloc[0].exterior.coords)

# Create a mask for points inside Minnesota using the Lambert coordinates
mask = minnesota_path.contains_points(np.column_stack((xx_lambert.flatten(), yy_lambert.flatten()))).reshape(xx.shape)
# Apply the mask to zz
zz_masked = np.ma.masked_where(~mask, zz)


cont_plt = ax.contourf(xx,yy,
                       zz_masked,
                       levels = 10,
                       cmap = cmap,
                       norm=norm,
                       transform = ccrs.PlateCarree(),
                       alpha = 0.25,
                       extend = 'both',
                       zorder = 5)

# # Remove contour lines
for collection in cont_plt.collections:
    collection.set_edgecolor("face")
    collection.set_linewidth(0)

# Add colorbar
plt.colorbar(scatter2, label=f'Normalized {var.name}')

# Add gridlines
ax.gridlines(draw_labels=True, x_inline=False,y_inline=False,zorder=1,
             linewidth = 1)

# Set extent (adjust these values based on your data)
ax.set_extent([min_lon,max_lon, min_lat, max_lat])

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path+'MK_'+name[0:3]+'_slopes_trend_'+f'min_{min_years}yrs'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

#%% print summary of reuslts
N = AMS_mk['id'].count().values
signif_N = (AMS_mk['trend']=='increasing').sum()
signif_percent = signif_N/N*100
print(f'{int(signif_percent)}% of {N} stations with significant trend')