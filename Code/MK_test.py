#%%
import pandas as pd
import xarray as xr
import pymannkendall as mk
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from geodatasets import get_path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

#%%
# path to data
path = '../Data/AMS_NOAA/'

#save path
save_path = '../Figures/MK_trend_maps/'

# name of file to read
name = '60d_AMS_NOAA_Stations.csv'

# minimum length of data series to analyze
min_years = 80

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
#%% define functions

#mann kendall function
def mk_test(series):
    
    result = mk.original_test(series)
    return {
        'trend': result.trend,
        'h': result.h,
        'p': result.p,
        'z': result.z,
        'Tau': result.Tau,
        's': result.s,
        'var_s': result.var_s,
        'slope': result.slope,
        'intercept': result.intercept,
        'dat_len': series.count()  # Add dat_len as an entry
    }
#%% Filter AMS series

# filter to series that meet minimum length requirement
AMS_lon_ix = AMS.count(dim='year') >= min_years
AMS_lon = AMS.where(AMS_lon_ix, drop = True)

# drop NANs from the leftover series and pick the last min_years in each for analysis
AMS_filt =[]

for i in range(AMS_lon.shape[1]):
    temp = AMS_lon[:,i].dropna(dim='year')
    temp = temp.isel(year=slice(-min_years,None))
    AMS_filt.append(temp)
    
AMS_filt = xr.concat(AMS_filt,dim='id')

#%% Perform MK test on filtered AMS array
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

#significance value for plot
sig = 0.05

# colors for circles
trend_clr = {'no trend': 'gray',
             'increasing': 'blue',
             'decreasing': 'red'
}

#max and min slopes
slope_hi = AMS_filt_mk['slope'].max().item()


# Colormap for slope values
cmap = plt.cm.rainbow_r

# Create figure and axis
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': lambert_proj})

plt.title(name[:-4]+' MK trend test')

# draw political boundaries 
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.RIVERS, linestyle = '--', 
               color = 'lightblue', linewidth = 0.4,zorder=1)
ax.add_feature(cfeature.LAKES, linestyle = '--', 
               color = 'lightblue', linewidth = 0.4,zorder=1)
minnesota = minnesota.to_crs(lambert_proj.proj4_init)
minnesota.boundary.plot(ax=ax, color='black', linewidth=1.5)

# iterate through each location and plot circle

for i in range(AMS_filt_mk.dims['id']):
    lat = AMS_mk['lat'].isel(id=i).item()
    lon = AMS_mk['lon'].isel(id=i).item()
    trend = AMS_mk['trend'].isel(id=i).item()
    slope = AMS_mk['slope'].isel(id=i).item()
    p_val = AMS_mk['p'].isel(id=i).item()
    t = AMS_mk['Tau'].isel(id=i).item()
    
    # determine color, radius and significance 
    clr = trend_clr.get(trend)
    rad = 100 * abs(slope)/slope_hi
    
    #plot circle
    ax.scatter(lon,lat, color = clr, s = rad,
               transform = ccrs.PlateCarree(),zorder=2)#,
               #edgecolor='black' if p_val <= sig else 'none')
               
# Add gridlines
ax.gridlines(draw_labels=True, x_inline=False,y_inline=False)

# Set extent (adjust these values based on your data)
ax.set_extent([df['lon'].min()-1, df['lon'].max()+1, 
               df['lat'].min()-1, df['lat'].max()+1], crs=ccrs.PlateCarree())


save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path+'MK_'+name[0:3]+'_slopes_trend_'+f'min_{min_years}yrs'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

