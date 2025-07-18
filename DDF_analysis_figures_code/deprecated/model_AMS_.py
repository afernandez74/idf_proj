import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import matplotlib.ticker as ticker
import seaborn as sns
plt.rcdefaults()
plt.style.use('seaborn-v0_8-poster')
#%% read AMS file

path = '../Data/AMS_modeled/'

adjusted = False

path = path + 'adjusted_' if adjusted else path + 'unadjusted_'

AMS_mod = xr.open_dataset(path + 'historical_AMS.nc')

# keep 1995 - 2010 data
AMS_mod = AMS_mod.isel(yearly=slice(0,16))

# keep only XX duration 
AMS_mod = AMS_mod['yearly_precip-inches-max_1-day'] if adjusted else AMS_mod['yearly_unadjustedPrecip-inches-max_1-day']
#%% read NOAA AMS 
# path to data
path = '../Data/AMS_NOAA/'

# name of file to read
name = '01d_AMS_NOAA_Stations.csv'

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

# keep 1995 - 2010 data
AMS_obs = AMS.sel(year = slice(1995,2010))
del df, df2, df_dat,dat,years,metadata_columns,name,path,AMS
#%% spread of AMS series spatially for each model and observations

qs = [0.25,0.5,0.75]

AMS_mod_spread = {}

for model in AMS_mod.model:
    ams = AMS_mod.sel(model = model)
    vals = ams.quantile(q=qs,dim=['lat','lon'])
    AMS_mod_spread[model.item()] = vals
    
AMS_obs_spread = AMS_obs.quantile(q=qs,dim=['id'])

#%% plot spread of AMS series

cmap = plt.get_cmap('jet', len(AMS_mod_spread)+1)
colors = [cmap(i) for i in range(len(AMS_mod_spread))]

# Define line styles for quantiles
line_styles = {0.25: '', 0.5: '-', 0.75: ''}

# Create a plot
plt.figure()

# Loop through the dictionary
for i, (key, data) in enumerate(AMS_mod_spread.items()):
    # Select the color for the current key
    color = colors[i]
    
    # Loop through the quantiles and plot the time series
    for quantile in data['quantile'].values:
        plt.plot(
            data['yearly'], 
            data.sel(quantile=quantile), 
            label=f'{key}' if quantile == 0.5 else None,
            linewidth = 3 if quantile == 0.5 else 1,
            color=color, 
            linestyle=line_styles[quantile]
        )

color = 'black'
line_styles = {0.25: ':', 0.5: '-', 0.75: ':'}

for quantile in AMS_obs_spread['quantile'].values:
    plt.plot(
        data['yearly'],
        AMS_obs_spread.sel(quantile=quantile).values, 
        label='OBS'if quantile == 0.5 else None,
        linewidth = 5 if quantile == 0.5 else 1,
        color=color, 
        linestyle=line_styles[quantile]
    )
# Add legend, labels, and title
plt.legend()
plt.xlabel('Year')
plt.ylabel('Value')
title = 'Observed vs modeled AMS series median'
title = title + ' adjusted' if adjusted else title + ' unadjusted'
plt.title(title)
plt.tight_layout()  # Adjust layout to fit the legend

# Show the plot

plt.show()

#%% gridcell point spread

lat = 48.8947
lon = -95.33

AMS_mod_pt = AMS_mod.sel(lat = lat, lon = lon,method = 'nearest')
AMS_obs_pt = AMS_obs.sel(id = 181)
        
# Create a plot
fig, ax = plt.subplots()
for model in AMS_mod_pt:
    ax.plot(model.yearly,model.values, 
            label = model.model.values, 
            linewidth = 3 if model.model.values == 'ensemble' else 1.5)
ax.plot(model.yearly,AMS_obs_pt, label = 'OBS', color = 'black', linewidth = 3)

ax.legend()
title = f'AMS series {str(AMS_obs_pt["name"].values)}'
title = title + ' adjusted'if adjusted else title + ' unadjusted'
plt.title(title)