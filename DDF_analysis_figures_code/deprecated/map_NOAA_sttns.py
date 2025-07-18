#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:53:01 2024

@author: afer
"""
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

#%% load minnesota outline for map 

url = "https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip"
usa = gpd.read_file(url)
minnesota = usa[usa['NAME'] == 'Minnesota']


#%% Read in data 

path = '../Data/AMS_NOAA/'
file = '01d_AMS_NOAA_Stations.csv'

# read in AMS data
df = pd.read_csv(path + file)
df2 = pd.read_csv(path + file,header = None) # for year values

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

# path for saving maps
save_path = '../Figures/NOAA_sttns/'

#%% projection 
# Lambert Conformal Conic projection 
lambert_proj = ccrs.LambertConformal(
    central_longitude = df['lon'].mean(),
    central_latitude = df['lat'].mean(),
    standard_parallels = (33,45)
    )

url = "https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip"
usa = gpd.read_file(url)
minnesota = usa[usa['NAME'] == 'Minnesota']

#%% plot timeseries of amount of stations per year

num_stns = AMS.count(dim = 'id',keep_attrs=True)
median_AMS = AMS.median(dim = 'id')

q25_AMS = AMS.quantile(0.25,dim = 'id')
q75_AMS = AMS.quantile(0.75,dim = 'id')
iqr_AMS = q75_AMS - q25_AMS

years = AMS.year.values

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('AMS spread', color=color)
ax1.plot(years, q25_AMS, color=color, linestyle = '--', linewidth = 0.5)
ax1.plot(years, q75_AMS, color=color, linestyle = '--', linewidth = 0.5)
ax1.plot(years,median_AMS, color = color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Count', color=color)  # we already handled the x-label with ax1
ax2.plot(years, num_stns, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'stations_length_spread_TS'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
#%% spread of station lenghts
stn_len = AMS.count(dim = 'year',keep_attrs=True).to_series()

fig, ax = plt.subplots()

sns.histplot(data=stn_len,
             # kde=True, 
             binwidth = 10, 
             ax = ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

# stn_len.plot.hist(bins = 16, orientation = 'vertical', 
#                   rwidth = 0.9)

#%% plot all stations 

# latitudes and longitudes for map extent
min_lon, min_lat = -97.94, 42.54
max_lon, max_lat = -88.69, 49.97

# Create map
fig, ax = plt.subplots(subplot_kw={'projection': lambert_proj})

# draw political boundaries 
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.RIVERS, linestyle = '--', 
               color = 'lightblue', linewidth = 0.4,zorder=1)
ax.add_feature(cfeature.LAKES, linestyle = '--', 
               color = 'lightblue', linewidth = 0.4,zorder=1)
minnesota = minnesota.to_crs(lambert_proj.proj4_init)
minnesota.boundary.plot(ax=ax, color='black', linewidth=1.5,zorder = 9)

# calculate amount of data points in each station
counts = AMS.notnull().sum(dim='year')

#normalize values 
norm = plt.Normalize(vmin = counts.min().values,
                     vmax = counts.max().values)

# Plot stations
scatter = ax.scatter(AMS.lon, AMS.lat, c=counts, cmap='hot_r', 
                     s=70, transform=ccrs.PlateCarree(),zorder = 10,edgecolor='black',
                     linewidth=0.5)
# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, label='Timeseries Length')

# Add gridlines
ax.gridlines(draw_labels=True, x_inline=False,y_inline=False,zorder=1,
             linewidth = 1)

# Set extent (adjust these values based on your data)
ax.set_extent([min_lon,max_lon, min_lat, max_lat])

# Add title
plt.title('Weather Station Timeseries Lengths')

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    save_path_name = save_path + 'stations_length_map'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

