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
#%% load minnesota outline for map 

url = "https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip"
usa = gpd.read_file(url)
minnesota = usa[usa['NAME'] == 'Minnesota']


#%% Read in data 

path = '../Data/AMS_NOAA/'
file = '01d_AMS_NOAA_counts.csv'

df = pd.read_csv(path+file)

# path for saving maps
save_path = '../Figures/NOAA_sttns_map/'

#%% plot all stations 
# Create map
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot Minnesota outline
minnesota.boundary.plot(ax=ax, color='black')

# Plot stations
scatter = ax.scatter(df['lon'], df['lat'], c=df['Count'], cmap='Reds', 
                     s=30, transform=ccrs.PlateCarree())
# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, label='Timeseries Length')

# Set extent (adjust these values based on your data)
ax.set_extent([df['lon'].min()-1, df['lon'].max()+1, 
               df['lat'].min()-1, df['lat'].max()+1], crs=ccrs.PlateCarree())

# Add gridlines
ax.gridlines(draw_labels=True)

# Add title
plt.title('Weather Station Timeseries Lengths')

# Optionally, add station names as labels
# for idx, row in df.iterrows():
#     ax.text(row['lon'], row['lat'], row['Station Name'], fontsize=8, 
#             ha='right', va='bottom', transform=ccrs.PlateCarree())

plt.savefig(save_path + 'weather_stations_map.png', dpi=300, bbox_inches='tight')
# plt.show()
#%% plot only >XXX year length stations

min_len = 100

# Create map
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot Minnesota outline
minnesota.boundary.plot(ax=ax, color='black')

# Plot stations
df_100 = df[df['Count'] > min_len]

scatter = ax.scatter(df_100['lon'], df_100['lat'], c=df_100['Count'], cmap='gist_heat_r', 
                     s=30, transform=ccrs.PlateCarree())
# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, label='Timeseries Length')

# Set extent (adjust these values based on your data)
ax.set_extent([df['lon'].min()-1, df['lon'].max()+1, 
               df['lat'].min()-1, df['lat'].max()+1], crs=ccrs.PlateCarree())

# Add gridlines
ax.gridlines(draw_labels=True)

# Add title
plt.title('Weather Station Timeseries Lengths')

# Optionally, add station names as labels
# for idx, row in df.iterrows():
#     ax.text(row['lon'], row['lat'], row['Station Name'], fontsize=8, 
#             ha='right', va='bottom', transform=ccrs.PlateCarree())

save_option = input("Save figure? (y/n): ").lower()

if save_option == 'y':
    # Save as SVG
    plt.savefig(save_path +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()

plt.show()
