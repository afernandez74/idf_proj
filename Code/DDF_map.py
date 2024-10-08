import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
import os
#%% load minnesota outline for map 
url = "https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip"
usa = gpd.read_file(url)
minnesota = usa[usa['NAME'] == 'Minnesota']

#%% Choose rasters to plot

base_path = '../Data/DDF_tif/'

# =============================================================================
# # options
# =============================================================================
ensemble = False
smooth = True
adjusted = False

path = base_path +'ensemble/' if ensemble else base_path + 'separate/'
path = path +'smooth/' if smooth else path + 'unsmooth/'
path = path +'adjusted/' if adjusted else path + 'unadjusted/'

scenarios = os.listdir(path)
print('Scenarios:')
for i, item in enumerate (scenarios):
    print(f'{i}:    {item}')

# =============================================================================
# # pick scenario
# =============================================================================
while True:
    try:
        ix = int(input(f"Enter the number of your choice (1-{len(scenarios)}): "))
        if 0 <= ix <= len(scenarios):
            break
        else:
            print(f"Please enter a number between 1 and {len(scenarios)}.")
    except ValueError:
        print("Please enter a valid number.")
path = path + scenarios[ix] + '/'

periods = os.listdir(path)
print('Periods:')
for i, item in enumerate (periods):
    print(f'{i}:    {item}')

# =============================================================================
# # pick period
# =============================================================================
while True:
    try:
        ix = int(input(f"Enter the number of your choice (1-{len(periods)}): "))
        if 0 <= ix <= len(periods):
            break
        else:
            print(f"Please enter a number between 1 and {len(periods)}.")
    except ValueError:
        print("Please enter a valid number.")
path = path + periods[ix] + '/'

files = os.listdir(path)

# =============================================================================
# 
# =============================================================================
#%% load data with rasterio as a dictionary

tif_files = [os.path.join(path, f) for f in os.listdir(path) 
             if os.path.splitext(f)[1].lower() == '.tif']

rasters = {}

for tif_file in tif_files:
    file_name = os.path.splitext(os.path.basename(tif_file))[0]
    rasters[file_name] = rasterio.open(tif_file)
    
# with rasterio.open('separate_smooth_unadjusted_historical_1995-2014_100yr01da.tif') as src:
#     print(src.width)  # Width of the raster
#     print(src.height)  # Height of the raster
#     print(src.count)  # Number of bands
#     print(src.crs)  # Coordinate Reference System
#     data_hist = src.read(1)  # Read the first band
#     data_hist[data_hist < 0] = 0
#     transform_hist = src.transform


# with rasterio.open('separate_smooth_unadjusted_ssp585_2060-2079_100yr01da.tif') as src:
#     print(src.width)  # Width of the raster
#     print(src.height)  # Height of the raster
#     print(src.count)  # Number of bands
#     print(src.crs)  # Coordinate Reference System
#     data_futu = src.read(1)  # Read the first band
#     data_futu[data_futu < 0] = 0
#     transform_futu = src.transform


#%%

plt.imshow(data_futu, cmap='Greys_r')  # Display the first band with a greyscale colormap
plt.colorbar()  # Optional: Add a colorbar
plt.title('Raster Band 1')
plt.show()
#%%
plt.rcParams.update({'font.size': 14})
# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Display the raster data using imshow
cax = ax.imshow(data_futu/data_hist, cmap='YlGnBu', extent=(
    transform_hist.c, transform_hist.c + transform_hist.a * data_hist.shape[1],
    transform_hist.f + transform_hist.e * data_hist.shape[0], transform_hist.f
))

# Add a colorbar
cbar = fig.colorbar(cax, ax=ax, label='Change factor')

minnesota.boundary.plot(ax=ax, color='black', linewidth=3)  # Change color and linewidth as needed

# Set titles and labels
ax.set_title('Event depth change from historical to 2060-2079 \n 100 year event, 1 day duration')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show the plot
plt.show()