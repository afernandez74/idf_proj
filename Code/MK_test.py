import pandas as pd
import numpy as np
import pymannkendall as mk
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from geodatasets import get_path
import os

# path to data
path = '../../Data/AMS_NOAA'

# name of file to read
name = '01d_AMS_NOAA_Stations.csv'

# read in AMS data

df = pd.read_csv(path +name)

# tidy up pandas dataframe
df_AMS = df.iloc[:, 4:].T # series as columns
df_AMS = df_AMS.reset_index() # make years column instead of having them as index
df_AMS.rename(columns={'index': 'Year'}, inplace=True)

#%% mann-kendall trend test

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

# Initialize a list to store results
results = []

# Iterate over each column in the DataFrame and perform the Mann-Kendall test
c=0
for column in df_AMS.columns:
    dat = df_AMS[column]
    result = mk_test(dat)
    results.append(result)
    c=c+1

results_df = pd.DataFrame(results[1:])
results_df = pd.concat([results_df, df.iloc[:, :5]], axis = 1)

#%% load minnesota outline for map 

url = "https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip"
usa = gpd.read_file(url)
minnesota = usa[usa['NAME'] == 'Minnesota']

#%% map results

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
plt.title(name[:-4]+' Mann-Kendall trend test slopes and trend analysis')


# Plot Minnesota outline
minnesota.boundary.plot(ax=ax, color='black')


# Set circle colors based trend
cats = results_df['trend'].unique()
color_dict = {
    'decreasing': '#FF0000',  # Red
    'no trend': '#808080',  # Gray
    'increasing': '#0000FF',  # Blue
}

colors = [color_dict[cat] for cat in results_df['trend']]

# Create circles
circles = []
colors = []
for x, y, slope, category, p_val in zip(results_df['lon'], results_df['lat'], results_df['slope'], results_df['trend'],results_df['p']):
    if p_val < 0.05:  # Example criterion: fill circles where value > 5
        circle = Circle((x, y), radius=slope*10, facecolor=color_dict[category], edgecolor=color_dict[category], alpha=0.6)
    else:
        circle = Circle((x, y), radius=slope*10, facecolor='none', edgecolor=color_dict[category], alpha=0.6)
    circles.append(circle)
    colors.append(color_dict[category])

# Add circles to the plot
for circle in circles:
    ax.add_patch(circle)

# # Set plot limits to Minnesota's bounds
ax.set_xlim(minnesota.total_bounds[[0, 2]])
ax.set_ylim(minnesota.total_bounds[[1, 3]])

# Add a legend for categories
for category, color in color_dict.items():
    ax.scatter([], [], c=color, label=category)
category_legend = ax.legend(title='Categories', loc='center left', bbox_to_anchor=(1, 0.6))

# Add a legend for circle sizes
size_values = [results_df['slope'].min()*10, results_df['slope'].mean()*10, results_df['slope'].max()*10]
size_labels = ['Min', 'Mean', 'Max']

# Create a separate axis for size legend
size_legend_ax = fig.add_axes([0.75, 0.15, 0.2, 0.3], frameon=False)
size_legend_ax.set_xlim(0, 1)
size_legend_ax.set_ylim(0, 1)
size_legend_ax.axis('off')

# Plot circles and labels
for i, (value, label) in enumerate(zip(size_values, size_labels)):
    y = 0.8 - (i * 0.3)  # Distribute circles vertically
    circle = plt.Circle((0.2, y), radius=value, fill=False, edgecolor='black')
    size_legend_ax.add_artist(circle)
    size_legend_ax.text(0.5, y, f'{label}: {value:.2f}', va='center', ha='left')

size_legend_ax.set_title('Value Scale', fontweight='bold')

# Adjust layout and show plot
plt.tight_layout()

save_option = input("Save figure? (y/n): ").lower()
save_path = os.path.expanduser('~/idf_cmip6_local/AMS_NOAA/figs/')

if save_option == 'y':
    save_path_name = save_path+'MK_'+name[0:3]+'_slopes_trend'
    # Save as SVG
    plt.savefig(save_path_name +'.svg', format='svg', dpi=300, bbox_inches='tight')
    # Save as PNG
    plt.savefig(save_path_name +'.png', format='png', dpi=300, bbox_inches='tight')
else:
    plt.show()
