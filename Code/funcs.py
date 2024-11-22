import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import cartopy.crs as ccrs
import geopandas as gpd
import xarray as xr
import pymannkendall as mk
#%% mann kendall trend test function
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

#%% reprojection function for raster DDF files

def reproject_raster(input_path, dst_crs):
    """
    Reproject a single-band raster to a specified coordinate reference system (CRS) and return it as a variable.
    
    Parameters:
    - input_path (str): Path to the input raster file (.tif).
    - dst_crs (dict or str): Destination CRS in PROJ4 or EPSG format (e.g., 'EPSG:4326' or '+proj=...').
    
    Returns:
    - reprojected_data (numpy array): The reprojected raster data as a numpy array.
    - metadata (dict): Metadata of the reprojected raster, including CRS, transform, and other info.
    """
    # Open the input raster file
    with rasterio.open(input_path) as src:
        # Calculate the transform and other metadata for the new projection
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        
        # Create metadata for the reprojected data
        metadata = src.meta.copy()
        metadata.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Create an empty array to store the reprojected data
        reprojected_data = np.empty((height, width), dtype=src.meta['dtype'])

        # Reproject the raster data to the new projection and store in reprojected_data
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

    return reprojected_data, metadata
#%% Projection for mapping all MN maps
def init_lambert_proj():
    # Lambert Conformal Conic projection 
    lambert_proj = ccrs.LambertConformal(
        central_longitude = -94.37,
        central_latitude = 46.03,
        standard_parallels = (33,45)
        )
    return lambert_proj

#%% load Minnesota outline
# Minnesota outline
def load_minnesota_reproj(lambert_proj):
    url = "https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip"
    usa = gpd.read_file(url)
    minnesota = usa[usa['NAME'] == 'Minnesota']
    minnesota = minnesota.to_crs(lambert_proj.proj4_init)
    return minnesota
#%% create xarray dataarray from raster transform

def create_dataarray(data, metadata, name):
    # Extract metadata information
    transform = metadata['transform']
    width = metadata['width']
    height = metadata['height']
    crs = metadata['crs']
    
    # Calculate the x and y coordinates using the affine transform
    x_coords = np.arange(width) * transform[0] + transform[2]
    y_coords = np.arange(height) * transform[4] + transform[5]
    
    source = name
    scenario = name[0:name.find('_')]
    if scenario != 'historical':
        period = name[name.find('_20')+1:name.find('_20')+10]
    else:
        period = name[name.find('_19')+1:name.find('_19')+10]
    RI = int(name[name.find(period)+len(period)+1:name.find('yr')])
    D = int(name[name.find('da')-2:name.find('da')])
    # Create DataArray
    dataarray = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "x": ("x", x_coords),
            "y": ("y", y_coords),
            "scenario": scenario,
            "period": period,
            "RI":RI,
            "D":D,
            "source":source

        },
        attrs={"crs": str(crs),
               "transform": transform,
               "name": name
                }
    )
    
    return dataarray
