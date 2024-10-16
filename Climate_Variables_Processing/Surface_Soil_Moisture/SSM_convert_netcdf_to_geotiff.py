import os
import glob
import rasterio 
import xarray as xr
import numpy as np
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import rasterio.mask
import argparse

#------------------------------------------------------
# Convert NETCDF files to GeoTIFF (it was needed before knowing how to download precipitation data as a geotiff)
#
#@author: Joao Pereira  
#------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert NetCDF files to a clipped GeoTIFF.')
    parser.add_argument("-i", "--input_dir", required=True, help="The directory containing the NetCDF files.")
    parser.add_argument("-o", "--output_dir", required=True, help="The directory to save the clipped GeoTIFF files.")
    parser.add_argument("-s", "--shapefile_path", required=True, help="The path to the shapefile to clip the GeoTIFF files to")
    args = parser.parse_args()
    return args

def read_shapefile(path):
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    return gdf

def convert_netcdf_to_geotiff(input_dir, output_dir, shapes):
    # loop through all NetCDF files in input directory
    for nc_file in glob.glob(os.path.join(input_dir, '*.nc')):
        # open NetCDF file as xarray dataset
        ncfile = xr.open_dataset(nc_file)
        sm = ncfile['sm']
        sm = sm.rio.set_spatial_dims('lon', 'lat')
        sm.rio.set_crs("epsg:4326")
        # set minimum value to 1
        sm = sm.where(sm >= 0, 0)

        # create output filename based on input filename
        output_filename = os.path.join(output_dir, os.path.basename(nc_file).replace('.nc', '.tif'))
        sm.rio.to_raster(output_filename)

        # clip GeoTIFF to shapefile extent
        with rasterio.open(output_filename) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes.geometry, crop=True)
            out_meta = src.meta.copy()

        # mask no data values
        out_image = np.ma.masked_where(out_image == 0, out_image)

        # update metadata to match clipped image
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "nodata":0})

        # write clipped GeoTIFF to file
        clipped_filename = os.path.join(output_dir, os.path.basename(nc_file).replace('.nc', '_clipped.tif'))
        with rasterio.open(clipped_filename, "w", **out_meta) as dest:
            dest.write(out_image)

if __name__ == '__main__':
    args = parse_arguments()
    shapes = read_shapefile(args.shapefile_path)
    # Convert NetCDF files to clipped GeoTIFF
    convert_netcdf_to_geotiff(args.input_dir, args.output_dir, shapes)
