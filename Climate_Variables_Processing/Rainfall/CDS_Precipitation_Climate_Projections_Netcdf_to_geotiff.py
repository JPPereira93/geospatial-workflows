import argparse
import netCDF4 as nc
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
import numpy as np
import os
import geopandas as gpd

#------------------------------------------------------
# Convert netcdf to geotiff from precipitation climate projections. Each single geotiff file is output as the monthly mean clipped to a input shapefile/geojson
#Apply this script to a single netcdf mean projection min from the "Downscaled bioclimatic indicators for selected regions from 1950 to 2100 derived from climate projections (GFDL-ESM2M (NOAA, USA))""

#
#@author: Joao Pereira  
#------------------------------------------------------


def convert_netcdf_to_geotiff(input_file, output_folder, rcp_scenario, clip_file):
    # Calculate the scaling factor
    scale_factor = 3600 * 24 * 1000 * 30.4 # 3600 (seconds in an hour) × 24 (hours in a day) × 30.4 (average number of days per month) × 1000 (to convert meters to millimeters). This makes a rainfall sum for the whole month
    
    # Open the NetCDF file
    nc_file = nc.Dataset(input_file, 'r')

    # Get the time variable and extract the dates
    time_variable = nc_file.variables['time']
    time_values = time_variable[:]
    units = time_variable.units
    calendar = time_variable.calendar

    # Convert time values to datetime objects
    cf_time_values = nc.num2date(time_values, units, calendar=calendar)

    # Determine the RCP scenario for the output file name
    rcp_str = "4.5" if rcp_scenario == "4.5" else "8.5"

    # Load the clipping shapefile
    clip_data = gpd.read_file(clip_file)

    # Set the CRS of the shapefile to EPSG 4326
    clip_data = clip_data.to_crs(epsg=4326)

    # Iterate over the dates and convert each band to GeoTIFF
    for i, date in enumerate(cf_time_values):
        year = date.year

        # Only process bands from the years 2020 to 2050
        if year < 2020 or year > 2050:
            continue

        band_index = i + 1
        band_data = nc_file.variables['precipitation_monthly-mean'][i, :, :]
        
        # Scale the band data
        band_data *= scale_factor

        # Format the date as YYYY_MM_DD
        date_str = date.strftime("%Y_%m_%d")

        # Create the output file path
        output_file = os.path.join(output_folder,
                                   f'Climate-projection-Precipitation-Monthly-Mean-{rcp_str}-{date_str}.tiff')

        # Get x and y values from the NetCDF file
        x_values = nc_file.variables['longitude'][:]
        y_values = nc_file.variables['latitude'][:]

        # Reverse the order of y_values
        y_values = y_values[::-1]

        # Define the geospatial information for the GeoTIFF
        transform = from_origin(x_values[0], y_values[-1], x_values[1] - x_values[0], y_values[1] - y_values[0])
        height, width = band_data.shape
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': band_data.dtype,
            'crs': 'EPSG:4326',
            'transform': transform,
            'nodata': -9999  # Set the nodata value
        }

        # Write the band data to the GeoTIFF file
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(band_data, 1)

        # Clip the GeoTIFF file
        clipped_output_file = os.path.join(output_folder,
                                           f'Climate-projection-Precipitation-Monthly-Mean-{rcp_str}-{date_str}_clipped.tiff')

        with rasterio.open(output_file) as src:
            # Clip the raster data using the bounding box of the clipping shapefile
            clipped_data, clipped_transform = mask(src, clip_data.geometry, crop=True)

            # Update the geospatial information for the clipped GeoTIFF
            clipped_profile = src.profile
            clipped_profile.update({
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2],
                'transform': clipped_transform
            })

            # Write the clipped data to a new GeoTIFF file
            with rasterio.open(clipped_output_file, 'w', **clipped_profile) as clipped_dst:
                clipped_dst.write(clipped_data)

                # Access the CRS of the clipped GeoTIFF
                clipped_crs = clipped_dst.crs
                print(f"CRS of the clipped GeoTIFF: {clipped_crs}")

    # Close the NetCDF file
    nc_file.close()

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Convert NetCDF file to GeoTIFF')
    # Add input file argument
    parser.add_argument('-i', '--input', dest='input_file', required=True,
                        help='Path to the input NetCDF file')
    # Add output folder argument
    parser.add_argument('-o', '--output', dest='output_folder', required=True,
                        help='Path to the output folder')
    # Add RCP scenario flag
    parser.add_argument('--rcp', choices=['4.5', '8.5'], default='4.5',
                        help='RCP scenario (default: 4.5)')
    # Add clipping shapefile argument
    parser.add_argument('--clip', dest='clip_file',
                        help='Path to the clipping shapefile')
    args = parser.parse_args()
    # Call the function to convert the NetCDF file to GeoTIFF
    convert_netcdf_to_geotiff(args.input_file, args.output_folder, args.rcp, args.clip_file)
