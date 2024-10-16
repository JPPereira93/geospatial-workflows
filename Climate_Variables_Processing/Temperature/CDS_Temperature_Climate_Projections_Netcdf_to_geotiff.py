import argparse
import netCDF4 as nc
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from rasterio.mask import mask
import numpy as np
import os

#------------------------------------------------------
# Convert netcdf to geotiff from temperature climate projections. Each year is output as a single geotiff file. 
#Apply this script to either the Yearly Max netcdf files and Yearly min from the ''Temperature statistics for Europe derived from climate projections (Cordex Regional Climate Model)''
#
#@author: Joao Pereira  
#------------------------------------------------------


def convert_netcdf_to_geotiff(input_file, output_folder, rcp_scenario, variable_type, clip_file):
    # Open the NetCDF file
    nc_file = nc.Dataset(input_file, 'r')

    # Get the time variable and extract the years
    time_variable = nc_file.variables['time']
    time_values = time_variable[:]
    calendar = time_variable.calendar

    # Convert time values to datetime objects
    cf_time_values = nc.num2date(time_values, time_variable.units, calendar=calendar)
    years = [t.year for t in cf_time_values]

    # Determine the variable name based on the input flag
    if variable_type == "min":
        variable_name = "mean_Tmin_Yearly"
    elif variable_type == "max":
        variable_name = "mean_Tmax_Yearly"
    else:
        raise ValueError("Invalid variable type. Choose 'min' or 'max'.")

    # Get x and y variables
    x_variable = nc_file.variables['lon']
    y_variable = nc_file.variables['lat']
    x_values = x_variable[:]
    y_values = y_variable[:]

    # Reverse the order of y_values
    y_values = y_values[::-1]

    # Load the clipping shapefile
    clip_data = gpd.read_file(clip_file)

    # Set the CRS of the shapefile to EPSG 4326
    clip_data = clip_data.to_crs(epsg=4326)

    # Iterate over the years and convert each band to GeoTIFF
    for i, year in enumerate(years):
        band_index = i + 1
        band_data = nc_file.variables[variable_name][i, :, :]

        # Determine the RCP scenario for the output file name
        rcp_str = "RCP_4.5" if rcp_scenario == "4.5" else "RCP_8.5"

        # Create the output file path
        output_file = os.path.join(output_folder, f'Climate-projection-{rcp_str}_Yearly_{variable_type}_{year}.tif')

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

        # Clip the GeoTIFF file with the clipping shapefile
        for feature in clip_data['geometry']:
            clipped_output_file = os.path.join(output_folder,
                                               f'Climate-projection-{rcp_str}_Yearly_{variable_type}_{year}_clipped.tif')

            # Open the output GeoTIFF file
            with rasterio.open(output_file) as src:
                # Clip the raster data using the feature geometry
                clipped_data, clipped_transform = mask(src, [feature], crop=True)

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

    # Add variable type flag
    parser.add_argument('--variable', choices=['min', 'max'], default='max',
                        help='Variable type (default: max)')

    # Add clipping shapefile argument
    parser.add_argument('--clip', dest='clip_file',
                        help='Path to the clipping shapefile')

    args = parser.parse_args()

    # Call the function to convert the NetCDF file to GeoTIFF
    convert_netcdf_to_geotiff(args.input_file, args.output_folder, args.rcp, args.variable, args.clip_file)
