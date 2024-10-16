import os
import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

def clip_raster(raster, geoms, out_raster):
    with rasterio.open(raster) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    with rasterio.open(out_raster, "w", **out_meta) as dest:
        dest.write(out_image)

def raster_to_points(clipped_raster, date):
    with rasterio.open(clipped_raster) as src:
        # Read the raster data
        array = src.read(1)
        print("Array Shape:", array.shape)  # Debugging: Check array shape
        # Get the coordinates of each pixel
        x = np.linspace(src.bounds.left, src.bounds.right, src.width)
        y = np.linspace(src.bounds.top, src.bounds.bottom, src.height)
        lon, lat = np.meshgrid(x, y)
        values = array.flatten()

    # Round latitude and longitude to 2 decimal places
    lon = np.round(lon, 2)
    lat = np.round(lat, 2)

    data = pd.DataFrame({'lon': lon.flatten(), 'lat': lat.flatten(), date: values})
    data = data.dropna()  # Remove rows with NaN values

    print("Data Shape:", data.shape)  # Debugging: Check DataFrame shape
    print("Number of NaN Values:", data.isna().sum().sum())  # Debugging: Check NaN values

    # Print some sample data points for inspection
    print("Sample Data Points:")
    print(data.head())

    return data

def main(args):
    input_folder = args.io_directory
    shapefile = args.shapefile
    output_folder = args.output_directory
    prefix = args.prefix
    suffix = 'air_max' if args.MAX else 'air_min'
    gdf = gpd.read_file(shapefile)
    geoms = [geom for geom in gdf.geometry]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    dfs = []  # List to store DataFrames for each raster
    for root, dirs, files in os.walk(input_folder):
        files.sort()  # Sort files to ensure dates are in ascending order
        for file in files:
            if file.endswith('.tif'):
                date_str = file.split('_')[-1].split('.tif')[0]
                year, month = date_str.split('-')
                date = year + month + '01'
                raster = os.path.join(root, file)
                clipped_raster = os.path.join(output_folder, f'clipped_{file}')
                clip_raster(raster, geoms, clipped_raster)
                df = raster_to_points(clipped_raster, date)
                dfs.append(df)
                #os.remove(clipped_raster)  # Remove clipped raster after processing
    
    # Merge all DataFrames
    if dfs:
        merged_df = dfs[0]  # Initialize merged_df with the first DataFrame
        for df in dfs[1:]:  # Merge remaining DataFrames one by one
            merged_df = pd.merge(merged_df, df, on=['lon', 'lat'])

        # Drop rows with NaN values
        merged_df = merged_df.dropna()

        # Save the merged DataFrame to a CSV file
        out_csv = os.path.join(output_folder, f'{prefix}_{suffix}.csv')
        merged_df.to_csv(out_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clip raster and convert to point grid')
    parser.add_argument('-io', '--io_directory', type=str, required=True, help='Path to the input and output directory containing TIFF files')
    parser.add_argument('-s', '--shapefile', type=str, required=True, help='Path to the shapefile or geojson used for clipping')
    parser.add_argument('-o', '--output_directory', type=str, required=True, help='Path to the output directory where results will be saved')
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix to add to the output CSV file names')
    parser.add_argument('-MAX', action='store_true', help='Add "_air_max" suffix to the output CSV file names')
    parser.add_argument('-MIN', action='store_true', help='Add "_air_min" suffix to the output CSV file names')
    args = parser.parse_args()
    main(args)
