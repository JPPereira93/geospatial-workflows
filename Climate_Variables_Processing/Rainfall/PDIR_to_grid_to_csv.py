import os
import pandas as pd
import numpy as np
from osgeo import gdal
import geopandas as gpd
from rasterio.mask import mask
import argparse
import rasterio 
from rasterio.warp import calculate_default_transform, reproject, Resampling

#------------------------------------------------------
# Organize Precipitation data (PDIR-NOW) folders and extract the data into two final csvs, one with merged dataframes and the other using spatial join operation
#
#@author: Joao Pereira  
#------------------------------------------------------


# create the argparse object
parser = argparse.ArgumentParser(description="Process precipitation data")

# add arguments
parser.add_argument("-i", "--input_dir", required=True, help="The input directory containing the folder files")
parser.add_argument("-od", "--output_dir", required=True, help="The output directory to save the merged file")
parser.add_argument("-of1", "--output_file", required=True, help="The directory and name of the csv merged output file")
parser.add_argument("-of2", "--output_file_2", required=True, help="The directory and name of the csv clipped merged output file")
parser.add_argument("-s", "--shapefile", required=True, help="The directory and input shapefile to clip the merged file") #Make sure that buffer is around 4 or 5km otherwise it can give bad results

# parse the arguments
args = parser.parse_args()

# set the directory where the folders are located
input_dir = args.input_dir
output_dir = args.output_dir
output_file = args.output_file
output_file_2= args.output_file_2
shapefile = args.shapefile

def read_shapefile(shapefile):
    gdf = gpd.read_file(shapefile)
    gdf = gdf.to_crs("EPSG:4326")
    return gdf

gdf = read_shapefile(shapefile)

def reproject_tiff(input_tiff_path, output_tiff_path, dst_crs):
    with rasterio.open(input_tiff_path) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tiff_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

def clip_rasters(input_dir, shapefile):
    tiff_files = [f for f in os.listdir(input_dir) if f.startswith("PDIR_1d") and f.endswith(".tif")]

    for tiff_file in tiff_files:
        input_tiff_path = os.path.join(input_dir, tiff_file)
        output_tiff_path = os.path.splitext(input_tiff_path)[0] + '_clipped.tif'

        with rasterio.open(input_tiff_path) as src:
            gdf = read_shapefile(shapefile)
            shapes = gdf.geometry

            # Clip the raster using the shapefile
            clipped_data, transform = rasterio.mask.mask(src, shapes, crop=True)

            # Update metadata for the clipped raster
            profile = src.profile
            profile.update({
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2],
                'transform': transform
            })

            # Write the clipped raster to a new file
            with rasterio.open(output_tiff_path, 'w', **profile) as dst:
                dst.write(clipped_data)

        # Replace the original TIFF file with the clipped version
        os.replace(output_tiff_path, input_tiff_path)

clip_rasters(input_dir, shapefile)

clip_rasters(input_dir, shapefile)

def raster_to_csv(input_dir):
    tiff_files = [f for f in os.listdir(input_dir) if f.startswith("PDIR_1d") and f.endswith(".tif")]

    for tiff_file in tiff_files:
        ds = gdal.Open(os.path.join(input_dir, tiff_file))
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        transform = ds.GetGeoTransform()
        x_origin = transform[0]
        y_origin = transform[3]
        pixel_width = transform[1]
        pixel_height = transform[5]

        values = []

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                x_coord = x_origin + x * pixel_width
                y_coord = y_origin + y * pixel_height
                value = data[y, x]
                values.append([x_coord, y_coord, value])

        csv_file = tiff_file.replace(".tif", ".csv")
        np.savetxt(os.path.join(input_dir, csv_file), values, delimiter=",")

raster_to_csv(input_dir)

def process_csv_files(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_dir, filename))
            df = pd.DataFrame(df.values, columns=["X", "Y", "Value (mm)"])
            df = df.to_csv(os.path.join(input_dir, filename), index=False)

process_csv_files(input_dir)

def add_date_to_csv_files(input_dir):
    filenames = [f for f in os.listdir(input_dir) if f.startswith("PDIR_1d") and f.endswith(".csv")]
    for filename in filenames:
        df = pd.read_csv(os.path.join(input_dir, filename))
        date = filename.replace("PDIR_1d", "").replace(".csv", "")
        df["Date"] = date
        df.to_csv(os.path.join(input_dir, filename), index=False)

add_date_to_csv_files(input_dir)

def process_and_round_values(input_dir):
    filenames = [f for f in os.listdir(input_dir) if f.startswith("PDIR_1d") and f.endswith(".csv")]
    for filename in filenames:
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path)
            df = df.rename(columns={"Value (mm)": df["Date"][0]})
            df = df.drop("Date", axis=1)
            df['X'] = df['X'].round(4)
            df['Y'] = df['Y'].round(4)
            df.to_csv(file_path, index=False)

process_and_round_values(input_dir)

# loop through all the .csv files
def process_and_merge_dataframes(input_dir, output_dir, output_file):
    df_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path)
            
            for i in range(2, len(df.columns)):
                old_column_name = df.columns[i]
                date_string = str(int(old_column_name))
                new_column_name = date_string[:4] + "/" + date_string[4:6] + "/" + date_string[6:]
                df = df.rename(columns={old_column_name: new_column_name})
                
            df_list.append(df)
            
    merged_df = pd.concat(df_list, axis=1)
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    merged_df.loc[:, merged_df.columns[2:]] = merged_df.loc[:, merged_df.columns[2:]].replace(-99.0, 0)
    merged_df.to_csv(os.path.join(output_dir, output_file), index=False)
    return merged_df

merged_df = process_and_merge_dataframes(input_dir, output_dir, output_file)

def save_intersected_data(merged_df, gdf, output_dir, output_file_2):
    merged_gdf = gpd.GeoDataFrame(
        merged_df, geometry=gpd.points_from_xy(merged_df["X"], merged_df["Y"]), crs="EPSG:4326")
    selected_data = gpd.sjoin(merged_gdf, gdf, predicate="intersects")

    # check if ["geometry", "index_right", "PK", "id"] are present and remove them
    cols_to_drop = []
    for col in ["geometry", "index_right", "PK", "id"]:
        if col in selected_data.columns:
            cols_to_drop.append(col)
    selected_data.drop(cols_to_drop, axis=1, inplace=True)
    
    selected_data.to_csv(os.path.join(output_dir, output_file_2), index=False)

save_intersected_data(merged_df, gdf, output_dir, output_file_2)

def main():
    gdf = read_shapefile(shapefile)
    clip_rasters(input_dir, shapefile)
    raster_to_csv(input_dir)
    process_csv_files(input_dir)
    add_date_to_csv_files(input_dir)
    process_and_round_values(input_dir)
    merged_df = process_and_merge_dataframes(input_dir, output_dir, output_file)
    save_intersected_data(merged_df, gdf, output_dir, output_file_2)

if __name__ == "__main__":
    main()
