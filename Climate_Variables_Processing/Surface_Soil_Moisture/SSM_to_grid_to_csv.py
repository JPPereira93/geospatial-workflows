import os
import pandas as pd
import numpy as np
from osgeo import gdal
import shutil
import geopandas as gpd
import argparse

#------------------------------------------------------
# Organize Surface Soil Moisture folders and extract the data into two final csvs, one with merged dataframes and the other using spatial join operation
#
#@author: Joao Pereira  
#------------------------------------------------------


# extract files from zip archives
def extract_files(input_dir):
    for subdir in os.listdir(input_dir):
        if subdir.startswith('SSM1km_'):
            for file in os.listdir(os.path.join(input_dir, subdir)):
                if file.startswith('c_gls_SSM1km_') and file.endswith('.zip'):
                    zip_file_path = os.path.join(input_dir, subdir, file)
                    extract_dir = os.path.join(input_dir, subdir)
                    tiff_file_exists = any(f.endswith('.tiff') for f in os.listdir(extract_dir))
                    if not tiff_file_exists:
                        shutil.unpack_archive(zip_file_path, extract_dir=extract_dir)

# rename folders based on date and filter out invalid files
def rename_folders_and_filter(input_dir):
    for folder in os.listdir(input_dir):
        if folder.startswith('SSM1km'):
            # get the date from the folder name
            date = folder.split('_')[1][:8]
            # build the new folder name with the shortened date
            new_folder_name = f'SSM1km_{date}_CEURO_S1CSAR_V1.1.1'
            # build the paths for the old and new folders
            old_folder_path = os.path.join(input_dir, folder)
            new_folder_path = os.path.join(input_dir, new_folder_name)
            # rename the folder
            try:
                os.rename(old_folder_path, new_folder_path)
            except FileNotFoundError:
                print(f"Error: Folder {old_folder_path} not found.")

    # copy the .tiff file inside the subdirectory, rename it, and place it in the path
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".tiff"):
                if "c_gls_SSM1km" in file:
                    src_path = os.path.join(root, file)
                    date = file.split("_")[3][:8]
                    dst_filename = "SSM_1km_" + date + ".tiff"
                    dst_path = os.path.join(input_dir, dst_filename)
                    shutil.copy(src_path, dst_path)

    # loop over the files in the directory
    for file_name in os.listdir(input_dir):
        # get the full file path
        file_path = os.path.join(input_dir, file_name)
        # check if the file is a TIFF file and has a size less than or equal to 13 KB = NO DATA
        if file_name.endswith(".tiff") and os.path.getsize(file_path) <= 13*1024:
            # Delete the file
            os.remove(file_path)

# process TIFF files and convert them to CSV
def process_tiff_to_csv(input_dir):
    tiff_files = [f for f in os.listdir(input_dir) if f.startswith("SSM_1km") and f.endswith(".tiff")]
    
    for tiff_file in tiff_files:
        # load the GeoTIFF
        ds = gdal.Open(os.path.join(input_dir, tiff_file))

        # get the data from the GeoTIFF as a NumPy array
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()

        # get the spatial information from the GeoTIFF
        transform = ds.GetGeoTransform()
        x_origin = transform[0]
        y_origin = transform[3]
        pixel_width = transform[1]
        pixel_height = transform[5]

        # create an empty list to store the values
        values = []

        # loop over the data array
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                # Calculate the x and y coordinates
                x_coord = x_origin + x * pixel_width
                y_coord = y_origin + y * pixel_height
                value = data[y, x]
                values.append([x_coord, y_coord, value])

        # save the values as a CSV file
        csv_file = tiff_file.replace(".tiff", ".csv")
        np.savetxt(os.path.join(input_dir, csv_file), values, delimiter=",")

        # close the dataset
        ds = None

# add date column, round decimal values, and divide by 2
def process_csv_files(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            # read the .csv file into a dataframe
            df = pd.read_csv(os.path.join(input_dir, filename))
            df = pd.DataFrame(df.values, columns=["X", "Y", "Value (percentage)"])

            # write the dataframe back to the .csv file
            df_3_colunas = df.to_csv(os.path.join(input_dir, filename), index=False)

           # replace values above 200 with "NoData" in the Value (mm) column
            df.loc[df['Value (percentage)'] > 200, 'Value (percentage)'] = "NaN"

            # write the filtered dataframe back to the .csv file
            df.to_csv(os.path.join(input_dir, filename), index=False)

            # add date column
            df = pd.read_csv(os.path.join(input_dir, filename))
            date = filename.replace("SSM_1km_", "").replace(".csv", "")
            df["Date"] = date
            df.to_csv(os.path.join(input_dir, filename), index=False)

            # Drop date column and replace it with the old Value (mm) header
            df = df.rename(columns={"Value (percentage)": df["Date"][0]})
            df = df.drop("Date", axis=1)

            # Round the decimal values of X, Y to 4
            df['X'] = df['X'].round(4)
            df['Y'] = df['Y'].round(4)

            # divide values in the third column by 2, excluding the header
            df.iloc[1:, 2] = df.iloc[1:, 2] / 2

            # write the updated dataframe back to the .csv file
            df.to_csv(os.path.join(input_dir, filename), index=False)

# merge all CSV files and export as a single file
def merge_and_export_csv(input_dir, output_dir, output_file):
    df_list = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)
            # read the csv into a dataframe
            df = pd.read_csv(file_path)

            # takes each column starting from the third one in the dataframe df
            # converts name (which is stored as a float) to a string,
            # separates it into year, month, and day format (YYYY/MM/DD).
            # it then appends the modified dataframe to a list

            for i in range(2, len(df.columns)):
                old_column_name = df.columns[i]
                date_string = str(int(old_column_name))
                new_column_name = date_string[:4] + "/" + date_string[4:6] + "/" + date_string[6:]
                df = df.rename(columns={old_column_name: new_column_name})

            df_list.append(df)

    # merge all the dataframes into a single one and export it
    merged_df = pd.concat(df_list, axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df.to_csv(os.path.join(output_dir, output_file), index=False)

# perform spatial join and clip data
def spatial_join_and_clip(output_dir, output_file_2, shapefile):
    gdf = gpd.read_file(shapefile)
    gdf = gdf.to_crs("EPSG:4326")

    # Read in the merged data CSV and convert to a GeoDataFrame
    merged_df = pd.read_csv(os.path.join(output_dir, output_file))
    merged_gdf = gpd.GeoDataFrame(
        merged_df, geometry=gpd.points_from_xy(merged_df["X"], merged_df["Y"]), crs="EPSG:4326")

    # perform the select by location using the intersect operation
    selected_data = gpd.sjoin(merged_gdf, gdf, predicate="intersects")
    # define the columns to remove if they exist
    columns_to_remove = ["geometry", "index_right", "WKT", "fid"]
    # check and remove the columns
    selected_data = selected_data.drop(columns=[col for col in columns_to_remove if col in selected_data.columns])
    # save the selected data to a new clipped CSV file
    selected_data.to_csv(os.path.join(output_dir, output_file_2), index=False)

# argument parsing and function calls
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output directories and file name")

    parser.add_argument("-i", "--input_dir", required=True, help="The input directory containing the folder files")
    parser.add_argument("-od", "--output_dir", required=True, help="The output directory to save the merged file")
    parser.add_argument("-of1", "--output_file", required=True, help="The directory and name of the csv merged output file")
    parser.add_argument("-of2", "--output_file_2", required=True, help="The directory and name of the csv clipped merged output file")
    parser.add_argument("-s", "--shapefile", required=True, help="The directory and input shapefile to clip the merged file")
    parser.add_argument("-uz", "--unzip_files", type=bool, default=False, help="Flag to indicate if zip files need to be extracted (True/False)")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_file = args.output_file
    output_file_2 = args.output_file_2
    shapefile = args.shapefile
    unzip_files = args.unzip_files

    if unzip_files:
        extract_files(input_dir)
    rename_folders_and_filter(input_dir)
    process_tiff_to_csv(input_dir)
    process_csv_files(input_dir)
    merge_and_export_csv(input_dir, output_dir, output_file)
    spatial_join_and_clip(output_dir, output_file_2, shapefile)
