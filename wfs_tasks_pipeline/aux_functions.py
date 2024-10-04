import rasterio
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import rioxarray
import geopandas as gpd
import xarray as xr

def stack_rasters(tiff_files, output_tiff, aoi_shapefile=None, chunk_size=None, operation=None):
    """
    Clips, stacks a list of raster files based on an AOI shapefile, calculates the sum, median, or mean of the stack,
    and saves the resulting raster to a file. Processes data in chunks to reduce memory usage.

    :param tiff_files: List of paths to the input TIFF files
    :param aoi_shapefile: Path to the AOI shapefile (optional, default is None)
    :param output_tiff: Path for the output raster file
    :param operation: Operation to perform on the stack ('mean')
    :param chunk_size: Size of chunks for processing (e.g., (500, 500))
    :return: None
    """
    if aoi_shapefile:
        aoi = gpd.read_file(aoi_shapefile)
        polygon_geometry = [aoi.geometry.iloc[0]]
    else:
        polygon_geometry = None

    raster_arrays = []
    no_data_values = []

    for tiff in tiff_files:
        raster = rioxarray.open_rasterio(tiff, chunks=chunk_size)
        print(f"Processing multiple raster files {tiff}")
        no_data = raster.rio.nodata
        no_data_values.append(no_data)

        if polygon_geometry:
            clipped_raster = raster.rio.clip(polygon_geometry, aoi.crs, drop=True, invert=False)
            # Replace NoData values with np.nan
            clipped_raster = clipped_raster.where(clipped_raster != no_data, other=np.nan)
            raster_arrays.append(clipped_raster)
        else:
            raster_arrays.append(raster.where(raster != no_data, other=np.nan))

    stacked_rasters = xr.concat(raster_arrays, dim='band')

    if operation == 'mean':
        # Exclude NoData pixels from the mean calculation
        result_raster = stacked_rasters.where(~np.isnan(stacked_rasters)).mean(dim='band')
    else:
        raise ValueError("Invalid operation. Choose 'mean'")

    # Set the CRS of the result raster to match the vector CRS if AOI is provided
    if aoi_shapefile:
        result_raster.rio.write_crs(aoi.crs, inplace=True)

    # Set the NoData value for the result raster to the NoData of the first raster
    result_no_data_value = no_data_values[0]
    result_raster.rio.write_nodata(result_no_data_value, inplace=True)

    # Replace remaining NaN values with NoData in the result
    result_raster = result_raster.where(~np.isnan(result_raster), other=result_no_data_value)
    result_raster.rio.to_raster(output_tiff)
    print(f"Stacked {operation} Raster saved at {output_tiff}")


def process_kmeans(input_raster, n_clusters=5):
    """Processes a single raster file and applies K-means clustering."""
    
    with rasterio.open(input_raster) as src:
        raster_data = src.read(1)  # Read the first band (for single-band raster)
        raster_meta = src.meta     # Get metadata (used for saving output later)
        nodata_value = src.nodata

    # Step 2: Prepare the data (flatten the 2D raster into a 1D array)
    raster_flat = raster_data.flatten()  # Convert 2D array to 1D
    raster_valid = raster_flat[raster_flat != nodata_value]  # Remove NoData values

    if np.any(np.isnan(raster_valid)):
        print("Warning: NaN values detected in raster_valid. They will be removed.")
    
    # Step 3: Apply K-means clustering
    raster_valid = raster_valid.reshape(-1, 1)  # Reshape to 2D (required by KMeans)

    # Only fit KMeans if there are valid data points
    if raster_valid.shape[0] == 0:
        raise ValueError("No valid data points found for clustering.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(raster_valid)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Step 4: Reassign cluster labels to the original raster shape
    clustered_raster = np.full_like(raster_flat, fill_value=nodata_value)
    clustered_raster[raster_flat != nodata_value] = cluster_labels + 1  # Add 1 for non-zero labels
    clustered_raster = clustered_raster.reshape(raster_data.shape)

    return clustered_raster, raster_meta

def relabel_clusters(clustered_raster, raster_flat, nodata_value):
    """Relabels the clusters based on their minimum values."""
    
    # Calculate the range of values for each cluster
    cluster_ranges = defaultdict(list)
    cluster_labels_flat = clustered_raster.flatten()

    for pixel_value, cluster_label in zip(raster_flat, cluster_labels_flat):
        if pixel_value != nodata_value:  # Ignore NoData values
            cluster_ranges[cluster_label].append(pixel_value)

    cluster_info = []
    for cluster, values in cluster_ranges.items():
        min_val = np.min(values)
        max_val = np.max(values)
        count = len(values)
        cluster_info.append((cluster, min_val, max_val, count))

    # Sort clusters by minimum value
    for i in range(len(cluster_info)):
        for j in range(len(cluster_info) - 1 - i):
            if cluster_info[j][1] > cluster_info[j + 1][1]:
                cluster_info[j], cluster_info[j + 1] = cluster_info[j + 1], cluster_info[j]

    # Create a new mapping based on the sorted order
    new_labels = {cluster_info[i][0]: i + 1 for i in range(len(cluster_info))}

    # Create a copy of the clustered raster for relabeling
    relabelled_raster = np.copy(clustered_raster)

    # Apply the new labels
    for old_label, new_label in new_labels.items():
        relabelled_raster[clustered_raster == old_label] = new_label

    return relabelled_raster

def process_directory(input_directory, output_directory, n_clusters=5, process_single_file=False):
    """
    Processes all TIFF files in a directory or a single file if specified.

    Args:
        input_directory (str): Path to the input directory or file.
        output_directory (str): Path to the output directory.
        n_clusters (int): Number of clusters for processing.
        process_single_file (bool): If True, process only a single file (input_directory as a file path).
    """
        
    os.makedirs(output_directory, exist_ok=True)

    if process_single_file:
        if os.path.isfile(input_directory):
            print(f"Processing single file: {input_directory}")
            clustered_raster, raster_meta = process_kmeans(input_directory, n_clusters)

            # Relabel the clusters
            relabelled_raster = relabel_clusters(clustered_raster, clustered_raster.flatten(), raster_meta['nodata'])

            new_output_raster = os.path.join(output_directory, os.path.basename(input_directory).replace('.tiff', '_relabelled.tiff'))

            # Use rioxarray to save the raster
            xr.DataArray(relabelled_raster, dims=("y", "x")) \
                .rio.write_crs(raster_meta['crs'], inplace=True) \
                .rio.write_transform(raster_meta['transform'], inplace=True) \
                .rio.write_nodata(raster_meta['nodata'], inplace=True) \
                .rio.to_raster(new_output_raster)

            print(f"Relabeling output saved at {new_output_raster}")
        else:
            print(f"Error: {input_directory} is not a valid file.")
    else:
        for filename in os.listdir(input_directory):
            if filename.endswith('.tiff') or filename.endswith('.tif'):
                input_raster = os.path.join(input_directory, filename)
                print(f"Processing {input_raster}")

                # Process the raster
                clustered_raster, raster_meta = process_kmeans(input_raster, n_clusters)

                # Relabel the clusters
                relabelled_raster = relabel_clusters(clustered_raster, clustered_raster.flatten(), raster_meta['nodata'])

                new_output_raster = os.path.join(output_directory, filename.replace('.tiff', '_relabeled.tiff'))

                xr.DataArray(relabelled_raster, dims=("y", "x")) \
                    .rio.write_crs(raster_meta['crs'], inplace=True) \
                    .rio.write_transform(raster_meta['transform'], inplace=True) \
                    .rio.write_nodata(raster_meta['nodata'], inplace=True) \
                    .rio.to_raster(new_output_raster)

                print(f"Relabeling output saved at {new_output_raster}")


def convert_raster_to_integers(input_tiff, output_tiff):
    """
    Converts raster values to integers and saves the output, preserving native NoData values.

    :param input_tiff: Path for the input raster file
    :param output_tiff: Path for the output raster file
    :return: None
    """
    raster = rioxarray.open_rasterio(input_tiff)

    # Get the native NoData value
    nodata_value = raster.rio.nodata

    float_raster = raster.round()

    # Preserve native NoData values
    int_raster = float_raster.where(float_raster != nodata_value, other=np.nan)

    # Convert to int32, ensuring NoData remains as np.nan
    int_raster = int_raster.where(~np.isnan(int_raster), other=np.nan).astype(np.float32)

    # Write the native NoData value to the raster metadata
    int_raster.rio.write_nodata(nodata_value, inplace=True)

    int_raster.rio.to_raster(output_tiff)

    print(f"Raster with integer dataype saved at {output_tiff}")