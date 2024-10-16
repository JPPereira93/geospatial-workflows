import os
from osgeo import gdal
import geopandas as gpd
import rasterio
from scipy.ndimage import distance_transform_edt
import numpy as np
from pysal.viz.mapclassify import NaturalBreaks
import pandas as pd
import subprocess
import shutil
import xarray as xr
import rioxarray 
from rasterio.crs import CRS 
import netCDF4 as nc
from rasterio.transform import from_origin
from rasterio.mask import mask
import os
import glob

def reproject_clip_resample_tiff(input_tiff, output_tiff=None, aoi_shapefile=None, target_srs=None, target_res_x=None, target_res_y=None, resampling_method=None, clip=False, clip_by_extent=False, no_data=None):
    """
    Reprojects, optionally clips, and resamples a TIFF file based on an AOI shapefile.

    :param input_tiff: Path to the input TIFF file
    :param output_tiff: Path for the output TIFF file
    :param aoi_shapefile: Optional path to the AOI shapefile or GeoDataFrame. Required if clip is True.
    :param target_srs: Optional target spatial reference system (ex: 'EPSG:32629')
    :param target_res_x: Optional target resolution in x (meters)
    :param target_res_y: Optional target resolution in y (meters)
    :param resampling_method: Optional resampling method (ex: 'bilinear')
    :param clip: Boolean to determine whether to clip the raster
    :param clip_by_extent: Boolean to determine whether to clip the raster by the extent of the AOI.
    :param no_data: Optional no data value to be set for the output TIFF
    """
    # Check if the output TIFF path is given, otherwise create a new one based on the input TIFF
    if not output_tiff:
        base, ext = os.path.splitext(input_tiff)
        output_tiff = f"{base}_new.tif"

    # Continue with your function implementation
    cmd_reproject = "gdalwarp"

    if target_srs:
        cmd_reproject += f" -t_srs {target_srs}"
    if target_res_x and target_res_y:
        cmd_reproject += f" -tr {target_res_x} {target_res_y}"
    if resampling_method:
        cmd_reproject += f" -r {resampling_method}"
    if clip:
        if aoi_shapefile:
            if clip_by_extent:
                # Extract the extent if it is a GeoDataFrame or load it from a shapefile
                if isinstance(aoi_shapefile, gpd.geodataframe.GeoDataFrame):
                    bounds = aoi_shapefile.total_bounds
                else:
                    aoi_gdf = gpd.read_file(aoi_shapefile)
                    bounds = aoi_gdf.total_bounds
                cmd_reproject += f' -te {bounds[0]} {bounds[1]} {bounds[2]} {bounds[3]}'
            else:
                cmd_reproject += f' -cutline \"{aoi_shapefile}\" -crop_to_cutline'
        else:
            raise ValueError("AOI shapefile must be provided if clip is True")
    if no_data is not None:
        cmd_reproject += f" -dstnodata {no_data}"
    
    cmd_reproject += f" \"{input_tiff}\" \"{output_tiff}\""
    
    os.system(cmd_reproject)
    
""" # Example usage
reproject_clip_resample_tiff(
    input_tiff="path/to/your/input_raster.tif", 
    output_tiff="path/to/your/output_raster.tif", 
    aoi_shapefile="path/to/your/aoi_shapefile.shp", 
    target_srs="EPSG:32629", 
    target_res_x=30, 
    target_res_y=30, 
    resampling_method="bilinear", 
    clip=True,
    clip_by_extent=True,
    no_data=-9999
) """


def idw_interpolation(input_geojson, output_raster, zfield, aoi_path):
    """
    Performs IDW interpolation on point data.

    :param input_geojson: Path to the input GeoJSON file with point data
    :param output_raster: Path for the output raster file
    :param zfield: Field name in the GeoJSON file to use for interpolation
    :param aoi_path: Path to the AOI GeoJSON file
    """
    aoi = gpd.read_file(aoi_path)
    aoi_bounds = aoi.total_bounds
    
    # Perform IDW interpolation
    gdal.Grid(output_raster, input_geojson, zfield=zfield, algorithm="invdist", outputBounds=aoi_bounds)
    
# Example usage
""" idw_interpolation(
    input_geojson="path/to/input.geojson",
    output_raster="path/to/output.tif",
    zfield="Cumulative-Precipitation",
    aoi_path="path/to/aoi.geojson"
) """


def convert_hgt_to_tiff(hgt_file, tiff_file):

    dataset = gdal.Open(hgt_file)

    # Check if the dataset was successfully opened
    if not dataset:
        print(f"Failed to open file {hgt_file}")
        return

    # Convert to TIFF
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(tiff_file, dataset)

    # Close the dataset
    dataset = None

    print(f"File converted and saved as {tiff_file}")
    

def calculate_slope(input_dem, output_slope):
    """
    Calculates Slope based on a Digital Elevation Model

    :param input_dem: Path to the input DEM Raster file.
    :param output_slope: Path for the output Slope Raster file.
    """
    
    cmd = f"gdaldem slope \"{input_dem}\" \"{output_slope}\" -compute_edges"
    os.system(cmd)

# Example usage
""" calculate_slope(
    input_dem ="path/to/input.tif",
    output_slope ="path/to/output.tif"
) """


def calculate_aspect(input_dem, output_aspect):
    """
    Calculates Aspect based on a Digital Elevation Model

    :param input_dem: Path to the input DEM Raster file
    :param output_aspect: Path for the output Aspect Raster file
    """
    
    cmd = f"gdaldem aspect \"{input_dem}\" \"{output_aspect}\" -compute_edges"
    os.system(cmd)

# Example usage
""" calculate_aspect(
    input_dem ="path/to/input.tif",
    output_aspect ="path/to/output.tif"
) """


def rasterize(target_resolution, input_geojson, output_tif, no_data_value, field_name=None):
    """
    Rasterizes a GeoJSON file using the specified field as the attribute to burn into the raster.
    Sets a mandatory no-data value for the output raster.

    :param target_resolution: Target resolution for the output raster.
    :param input_geojson: Path to the input GeoJSON file.
    :param output_tif: Path for the output TIFF file.
    :param no_data_value: The no-data value to set for the output raster.
    :param field_name: Name of the field in the GeoJSON to use for rasterization. If None, default behavior is applied.
    """
    # Initialize the base command with target resolution and output format
    cmd = f"gdal_rasterize -tr {target_resolution} {target_resolution} -of GTiff"
    
    # Add the attribute to burn into the raster if specified
    if field_name:
        cmd += f" -a {field_name}"
    
    # Include the mandatory no-data value
    cmd += f" -a_nodata {no_data_value}"
    
    # Complete the command with input and output file paths
    cmd += f" \"{input_geojson}\" \"{output_tif}\""
    
    # Execute the command
    os.system(cmd)
 
""" # Example usage
rasterize(
    target_resolution=30,
    input_geojson=r"E:\Spotlite_JPereira\E-REDES\Fire_Susceptibility_Mapping\FFSM\Fatores-Condicionantes\Idade_linhas\E_REDES_FFSM_AOI_trocos_caldas_idade_metric.geojson",
    output_tif=r"E:\Spotlite_JPereira\E-REDES\Fire_Susceptibility_Mapping\FFSM\Fatores-Condicionantes\Idade_linhas\E_REDES_FFSM_AOI_trocos_caldas_idade_metric_rasterized_30m_t.tif",
    no_data_value = -9999
    field_name="CURRENT_AGE"
) """

def distance_matrix(input_raster_path, output_raster_path, target_value=1):
    """
    Create a distance matrix from a raster by calculating the distance from each cell to the nearest target cell.

    :param input_raster_path: Path to the input GeoTIFF raster file.
    :param output_raster_path: Path for the output distance raster file.
    :param target_value: The target cell value for which to calculate distances (default is 1).
    """
    
    with rasterio.open(input_raster_path) as src:
        raster_data = src.read(1)
        meta = src.meta

        # Define your target cells
        target_cells = (raster_data == target_value)

        # Calculate the distance from each cell to the nearest target cell
        distance = distance_transform_edt(~target_cells)

        # Update metadata for the output raster
        meta.update(dtype=rasterio.float32)

        with rasterio.open(output_raster_path, 'w', **meta) as dst:
            dst.write(distance, 1)

"""     # Example usage
    create_distance_matrix(
    input_raster_path = "/path/to/input.tif", 
    output_raster_path = "/path/to/output.tif", 
    target_value=1
) """


def reclassify_raster(input_raster_path):
    base, ext = os.path.splitext(input_raster_path)
    output_raster_path = f"{base}_ebreaks_reclassified{ext}"
    
    with rasterio.open(input_raster_path) as src:
        # Read the first band
        raster_data = src.read(1).astype(float)  # Cast raster data to float

        # Identify the 'no data' value from the source
        nodata = src.nodata

        # Exclude 'no data' value and zeros to find the actual minimum value
        mask = np.ones_like(raster_data, dtype=bool)
        if nodata is not None:
            mask &= (raster_data != nodata)
        mask &= (raster_data > 0.0)
        
        valid_data = raster_data[mask]
        min_val = valid_data.min() if valid_data.size > 0 else np.nan

        # Find maximum values
        max_val = raster_data.max()

        # Calculate interval
        interval = (max_val - min_val) / 5

        # Define the classification ranges in reverse order
        classification_values = [max_val - i * interval for i in range(5)]

        # Reclassify the raster
        reclassified_raster = np.copy(raster_data)
        reclassified_raster[mask] = 6 - np.digitize(raster_data[mask], classification_values, right=True)

        # Retain the original 'no data' value
        if nodata is not None:
            reclassified_raster[~mask] = nodata

        # Adjust the profile for a single band and write the output
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=nodata)

        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(reclassified_raster.astype(rasterio.float32), 1)

    print(f"Reclassified raster written to: {output_raster_path}")


"""     # Example usage
    reclassify_raster(
    input_raster_path = "/path/to/input.tif"
) """

def reclassify_raster_nbreaks(input_raster_path):

    base, ext = os.path.splitext(input_raster_path)
    output_raster_path = f"{base}_nbreaks_reclassified{ext}"
    
    with rasterio.open(input_raster_path) as src:
        # Read the first band
        raster_data = src.read(1).astype(float)  # Cast raster data to float

        # Identify the 'no data' value from the source
        nodata = src.nodata

        # Exclude 'no data' value and zeros to find the actual minimum value
        mask = np.ones_like(raster_data, dtype=bool)
        if nodata is not None:
            mask &= (raster_data != nodata)
        mask &= (raster_data > 0.0)
        
        valid_data = raster_data[mask]
        min_val = valid_data.min() if valid_data.size > 0 else np.nan

        # Find maximum values
        max_val = raster_data.max()

        # Calculate the number of classes (adjust this as needed)
        num_classes = 5

        # Use Jenks Natural Breaks classification
        breaks = NaturalBreaks(valid_data, k=num_classes)

        # Reclassify the raster based on breaks
        reclassified_raster = np.copy(raster_data)
        reclassified_raster[mask] = breaks.yb + 1  # Add 1 to make classes start from 1

        # Set 'no data' and zero areas to NaN
        reclassified_raster[~mask] = np.nan

        # Adjust the profile for a single band and write the output
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(reclassified_raster.astype(rasterio.float32), 1)

    print(f"Natural Breaks reclassified raster written to: {output_raster_path}")

"""     # Example usage
    reclassify_raster_nbreaks(
    input_raster_path = "/path/to/input.tif"
) """


def set_nodata_value(input_raster_path, nodata_value):
    """
    Set the NoData value of a raster and overwrite the input file.

    Parameters:
    input_raster_path (str): Path to the raster file.
    nodata_value (numeric): The value to be set as NoData.
    """
    try:
        # Open the input raster in update mode
        raster = gdal.Open(input_raster_path, gdal.GA_Update)

        if not raster:
            raise IOError("Could not open raster file.")

        # Set the NoData value for each band
        for i in range(1, raster.RasterCount + 1):
            band = raster.GetRasterBand(i)
            data_type = band.DataType
            # Ensure the data type supports negative values if nodata_value is negative
            if nodata_value < 0 and gdal.GetDataTypeName(data_type).startswith('UInt'):
                raise ValueError(f"The raster data type is {gdal.GetDataTypeName(data_type)}, which does not support negative NoData values.")
            band.SetNoDataValue(nodata_value)
            band.FlushCache()  # Ensure changes are written immediately

        print(f"NoData value set to {nodata_value} successfully in {input_raster_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the raster file to ensure changes are saved
        raster = None
        
"""     # Example usage
   set_nodata_value(
    input_raster_path = "/path/to/input.tif"
    nodata_value = -9999
) """

def normalize_raster(input_raster_path, output_normalized_raster_path):
    with rasterio.open(input_raster_path) as src:
        # Read the first band and cast raster data to float
        raster_data = src.read(1).astype(float)

        # Identify the 'no data' value from the source
        nodata = src.nodata

        # Create a mask to exclude 'no data' value only
        mask = raster_data != nodata if nodata is not None else np.ones_like(raster_data, dtype=bool)

        # Calculate the minimum and maximum values using the mask
        min_value = raster_data[mask].min() if np.any(mask) else np.nan
        max_value = raster_data[mask].max() if np.any(mask) else np.nan

        # Normalize the raster values to the range [0, 1]
        normalized_data = np.copy(raster_data)
        if max_value != min_value:  # Avoid division by zero
            normalized_data[mask] = (raster_data[mask] - min_value) / (max_value - min_value)
            # Ensure that the lowest and highest values are exactly 0.0 and 1.0
            normalized_data[mask][normalized_data[mask].argmin()] = 0.0
            normalized_data[mask][normalized_data[mask].argmax()] = 1.0
        else:
            # Handle case where all values are the same
            normalized_data[mask] = 0.0

        # Set 'no data' areas to original nodata value
        if nodata is not None:
            normalized_data[~mask] = nodata

        # Adjust the profile for a single band and write the output
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=nodata)

        with rasterio.open(output_normalized_raster_path, 'w', **profile) as dst:
            dst.write(normalized_data.astype(rasterio.float32), 1)

    # Print the input raster path and the calculated maximum value
    print(f"Input Raster: {input_raster_path}")
    print(f"Minimum Value: {min_value}")
    print(f"Maximum Value: {max_value}")

    # Print the normalized data statistics (excluding NoData values)
    print(f"Normalized Data Min (excluding NoData): {normalized_data[mask].min()}")
    print(f"Normalized Data Max (excluding NoData): {normalized_data[mask].max()}")

"""     # Example usage
    normalize_raster(
    input_raster_path = "/path/to/input.tif"
    output_normalized_raster_path = = "/path/to/output.tif"
) """

def normalize_raster_fixed_scale(input_raster_path, output_normalized_raster_path, fixed_min, fixed_max):
    # Use the provided fixed minimum and maximum values
    fixed_min_value = fixed_min
    fixed_max_value = fixed_max

    with rasterio.open(input_raster_path) as src:
        # Read the first band and cast raster data to float
        raster_data = src.read(1).astype(float)

        # Identify the 'no data' value from the source
        nodata = src.nodata

        # Create a mask to exclude 'no data' value only
        mask = raster_data != nodata if nodata is not None else np.ones_like(raster_data, dtype=bool)

        # Normalize the raster values to the range [0, 1] based on the fixed scale
        normalized_data = np.copy(raster_data)
        scale_range = fixed_max_value - fixed_min_value  # Calculate the range of the scale
        if scale_range != 0:  # Avoid division by zero
            normalized_data[mask] = (raster_data[mask] - fixed_min_value) / scale_range
        else:
            # Handle case where scale range is zero (unlikely in this scenario)
            normalized_data[mask] = 0.0

        # Set 'no data' areas to original nodata value
        if nodata is not None:
            normalized_data[~mask] = nodata

        # Adjust the profile for a single band and write the output
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=nodata)

        with rasterio.open(output_normalized_raster_path, 'w', **profile) as dst:
            dst.write(normalized_data.astype(rasterio.float32), 1)

    # Print the input raster path and the fixed scale values
    print(f"Input Raster: {input_raster_path}")
    print(f"Fixed Minimum Value: {fixed_min_value}")
    print(f"Fixed Maximum Value: {fixed_max_value}")

    # Print the normalized data statistics (excluding NoData values)
    print(f"Normalized Data Min (excluding NoData): {normalized_data[mask].min()}")
    print(f"Normalized Data Max (excluding NoData): {normalized_data[mask].max()}")

""" # Example usage
input_raster_path = "path_to_your_input_raster.tiff"
output_normalized_raster_path = "path_to_your_output_normalized_raster.tiff"
fixed_min = 1.0
fixed_max = 5.0
normalize_raster(input_raster_path, output_normalized_raster_path, fixed_min, fixed_max) """


def get_raster_values(input_raster_path, input_points):
    with rasterio.open(input_raster_path) as raster:
        values = [val[0] for _, row in input_points.iterrows()
                  for val in raster.sample([(row.geometry.x, row.geometry.y)])]
        return values

"""     # Example usage
    get_raster_values(
    input_raster_path = "/path/to/input.tif"
    input_points = "/path/to/input.geojson"
) """


def normalize_custom_ranking(ranking_data, max_rank, min_new_scale=1, max_new_scale=9):
    """
    Normalize a given ranking data dictionary to a specified scale, where higher original ranks correspond to higher normalized values.

    :param ranking_data: A dictionary with factors as keys and their ranks as values.
    :param max_rank: The maximum rank in the input ranking data.
    :param min_new_scale: The minimum value of the output scale (default 1).
    :param max_new_scale: The maximum value of the output scale (default 9).
    :return: A dictionary with factors and their normalized ranks.
    """
    return {key: min_new_scale + int((value - 1) / (max_rank - 1) * (max_new_scale - min_new_scale))
            for key, value in ranking_data.items()}


"""     # Example usage
ranking_data = {
    "Uso e Ocupação do Solo": 1,
    "NDVI": 2,
    "Humidade do Solo à Superfície": 3,
    "Temperatura Máxima (Média anual)": 4,
    "Precipitação Acumulada (Anual)": 5,
    "Velocidade do Vento": 6,
    "Distância a Corpos de Água": 7,
    "Orientação das Vertentes": 8,
    "Distância a Zonas Residenciais": 9,
    "Densidade Populacional": 10,
    "Idade das linhas": 11,
    "Elevação": 12,
    "Inclinação": 13,
    "Distância à estrada": 14

max_rank = 14

normalized_ranks_1 = normalize_custom_ranking(ranking_data, max_rank)


    normalize_custom_ranking(
    ranking_data,
    max_rank
)
   
    """
    
def create_ahp_matrix(normalized_ranks):
    factors = list(normalized_ranks.keys())
    n = len(factors)
    ahp_matrix = pd.DataFrame(index=factors, columns=factors, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                ahp_matrix.iloc[i, j] = 1.0  # Equal importance for the same factor
            else:
                ahp_matrix.iloc[i, j] = normalized_ranks[factors[i]] / normalized_ranks[factors[j]]

    return ahp_matrix


"""     # Example usage
    create_ahp_matrix(
    normalized_ranks_1
) """

def calculate_ahp_weights(ahp_matrix):
    # Calculate the sum of each column
    column_sums = ahp_matrix.sum()

    # Normalize each cell by the column sum and then calculate the row mean
    normalized_matrix = ahp_matrix.divide(column_sums, axis=1)
    weights = normalized_matrix.mean(axis=1)

    # Convert weights to percentage
    weights_percentage = (weights / weights.sum()) * 100

    return weights_percentage

"""     # Example usage
    calculate_ahp_weights(
    ahp_matrix
) """

def invert_raster_values(input_raster_path, output_raster_path):
    with rasterio.open(input_raster_path) as src:
        # Read the raster data into a NumPy array
        raster_data = src.read(1)

        # Calculate the maximum value of the raster
        max_value = np.max(raster_data)

        # Calculate the expression
        result = -1 * raster_data + max_value

        # Create a new TIFF file for the result
        with rasterio.open(output_raster_path, 'w', **src.profile) as dst:
            dst.write(result, 1)

"""     # Example usage
    invert_raster_values(
    input_raster_path = "/path/to/input.tif",
    output_raster_path ="/path/to/output.tif"
) """

def align_rasters(rasters, source_path, output_suffix):
    """
    Aligns list of rasters to have the same resolution and 
    cell size for pixel-based calculations. Saves aligned rasters in a new folder 'Aligned'.
    
    :param rasters: List of raster paths.
    :type rasters: List
    :param source_path: Path to the source directory of rasters.
    :type source_path: String
    :param output_suffix: The output aligned rasters files suffix with extension.
    :type output_suffix: String
    :return: True if the process runs and False if the data couldn't be read. 
    :rtype: Boolean
    """
    # Calculate the parent directory of source_path
    parent_dir = os.path.dirname(source_path.rstrip(os.sep))
    
    # Construct the path to the new "Aligned" folder
    aligned_dir = os.path.join(parent_dir, "Aligned")
    
    # Check if the "Aligned" folder exists, if not, create it
    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)
    
    command = ["gdalbuildvrt", "-te"]
    hDataset = gdal.Open(rasters[0], gdal.GA_ReadOnly)
    if hDataset is None:
        return False
    
    adfGeoTransform = hDataset.GetGeoTransform(can_return_null=True)
    if adfGeoTransform is None:
        return False
    
    for tif_file in rasters:
        base_filename = os.path.basename(tif_file)
        vrt_file = os.path.join(aligned_dir, base_filename.replace('.tif', '.vrt'))

        dfGeoXUL = adfGeoTransform[0]
        dfGeoYUL = adfGeoTransform[3]
        dfGeoXLR = adfGeoTransform[0] + adfGeoTransform[1] * hDataset.RasterXSize + adfGeoTransform[2] * hDataset.RasterYSize
        dfGeoYLR = adfGeoTransform[3] + adfGeoTransform[4] * hDataset.RasterXSize + adfGeoTransform[5] * hDataset.RasterYSize
        xres = str(abs(adfGeoTransform[1]))
        yres = str(abs(adfGeoTransform[5]))
        
        subprocess.call(command + [str(dfGeoXUL), str(dfGeoYLR), str(dfGeoXLR),
                                   str(dfGeoYUL), "-q", "-tr", xres, yres,
                                   vrt_file, tif_file])
        
        output_file = os.path.join(aligned_dir, base_filename.replace('.tif', output_suffix))
        cmd = f'gdal_translate -q "{vrt_file}" "{output_file}"'
        subprocess.call(cmd, shell=True)
        os.remove(vrt_file)
    
    return True

""" # Example usage
    rasters = [...]  # List of raster file paths
    output_suffix = '_dimensioned.tif')

    align_rasters(['path/to/raster1.tif', 'path/to/raster1.tif'], '_aligned.tif') """
    
def convert_pk_to_string(pk_string):
    """
    Function to convert "PK" to a formatted string
    :param pk_string: String column type.
    :type pk_string: string in a geodataframe,
    """

    if '+' in pk_string:
        # Remove any '+' characters and convert to an integer
        pk_integer = int(pk_string.replace('+', ''))
        # Format the integer as a string with leading zeros
        pk_formatted = f"{pk_integer:04d}"
    else:
        # If there's no '+', assume it's already an integer and format it with leading zeros
        pk_formatted = f"{int(pk_string):04d}"
    return pk_formatted


""" # Example usage
    gdf['PK'] = gdf['PK'].apply(convert_pk_to_string) """
    
    
def assign_crs(input_tiffs, crs):
    """
    Assigns a specified CRS to a single TIFF file or a list of TIFF files, saves them with a new filename that includes the CRS suffix, and then replaces the original files.

    :param input_tiffs: A single path to a TIFF file or a list of paths to TIFF files.
    :param crs: CRS to assign (e.g., 4326 for 'EPSG:4326').
    """
    # Ensure input_tiffs is a list even if a single file path is provided
    if not isinstance(input_tiffs, list):
        input_tiffs = [input_tiffs]
    
    for tiff in input_tiffs:
        with rasterio.open(tiff) as src:
            profile = src.profile
            profile.update(crs=CRS.from_epsg(crs))

            # Read data from the source TIFF
            data = src.read(1)  # Assuming it's a single band raster

            # Construct new filename with CRS suffix
            output_file = os.path.splitext(tiff)[0] + '_epsg' + str(crs) + '.tif'

            # Write out the new TIFF with the updated CRS
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(data, 1)

        # Replace the original file with the new file
        os.remove(tiff)
        os.rename(output_file, tiff)

""" # Example Usage for a single file
assign_crs('path_to_your_single_tiff_file.tif', 4326)

# Example Usage for multiple files
assign_crs(['path_to_your_first_tiff_file.tif', 'path_to_your_second_tiff_file.tif'], 4326)
 """

def stack_rasters(tiff_files, aoi_shapefile, output_tiff, chunk_size=None, operation=None):
    """
    Clips, stacks a list of raster files based on an AOI shapefile, calculates the sum, median, or mean of the stack,
    and saves the resulting raster to a file. Processes data in chunks to reduce memory usage.

    :param tiff_files: List of paths to the input TIFF files
    :param aoi_shapefile: Path to the AOI shapefile
    :param output_tiff: Path for the output raster file
    :param operation: Operation to perform on the stack ('sum', 'median', or 'mean')
    :param chunk_size: Size of chunks for processing (e.g., (500, 500))
    :return: None
    """
    # Load AOI shapefile as a GeoDataFrame
    aoi = gpd.read_file(aoi_shapefile)

    # Extract the first geometry from the AOI and put it in a list
    polygon_geometry = [aoi.geometry.iloc[0]]
    
    raster_arrays = []

    for tiff in tiff_files:
        raster = rioxarray.open_rasterio(tiff, chunks=chunk_size)

        # Clip the raster using the AOI geometry
        clipped_raster = raster.rio.clip(polygon_geometry, aoi.crs, drop=True, invert=False)
        no_data = clipped_raster.rio.nodata
        clipped_raster = clipped_raster.where(clipped_raster != no_data, other=0)
        raster_arrays.append(clipped_raster)

    stacked_rasters = xr.concat(raster_arrays, dim='band')

    # Perform the specified operation across the 'band' dimension
    if operation == 'sum':
        result_raster = stacked_rasters.sum(dim='band')
    elif operation == 'median':
        result_raster = stacked_rasters.median(dim='band')
    elif operation == 'mean':
        result_raster = stacked_rasters.mean(dim='band')
    else:
        raise ValueError("Invalid operation. Choose 'sum', 'median', or 'mean'.")

    # Set the CRS of the result raster to match the AOI shapefile's CRS
    result_raster.rio.write_crs(aoi.crs, inplace=True)

    # Write the result raster to a new file
    result_raster.rio.to_raster(output_tiff)

""" # Example usage
tiff_files = [...]  # List of paths to input TIFF files # 

tiff_files = glob.glob(os.path.join(raster_dir, '*.tif'))

output_tiff = r"E:\Spotlite_JPereira\Ascendi\Concessao_Norte\2023\Rainfall\sum_raster.tif"
aoi_shapefile = r"E:\Spotlite_JPereira\Ascendi\Concessao_Norte\AOI\CN_A7_A11_A42_expanded_bounds_wgs84.geojson"
chunk_size = (500, 500)  # Adjust based on your system's memory capacity

stack_rasters(tiff_files, aoi_shapefile, output_tiff, chunk_size, operation='sum') """

def addNDVI_ee(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
    return image.addBands(ndvi)

#https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#colab-python

def apply_cloud_masks_ee(image):
    # Cloud shadow 
    cloud_shadow = image.select('SCL').eq(3)
    # Medium Probability
    cloud_low = image.select('SCL').eq(7)
    # Medium Probability
    cloud_med = image.select('SCL').eq(8)
    # High Probability
    cloud_high = image.select('SCL').eq(9)
    # Cirrus Mask
    cloud_cirrus = image.select('SCL').eq(10)
    cloud_mask = cloud_shadow.add(cloud_low).add(cloud_med).add(cloud_high).add(cloud_cirrus)

    # Invert the selected images to mask out the clouds
    invert_mask = cloud_mask.eq(0)

    # Apply the mask to the image
    return image.updateMask(invert_mask)

def apply_scale_factor_ee(image):
    return image.multiply(0.0001)


def extract_files_ssm(input_dir):
    """
    Unzips the files corresponding to the SSM (without the noise band) inside the sub directories inside the main folder
    Build new folders in every subfolder with only the date of the file.
    
    :param input_dir: Path of the main directory where download files from SSM and LST are located
    """
    
    for subdir in os.listdir(input_dir):
        if subdir.startswith('SSM1km_'):
            for file in os.listdir(os.path.join(input_dir, subdir)):
                if file.startswith('c_gls_SSM1km_') and file.endswith('.zip'):
                    zip_file_path = os.path.join(input_dir, subdir, file)
                    extract_dir = os.path.join(input_dir, subdir)
                    tiff_file_exists = any(f.endswith('.tiff') for f in os.listdir(extract_dir))
                    if not tiff_file_exists:
                        shutil.unpack_archive(zip_file_path, extract_dir=extract_dir)
                        
    """     Example Usage  
    input_dir = r"E:\Spotlite_JPereira\Ascendi\Concessao_Norte\2023\SSM"
 
    extract_files_ssm_lst(input_dir) """
    
    
def folder_stack_ssm(input_dir):
    """
    Copies tiffs files and place it into the main input directory.
    Deletes no data files (because of no satellite passages in that date) and finally copies all tiffs in main folder 
    into a single folder with the name of first four characters of a single filename.

    :param input_dir: Path of the main directory where files of each sub directory are alredy unzipped.
    """
    
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
            
    # Iterate over the files in the input directory
    for file_name in os.listdir(input_dir):
        # Check if the file is a TIFF file
        if file_name.endswith(".tiff"):
            # Extract the year from the file name
            year = file_name.split("_")[2][:4]
            # Create the destination folder name based on the year
            dest_folder_name = f'SSM_1km_{year}'
            dest_folder_path = os.path.join(input_dir, dest_folder_name)

            # Create the destination folder if it doesn't exist
            if not os.path.exists(dest_folder_path):
                os.makedirs(dest_folder_path)

            # Construct the source and destination file paths
            src_file_path = os.path.join(input_dir, file_name)
            dst_file_path = os.path.join(dest_folder_path, file_name)

            # Move the file to the destination folder
            shutil.move(src_file_path, dst_file_path)
    
    """     Example Usage  
    input_dir = r"E:\Spotlite_JPereira\Ascendi\Concessao_Norte\2023\SSM"
 
    folder_stack_ssm_lst(input_dir)   """        
    
    
def ssm_nan_fix(raster_dir):
    """
    Fixes SSM data by replacing values above 200 with 255, then divide by 2 (scale factor of 0.5)
    except for the values that were set as 255.

    :param raster_dir: Path of the main directory where files of each subdirectory are already unzipped.
    """

    for filename in os.listdir(raster_dir):
        if filename.endswith('.tiff'):
            tiff_path = os.path.join(raster_dir, filename)
            
            # Open the TIFF file
            dataset = gdal.Open(tiff_path, gdal.GA_Update)
            
            if dataset is not None:
                # Read the raster data as a numpy array
                raster_array = dataset.ReadAsArray()
                
                # Set values above 200 as 255
                raster_array[raster_array > 200] = 255
                
                # Convert the array to float
                raster_array = raster_array.astype(float)
                
                # Divide values that are not equal to 255 by 2
                raster_array[raster_array != 255] /= 2
                
                # Write the modified array back to the TIFF file
                dataset.GetRasterBand(1).WriteArray(raster_array)
                
                # Close the dataset
                dataset = None
            else:
                print(f"Failed to open {tiff_path}")

    print("Processing complete.")
    
    
    """     Example Usage  
    raster_dir = r"E:\Spotlite_JPereira\Ascendi\Concessao_Norte\2023\SSM\SSM_1km_2023"
 
    ssm_nan_fix(raster_dir)  """  
    
    
def thin_laz_files(input_directory, step_size):
    # Pattern to match all LAZ files
    file_pattern = "*.laz"

        # Iterate over each LAZ file in the directory
    for laz_file in glob.glob(os.path.join(input_directory, file_pattern)):
        # Construct the output filename
        base_name = os.path.splitext(os.path.basename(laz_file))[0]
        output_file = os.path.join(input_directory, f"{base_name}_thinned.laz")

        # Decimate original laz files to decrease resolution and size
        lasthin_command = [
            "lasthin", "-i", laz_file, "-o", output_file, "-step", str(step_size)
        ]

        subprocess.run(lasthin_command)

    print(f"Decimation completed in {input_directory}")
    
    """    Example Usage  
input_directory_parte_1 = r"E:\Spotlite_JPereira\Ascendi\Concessao_Norte\Dados_Fornecidos\ANT_LiDar\ANT_Parte_1"
 
thin_laz_files(input_directory_parte_1, 1.0)  """  
    
    
def find_and_merge_thinned_files(input_directories, output_file):
    thinned_files = []
    # Pattern to match all thinned LAZ files
    thinned_file_pattern = "*_thinned.laz"

    # Finding thinned files in each input directory
    for directory in input_directories:
        thinned_files += glob.glob(os.path.join(directory, thinned_file_pattern))

    # Check if there are thinned files to merge
    if not thinned_files:
        print("No thinned files found to merge.")
        return

    # Merge thinned files to a single one
    lasmerge_command = ["lasmerge", "-i"] + thinned_files + ["-o", output_file]
    subprocess.run(lasmerge_command)
    print("Merging completed! Output file:", output_file)
    
"""    Example Usage  
    merged_output_file_1 = r"E:\Spotlite_JPereira\Ascendi\Concessao_Norte\Dados_Fornecidos\ANT_LiDar\ANT_Parte_1\CN_Parte1_Merged.laz"
    input_directory_parte_1 = r"E:\Spotlite_JPereira\Ascendi\Concessao_Norte\Dados_Fornecidos\ANT_LiDar\ANT_Parte_1"
    
    find_and_merge_thinned_files([input_directory_parte_1], merged_output_file_1)
  """  
  
  
def netcdf_to_geotiff_rainfall(input_file, output_folder, rcp_scenario, clip_file, start_year, end_year):
    """
    Converts NetCDFs to GeoTIFFs for a specified year range and scales rainfall factor for yearly mean.

    :param input_file: Path to the input NetCDF file.
    :param output_folder: Path to the output folder.
    :param rcp_scenario: RCP scenario (e.g., '4.5', '8.5').
    :param clip_file: Path to the clipping shapefile.
    :param start_year: Start year of the data to process.
    :param end_year: End year of the data to process.
    """
    # Calculate the scaling factor
    scale_factor = 3600 * 24 * 1000 * 30.4  # Conversion factor for precipitation

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
    clip_data = gpd.read_file(clip_file).to_crs(epsg=4326)

    # Iterate over the dates and convert each band to GeoTIFF
    for i, date in enumerate(cf_time_values):
        year = date.year

        # Only process bands within the specified year range
        if year < start_year or year > end_year:
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
    
    
"""    Example Usage  
    netcdf_to_geotiff_rainfall(
        input_file = *.nc, 
        output_folder = /path/to/directory, 
        rcp_scenario = '4.5', 
        clip_file = /path/to/geojson, 
        start_year = 2030,  # must be an integer, not string
        end_year = 2030)
  """  
  
def netcdf_to_geotiff_windspeed(input_file, output_folder, selected_year, rcp, clip_file=None):
    """
    Converts NetCDF wind speed data to GeoTIFF for a selected year and RCP scenario, and optionally clips the output.

    :param input_file: Path to the input NetCDF file.
    :param output_folder: Path to the output folder.
    :param selected_year: The year for which to process the data.
    :param rcp: The RCP scenario (e.g., '4.5', '8.5').
    :param clip_file: Optional path to a clipping shapefile.
    """
    # Open the NetCDF file
    nc_file = nc.Dataset(input_file, 'r')

    # Extract time-related information from the NetCDF file
    time_variable = nc_file.variables['time']
    time_values = time_variable[:]
    units = time_variable.units
    calendar = time_variable.calendar

    # Convert time values to datetime objects
    cf_time_values = nc.num2date(time_values, units, calendar=calendar)

    # Load the clipping shapefile if provided
    clip_data = gpd.read_file(clip_file).to_crs(epsg=4326) if clip_file else None

    # Iterate over time values and convert each band to GeoTIFF
    for i, date in enumerate(cf_time_values):
        if date.year == selected_year:  # Process only the selected year
            band_data = nc_file.variables['wind-speed'][i, :, :]

            date_str = date.strftime("%Y_%m_%d")
            output_file = os.path.join(output_folder, f'Climate-projection-Wind-Speed-Monthly-Mean-{rcp}-{date_str}.tiff')

            # Extract longitude and latitude values from the NetCDF file
            x_values = nc_file.variables['longitude'][:]
            y_values = nc_file.variables['latitude'][:]
            y_values = y_values[::-1]  # Reverse the order of y_values

            # Define the transformation and GeoTIFF profile
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
                'nodata': -9999
            }

            # Write the band data to the GeoTIFF file
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(band_data, 1)

            # Perform clipping if clip_data is provided
            if clip_data is not None and not clip_data.empty:
                clipped_output_file = os.path.join(output_folder, f'Climate-projection-Wind-Speed-Monthly-Mean-{rcp}-{date_str}_clipped.tiff')

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

    # Close the NetCDF file
    nc_file.close()

""" # Example usage
convert_netcdf_to_geotiff_windspeed('input.nc', 'output_folder', 2023, '4.5', 'clip_file.shp', ) """


def netcdf_to_geotiff_temperature(input_file, output_folder, rcp_scenario, variable_type, clip_file, selected_year):
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
    clip_data = gpd.read_file(clip_file) if clip_file else None

    # Set the CRS of the shapefile to EPSG 4326 if it's loaded
    if clip_data is not None:
        clip_data = clip_data.to_crs(epsg=4326)

    # Iterate over the years and convert each band to GeoTIFF
    for i, year in enumerate(years):
        if year == selected_year:  # Only process the selected year
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

            # Clip the GeoTIFF file with the clipping shapefile if it's loaded
            if clip_data is not None and not clip_data.empty:
                clipped_output_file = os.path.join(output_folder,
                                                   f'Climate-projection-{rcp_str}_Yearly_{variable_type}_{year}_clipped.tif')

                # Open the output GeoTIFF file
                with rasterio.open(output_file) as src:
                    # Clip the raster data using the feature geometry
                    clipped_data, clipped_transform = mask(src, clip_data['geometry'], crop=True)

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
    
def clip_raster_all_pixels(raster_path, vector_path, output_path, all_touched=True):
    
    """
    Clips a raster file with a vector one but including all the pixels that have at least one intersection wtith the raster.
    
    :param raster_path: Path to the input raster file.
    :param vector_path: Path to the vector mask layer.
    :output_path: Path to the output clipped file.
    :all_touched = True: Parameter to set the clipping to the maximum extent.
    """
    
    # Read the vector data
    vector_data = gpd.read_file(vector_path)

    # Read the raster data
    with rasterio.open(raster_path) as src:
        # Clip the raster with the vector data
        out_image, out_transform = mask(src, vector_data.geometry, crop=True, all_touched=all_touched)
        
        # Copy the metadata
        out_meta = src.meta.copy()

    # Update the metadata to have the new shape
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Write the clipped raster to a new file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)
    

        
""" # Example usage
raster_path = 'path_to_your_raster.tif'
vector_path = 'path_to_your_vector.shp'
output_path = 'path_to_output_raster.tif'

clip_raster_all_pixels(raster_path, vector_path, output_path) """



def split_raster_bands(file_path, directory_path):
    """
    Splits the bands of a multi-band raster file into individual single-band raster files.

    :param file_path: Path to the input multi-band raster file.
    :param directory_path: Directory where the output single-band raster files will be saved.
    """
    
    # Set the "no data" value to 0 in the main input raster
    with rasterio.open(file_path, 'r+') as src:
        src.nodata = 0

        # Iterate through each band and export as single-band TIFF
        for band_idx in range(1, src.count + 1):
            with rasterio.open(file_path, 'r') as src_band:
                # Read the band data for the current band
                band_data = src_band.read(band_idx)

                # Create the output TIFF file name (e.g., file_B1.tif, file_B2.tif, etc.)
                output_tiff_name = f"{os.path.basename(file_path).split('.')[0]}_B{band_idx}.tif"
                output_tiff_path = os.path.join(directory_path, output_tiff_name)

                # Create a new raster dataset for the current band
                profile = src_band.profile
                profile.update(
                    count=1,  # Set the count to 1 to create a single-band TIFF
                    dtype=rasterio.float64  # Adjust the data type as needed
                )

                with rasterio.open(output_tiff_path, 'w', **profile) as dst:
                    dst.write(band_data, 1)

                # Print a message indicating the export is complete for each band
                print(f"Exported {output_tiff_name}")

    # Print a final message
    print("All bands exported.")
    
    
    """ # Example usage
file_path = 'E:\\Spotlite_JPereira\\E-REDES\\Bruno\\spot_cutted.tif'
directory_path = r"E:\Spotlite_JPereira\E-REDES\Bruno"

split_raster_bands(file_path, directory_path) """