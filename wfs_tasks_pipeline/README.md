## Overview

This repository presents the tasks developed for the World From Space project. The tasks are divided into two main groups:

- **Task 1**:  Create a single raster image representing the mean of the input NDVI rasters, excluding no-data values from the calculation.

  - The **output raster file** is saved at: "{current_dir}gis_star_task_01/NDVI/brno_mean_ndvi_2023.tiff"
  - The **vegetation map of Brno** is saved at: "{current_dir}/gis_star_task_01/NDVI_Image/task1_brno_mean_ndvi_illustrator-01.png"

- **Task 2**: Reclassify each input EVI raster using a chosen method. For this task, k-means clustering was used to create five classes.
  
  - Each **individual reclassified EVI** is stored in: "{current_dir}/gis_star_task_02/EVI/output"
  - The **mean from the 5 reclassified rasters** is located at: "{current_dir}/gis_star_task_02/EVI\output/vraz_evi_kmeans_mean.tiff".
  - The **final mean raster converted to integer** is located at: "{current_dir}/gis_star_task_02/EVI/output/vraz_evi_kmeans_mean_integer.tiff"
  

## Auxiliary functions python to file to store all the functions used in the jpereira_wfs_tasks.ipynb: aux_functions.py

- stack_rasters
- process_directory
- convert_raster_to_integers

**To apply the functions in the notebook, use af.{function_name}**

## Created a .yaml file to help with the package management installation

