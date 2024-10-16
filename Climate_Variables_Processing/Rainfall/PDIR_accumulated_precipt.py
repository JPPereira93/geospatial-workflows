import pandas as pd
import argparse

#------------------------------------------------------
# Calculate accumulated precipitation for each row. Use this script after using: https://github.com/JPPereira93/pygeo/blob/main/PDIR_to_grid_to_csv.py
#
#@author: Joao Pereira  
#------------------------------------------------------


parser = argparse.ArgumentParser(description='Process CSV file for cumulative precipitation')

parser.add_argument('-i', '--input_file', required=True, help='Path to the input CSV file.')
parser.add_argument('-o', '--output_file', required=True, help='Path to the output CSV file.')

args = parser.parse_args()

# read the input CSV file
csv_file = pd.read_csv(args.input_file)

# define the field name and first precipitation column index
field_name = "Cumulative-Precipitation"
first_precipitation_column_index = 2

# sum the precipitation values for each row and add the cumulative precipitation sums to  field
precipitation_columns = csv_file.columns[first_precipitation_column_index:]
csv_file[field_name] = csv_file[precipitation_columns].sum(axis=1)

# save the modified CSV file
csv_file.to_csv(args.output_file, index=False)


