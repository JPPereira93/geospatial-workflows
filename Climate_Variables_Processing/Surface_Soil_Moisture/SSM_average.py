import pandas as pd
import argparse
import os

#------------------------------------------------------
# Calculate average surface soil moisture for each row. Use this script after https://github.com/JPPereira93/pygeo/blob/main/SSM_to_csv.py
#
# author: Joao Pereira
#------------------------------------------------------

parser = argparse.ArgumentParser(description='Process CSV file for average surface soil moisture')

parser.add_argument('-i', '--input_file', required=True, help='Path to the input CSV file.')

args = parser.parse_args()

input_file = args.input_file

# Use the input filename to generate the output filename
output_name = os.path.splitext(input_file)[0] + '_average.csv'

csv_file = pd.read_csv(input_file)

# calculate the average of each row
csv_file['AVERAGE'] = csv_file.iloc[:, 2:].mean(axis=1).round(2)

csv_file.to_csv(output_name, index=False)
