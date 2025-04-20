import pandas as pd 
import pygrib # For working with meteorological data files in GRIB format
import sys
import os

"""

###########################################
# ### 1. Load Data
###########################################

DATA SOURCE:
-----------

https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download


DATA DESCRIPTION:
-----------------

Years: Select all years since 1940 to current year
Month: Select all months
Day: Select all days
Time: Select Time 00:00

Geographical area and Sub-region extraction (Cantabrian coast):
   • North: 43.67
   • West: -4.5
   • South: 43.35
   • East: -3.15

Data format: GRIB

Unarchived (not zipped if single file)


WIND:
----

10m u-component of wind, 
10m v-component of wind, 
100m u-component of wind, 
100m v-component of wind, 
10m u-component of neutral wind, 
10m v-component of neutral wind, 
10m wind speed, 
Instantaneous 10m wind gust, 


TEMPERATURE AND PRESSURE:
------------------------

2m dewpoint temperature, 
2m temperature, 
Mean sea level pressure, 
Mean wave direction, 
Mean wave period, 
Sea surface temperature, 
Surface pressure, 
Skin temperature, 


RADIATION AND HEAT:
-------------------

Clear-sky direct solar radiation at surface, 
Downward UV radiation at the surface, 
Forecast logarithm of surface roughness for heat, 
Instantaneous surface sensible heat flux, 
Near IR albedo for diffuse radiation, 
Near IR albedo for direct radiation, 
Surface latent heat flux, 
Surface net solar radiation, 
Surface net solar radiation, clear sky, 
Surface net thermal radiation, 
Surface net thermal radiation, clear sky, 
Surface sensible heat flux, 
Surface solar radiation downward, clear sky, 
Surface solar radiation downwards, 
Surface thermal radiation downward, clear sky, 
Surface thermal radiation downwards, 
TOA incident solar radiation, 
Top net solar radiation, 
Top net solar radiation, 
clear sky, Top net thermal radiation, 
Top net thermal radiation, clear sky, 
Total sky direct solar radiation at surface, 
UV visible albedo for diffuse radiation, 
UV visible albedo for direct radiation, 


OCEAN WAVES:
------------

Air density over the oceans
Coefficient of drag with waves
Free convective velocity over the oceans
************** Maximum individual wave height # THIS IS THE TARGET VARIABLE
Mean direction of total swell
Mean direction of wind waves
Mean period of total swell
Mean period of wind waves
Mean square slope of waves
Mean wave direction
Mean wave direction of first swell partition
Mean wave direction of second swell partition
Mean wave direction of third swell partition
Mean wave period
Mean wave period based on first moment
Mean wave period based on first moment for swell
Mean wave period based on first moment for wind waves
Mean wave period based on second moment for swell
Mean wave period based on second moment for wind waves
Mean wave period of first swell partition
Mean wave period of second swell partition
Mean wave period of third swell partition
Mean zero-crossing wave period
Model bathymetry
Normalized energy flux into ocean
Normalized energy flux into waves
Normalized stress into ocean
Ocean surface stress equivalent 10m neutral wind direction
Ocean surface stress equivalent 10m neutral wind speed
Peak wave period
Period corresponding to maximum individual wave height
Significant height of combined wind waves and swell
Significant height of total swell
Significant height of wind waves
Significant wave height of first swell partition
Significant wave height of second swell partition
Significant wave height of third swell partition
Wave spectral directional width
Wave spectral directional width for swell
Wave spectral directional width for wind waves
Wave spectral kurtosis
Wave spectral peakedness
Wave spectral skewness

"""


# **Data extraction.**

# GRIB file Path:
file_path = 'ERA5_Cantabrian_Coast_1940-2024.grib'
original_csv_file_path = 'original_dataset.csv'
final_csv_file_path = 'final_dataset.csv'


# Open GRIB file:
if os.path.exists(file_path):  # Check if file exists
  grib_file = pygrib.open(file_path)
else:
  print(f"ERROR: file {file_path} not found")
  sys.exit()
  

if not os.path.exists(original_csv_file_path):   
  # Dictionary to store all the information necessary for our study:
  data = {}

  for grb in grib_file:
    if grb.validDate not in data:
        data[grb.validDate] = {} # Dictionary to store data by date:

    data[grb.validDate][grb.name] = grb.average
    #data[grb.validDate][grb.name + ' - mean'] = grb.average
    #data[grb.validDate][grb.name + ' - std'] = grb.standardDeviation

  # Close grib file
  grib_file.close()

  df_data = pd.DataFrame(data)
  df_data = df_data.T # Transpone the DataFrame
  # **Save the dataset df_data in a csv file.**
  df_data.to_csv(original_csv_file_path, index=True)
else:
  df_data = pd.read_csv(original_csv_file_path, index_col=0, parse_dates=True)
  print(f"Loaded file {original_csv_file_path}")


###########################################
# ### 2.Data transformation
###########################################

# Sort by index ‘date’.
df_data = df_data.sort_index()

print(f'Number of variables (columns): {df_data.shape[1]}')
print(f'NNumber of records: {df_data.shape[0]}')

# There are 94 columns corresponding to the monthly means and standard deviations of the 47 variables obtained from the grib file.
# 
# We have 2020 records twice the number of records obtained per variable in the grib file. The reason for this quantum mismatch is analysed..

#df_data.head(20) # Displays the first twenty records of the dataset

# Looking at the dataset df_data, we can see a clear pattern in the missing values: there is a group of variables that have missing values on the same dates and another group of variables with missing values on different dates.
# It is verified that this pattern occurs throughout the dataset.


# Create two lists
group1_variable_list = [df_data.columns[0]] # This list will have all variables that have missing values on the same dates compared to the first variable of df_data
group2_variable_list = [] # This list will have all the variables that do not have missing values on the same dates

# Generate a list with all variables except the first one in df_data
columns_to_compare = df_data.columns.drop(df_data.columns[0])

# Generate a list of all dates(indices) that have null values in the first variable of df_data
indexes_with_missing = df_data[df_data[df_data.columns[0]].isnull()].index.tolist()

for var in columns_to_compare:
    indexes_with_missing_var = df_data[df_data[var].isnull()].index.tolist()
    if indexes_with_missing == indexes_with_missing_var:
        group1_variable_list.append(var) # add the variable to the list of variables with missing values on the same dates(indices) as the first variable of df_data
    else:
        group2_variable_list.append(var) # if not, add the variable to the list of variables with missing values with dates(indices) different from the dates(indices) of the first variable of df_data

print('\033[1mFIRST COMPARISON of variables with missing values on the same dates: \033[0m')
print('\033[1mVariables with missing values on the same dates:\033[0m')
i = 1
if len(group1_variable_list) != 0:
    for var in group1_variable_list:
        print(f' {i}. {var}')
        i += 1
else:
    print('No variable has missing values on different dates')

print('')
print('\033[1mVariables with missing values at different dates:\033[0m')
i = 1
if len(group2_variable_list) != 0:
    for var in group2_variable_list:
        print(f' {i}. {var}')
        i += 1
else:
    print('No variable has missing values on different dates')

print('')
print(f'\033[1mNumber of variables with missing values on the same dates:\033[0m  {len(group1_variable_list)}')
print(f'\033[1mNumber of variables with missing values at different dates:\033[0m {len(group2_variable_list)}')
total_listas = len(group1_variable_list) + len(group2_variable_list)
print(f'\033[1mTotal number of variables in the two lists:\033[0m {total_listas}')

if df_data.shape[1] == total_listas:
    print('The total number of variables in the two lists MATCHES the total number of variables in df_total\n')
else:
    print('The total number of variables in the two lists DOES NOT MATCH the total number of variables in df_total\n')

# -- Perform the same operation but with the list ‘group2_list_variables’ to confirm if all its variables have their missing values on the same dates(indices).
# -- Otherwise, perform the operation as many times as different groups of variables appear.

# Copy group2_list_variables:
columns_to_compare = group2_variable_list.copy()

# Remove the first variable from the list group2_list_variables:
columns_to_compare.remove(group2_variable_list[0])

group21_variable_list = [group2_variable_list[0]]
group22_variable_list = []

indexes_with_missing = df_data[df_data[group2_variable_list[0]].isnull()].index.tolist()

for var in columns_to_compare:
    indexes_with_missing_var = df_data[df_data[var].isnull()].index.tolist()
    if indexes_with_missing == indexes_with_missing_var:
        group21_variable_list.append(var)
    else:
        group22_variable_list.append(var)

print('\033[1mSECOND COMPARISON of variables with missing values on the same dates:\033[0m')
print('\033[1mVariables with missing values on the same dates of the second block:\033[0m')
i = 1
if len(group21_variable_list) != 0:
    for var in group21_variable_list:
        print(f' {i}. {var}')
        i += 1
else:
    print('No variable has missing values on different dates')

print('')
print('\033[1mVariables with missing values at different dates:\033[0m')
i = 1
if len(group22_variable_list) != 0:
    for var in group22_variable_list:
        print(f' {i}. {var}')
        i += 1
else:
    print('No variable has missing values on different dates')

print('')
print(f'\033[1mNumber of variables with missing values on the same dates:\033[0m  {len(group21_variable_list)}')
print(f'\033[1mNumber of variables with missing values at different dates:\033[0m {len(group22_variable_list)}')


# The variables ‘Friction velocity’, ‘Instantaneous 10 metre wind gust’, 'Surface latent heat flux’, ‘Surface latent heat flux’ have missing values on different dates (indices) than the rest of the variables. 
# These dates correspond to the day before 18:00 for the rest of the variables.
# To keep the data for these variables, the data is copied to the next day.


indexes_with_missing_Air = df_data[df_data[group1_variable_list[0]].isnull()].index.tolist()
indexes_with_missing_Friction = df_data[df_data[group2_variable_list[0]].isnull()].index.tolist()

print(f'Dates with values in the variables listed above:\n {indexes_with_missing_Air}\n')
print(f'Dates with missing values on the dates listed above:\n {indexes_with_missing_Friction}')

# Considering that weather conditions from one day to the next may affect hours later, we will fill in the value of these variables on the following day.
# To do this, when we find a non-missing value in each of the above-mentioned variables.
# Compare whether the next row is the next day and, if so, copy the data to that row.

df_data.index = pd.to_datetime(df_data.index) # It is changed to date format

# Iterates from the second row to the second-to-last row:
for i in range(len(df_data)-1):
    # Iterate over all variables in each row:
    for var in df_data.columns:
        if pd.notnull(df_data.iloc[i][var]): # if the value in the current variable is not null
            current_date = df_data.index[i].date()
            next_date = df_data.index[i+1].date()

            if current_date + pd.Timedelta(days=1) == next_date: # if the date of the current row is one day before the date of the next row
                df_data.iloc[i+1][var] = df_data.iloc[i][var]


# Delete rows where there are no values in the rest of the variables:
for date in indexes_with_missing_Air:
    df_data = df_data.drop(date, axis=0) # Delete the row


# Check whether any missing values have been left throughout the process:
print(f'\033[1mMissing values in the final dataset:\033[0m {df_data.isnull().sum().sum()} valores \n')

# Check the total number of rows which should correspond to 1010 which are the messages in the original grib file for each variable:
print(f'\033[1mTotal number of rows in the final dataset:\033[0m {df_data.shape[0]} filas(instancias)')


df_data.to_csv(final_csv_file_path, index=True)



