import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

###########################################
# ### **5.EDA: Descriptive Analysis of the "st_Maximum_individual_wave" TIME SERIES.**
###########################################

df_data = pd.read_csv('final_dataset.csv', index_col=0)
range_limit_of_selected_data = '2024-12-31' # ONLY TAKE DATA UNTIL 2024

# Sort by index
df_data = df_data.sort_index()
df_data


# **The time series “st_Maximum_individual_wave” is defined and transformed.**

max_wave_height_var= 'Maximum individual wave height'

# Define the time series with the values of the 'Maximun individual wave height':
st_Maximum_individual_wave = df_data[[max_wave_height_var]]

# Sort if not sorted:
st_Maximum_individual_wave.sort_index(inplace=True)

# Convert date to 'timestamp'' format:
st_Maximum_individual_wave.index = pd.to_datetime(st_Maximum_individual_wave.index, format='%Y-%m-%d')


# Define the frequency of the time series as monthly
st_Maximum_individual_wave = st_Maximum_individual_wave.asfreq('MS')


# #### **5.1. Descriptive analysis of the l "st_Maximum_individual_wave" time series.**
# 

# time series mean
print(f'The sample mean: {round(st_Maximum_individual_wave[max_wave_height_var].mean(), 2)}')

# median of the time series
print(f'The sample median: {round(st_Maximum_individual_wave[max_wave_height_var].median(), 2)}')

# time series minimum
print(f'The minimum sample value: {round(st_Maximum_individual_wave[max_wave_height_var].min(), 2)}')

# time series maximum
print(f'The maximum sample value: {round(st_Maximum_individual_wave[max_wave_height_var].max(), 2)}')

# time series variance
print(f'The sampling variance: {round(st_Maximum_individual_wave[max_wave_height_var].var(), 2)}')

# time series standard deviation
print(f'The sample standard deviation: {round(st_Maximum_individual_wave[max_wave_height_var].std(), 2)}')


# **The dates when the waves were lowest and highest are identified.**



# Define the interval of the height of the smallest waves:
lower_limit_small_wave = 1.0
upper_limit_small_wave = 1.38

# Define the range of the highest wave heights:
lower_limit_big_wave = 5.2
upper_limit_big_wave = 6.0

# Plot the time series:
plt.figure(figsize=(20, 6))
plt.plot(np.array(st_Maximum_individual_wave.index), st_Maximum_individual_wave.values, color='darkorange')

# Draw the horizontal line of the mean of the observed values:
mean_observed_values = st_Maximum_individual_wave[max_wave_height_var].mean()
plt.axhline(y=mean_observed_values, color='green', linestyle='--', linewidth=2)

# Add the value of the mean on the line:
plt.text(st_Maximum_individual_wave.index[-1], mean_observed_values,f'E[X] = {mean_observed_values:.2f}', fontsize=12, ha='left', va='bottom', color='darkgreen')

# Find the indexes of the points within the interval of the small wave height:
small_wave_indexes_interval = (st_Maximum_individual_wave[max_wave_height_var] >= lower_limit_small_wave) & (st_Maximum_individual_wave[max_wave_height_var] <= upper_limit_small_wave)

# Mark the points within the range of small wave height:
plt.scatter(st_Maximum_individual_wave.index[small_wave_indexes_interval], st_Maximum_individual_wave[max_wave_height_var][small_wave_indexes_interval], color='red', label='Rango rojo')

# Find the indexes of the points within the interval of the height of the large waves:
big_wave_indexes_interval = (st_Maximum_individual_wave[max_wave_height_var] >= lower_limit_big_wave) & (st_Maximum_individual_wave[max_wave_height_var] <= upper_limit_big_wave)

# Mark the points within the range of the large wave height:
plt.scatter(st_Maximum_individual_wave.index[big_wave_indexes_interval], st_Maximum_individual_wave[max_wave_height_var][big_wave_indexes_interval], color='blue', label='Rango azul')

# Add dates and values in the highlighted points:
for date, value in zip(st_Maximum_individual_wave.index[small_wave_indexes_interval], st_Maximum_individual_wave[max_wave_height_var][small_wave_indexes_interval]):
    plt.text(date, value, pd.to_datetime(date).strftime('%Y-%m-%d'), fontsize=12, ha='right', va='bottom', color='red')
    plt.text(date, value - 0.1, f'{value:.2f}', fontsize=12, ha='right', va='top', color='red')

for date, value in zip(st_Maximum_individual_wave.index[big_wave_indexes_interval], st_Maximum_individual_wave[max_wave_height_var][big_wave_indexes_interval]):
    plt.text(date, value, pd.to_datetime(date).strftime('%Y-%m-%d'), fontsize=12, ha='right', va='bottom', color='blue')
    plt.text(date, value - 0.1, f'{value:.2f}', fontsize=12, ha='right', va='top', color='blue')

# Set labels and title:
plt.title('Maximum height of the most significant waves between January 1940 and 2024', fontsize=15, pad=15)
plt.ylabel('Maximum wave height (metres)',  fontsize=12)

plt.grid(True, color='orange', linestyle='--')

# Change chart border colour to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')

# Show plot:
plt.savefig("03 - Maximum height of the most significant waves between January 1940 and 2024.png", dpi=300)
plt.show()




# Graphical representation of the (time) series:
st_Maximum_individual_wave.plot(figsize=(20, 6), color='darkorange')

plt.title('Time Series - Maximum wave height (January 1940 - December 2024)', fontsize=15, pad=15)
plt.ylabel('Maximum wave height (metres)', fontsize=12)

plt.legend(frameon=False)

# Change chart border colour to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')


# **Frequency table of the observed values of the series.**


# Calculate the frequency table:
frequency_table = st_Maximum_individual_wave[max_wave_height_var].value_counts().reset_index()
frequency_table.columns = ['Value', 'Frequency']

# Sort the table by values:
frequency_table = frequency_table.sort_values(by='Value')

# Filter to show only frequencies greater than 1 to see if the frequency distribution is uni-variate or multivariate.
frequencies_greater_than_1 = frequency_table[frequency_table['Frequency'] > 1]

# Display frequency table or message:
if frequencies_greater_than_1.empty: # check if it is empty
    print('No frequencies higher than 1.')
else:
    print('Frequency table with frequencies greater than 1:\n', frequencies_greater_than_1)


# **Histogram of the observed values of the time series.**



# Calculating mean, meann and mode:
mean = st_Maximum_individual_wave[max_wave_height_var].mean()
median = st_Maximum_individual_wave[max_wave_height_var].median()

# Make plot:
plt.figure(figsize=(10, 6))
plt.hist(st_Maximum_individual_wave[max_wave_height_var], bins=70, color='darkorange', edgecolor='black') # The number of bins can be adjusted according to preference

# Add vertical lines for the mean, median and mode:
plt.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')

plt.title('Histogram of the observed values of the time series', fontsize=14)
plt.xlabel('Observed values', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Change chart border colour to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')

# Add legend:
plt.legend(frameon=False, fontsize=15)
plt.savefig("03 - Histogram of the observed values of the time series.png", dpi=300)
plt.show()


# **QQ-plot, to check if the distribution is normal.**



import statsmodels.api as sm

# Make the Q-Q plot:
plt.figure(figsize=(10, 6))
sm.qqplot(st_Maximum_individual_wave[max_wave_height_var],  line='s')
          #line='s', markerfacecolor='darkorange')

plt.title('Q-Q plot of the time series frequency distribution', fontsize=14)
plt.xlabel('Theoretical quantiles', fontsize=12)
plt.ylabel('Observed quantiles', fontsize=12)

# Change chart border colour to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')

plt.savefig("03 - Q-Q plot of the time series frequency distribution.png", dpi=300)
plt.show()




# **Mann-Kendall test for stationarity of the time series with respect to the mean.**
# pip install pymannkendall
from pymannkendall import original_test

# Target significance level (alpha):
significance_level = 0.05

# Perform the Mann-Kendall test:
mann_kendall_results = original_test(st_Maximum_individual_wave)

# Print additional details of the results:
print('Mann-Kendall test:')
print(f'Test statistician: {round(mann_kendall_results.z, 4)}')
print(f'p-value: {mann_kendall_results.p}')

# Check the direction of the trend:
if mann_kendall_results[0] == 'increasing':
    print('Trend: ascending \n')
elif mann_kendall_results[0] == 'decreasing':
    print('Trend: downward \n')
else:
    print('No clear trend in the data \n')


# Compare p-value with significance level:
if mann_kendall_results.p < significance_level:
    print('The null hypothesis is rejected. There is statistical evidence that there is a significant trend in the data.')
else:
    print('The null hypothesis is NOT rejected. There is NOT enough evidence to conclude that there is a significant trend in the data.')

"""
The Mann-Kendall test indicates a statistically significant increasing trend in the time series of individual maximum wave heights. 
The low p-value (0.00157) and positive Z-statistic (3.1613) reinforce the presence of this trend. 
These results suggest that the values of maximum wave heights have increased over time, which could be associated with environmental or climatic factors.
"""


# **Standard deviation-mean diagram for stationarity of the time series with respect to variance.**



# Number of 12 data sets :
num_groups = len(st_Maximum_individual_wave) // 12

# Lists to store means, standard deviations and years for each group:
means = []
stds = []
years = []

# Iterate over the data sets and calculate the mean, standard deviation and year for each:
for i in range(num_groups):
    group_start = i * 12
    group_end = (i + 1) * 12
    group_data = st_Maximum_individual_wave.iloc[group_start:group_end]
    group_mean = group_data.mean()
    group_std = group_data.std()
    group_year = st_Maximum_individual_wave.index[group_start].year  # Get the year of the first data of the group
    means.append(group_mean)
    stds.append(group_std)
    years.append(group_year)

# Calculate the mean and standard deviation for the last group (year 2024) that has less than 12 data:
last_group_data = st_Maximum_individual_wave.iloc[num_groups * 12:]
last_group_mean = last_group_data.mean()
last_group_std = last_group_data.std()
last_group_year = st_Maximum_individual_wave.index[num_groups * 12].year  # get the year of the first data of the last group
means.append(last_group_mean)
stds.append(last_group_std)
years.append(last_group_year)

# Scatter plot:
plt.figure(figsize=(10, 6))
plt.scatter(means, stds, color='darkorange')

plt.title('Standard Deviation - Mean Diagram', fontsize=15)
plt.xlabel('Mean per year', fontsize=12)
plt.ylabel('Standard deviation by year', fontsize=12)

# Show the years of each point on the chart:
for i in range(len(means)):
    plt.text(means[i], stds[i] + 0.01, years[i], fontsize=8, ha='center', va='bottom', color='brown')

# Change chart border colour to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')
    
plt.savefig("03 - Standard Deviation - Mean Diagram.png", dpi=300)
plt.show()


# **Coefficient of variation of each year of the time series.**


# Calculate the coefficient of variation for each year:
cv_per_year = {}

# Group the data by year and calculate the mean and standard deviation for each year:
for year, data_year in st_Maximum_individual_wave[max_wave_height_var].groupby(st_Maximum_individual_wave.index.year):
    mean_year = data_year.mean()
    std_year = data_year.std()

    # Calculate the coefficient of variation:
    cv_per_year[year] = (std_year / mean_year) * 100  # Multiply by 100 to express as a percentage

# Convert dictionary to DataFrame for displaying results:
cv_per_year_df = pd.DataFrame.from_dict(cv_per_year, orient='index', columns=['Coeficiente de Variacion'])
cv_per_year_df.index.name = 'Año'

# Create the scatter plot:
plt.figure(figsize=(20, 6))
plt.scatter(cv_per_year_df.index, cv_per_year_df['Coeficiente de Variacion'], color='darkorange')

# Add the coefficient of variation values next to each point:
for i, txt in enumerate(cv_per_year_df['Coeficiente de Variacion']):
    plt.text(cv_per_year_df.index[i], txt, f'{txt:.2f}', ha='center', va='bottom', color='brown', fontsize=15)

plt.title('Coefficient of Variation per Year', fontsize=15)
plt.ylabel('Coefficient of Variation (%)', fontsize=12)

# Adjust x-axis ticks to display only every two years:
plt.xticks(cv_per_year_df.index[::5])

# Change chart border colour to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')

plt.savefig("03 - Coefficient of Variation per Year.png", dpi=300)
plt.show()


# #### **5.2. Decomposition of the time series.**


from statsmodels.tsa.seasonal import seasonal_decompose

st_Maximum_individual_wave.index = pd.to_datetime(st_Maximum_individual_wave.index)
series = st_Maximum_individual_wave[max_wave_height_var]

# Additive decomposition:
result_add = seasonal_decompose(series, model='additive', period=12)

# Multiplicative decomposition:
result_mul = seasonal_decompose(series, model='multiplicative', period=12)

# Create figure for parallel graphs:
fig, axes = plt.subplots(4, 2, figsize=(15, 12))

# Additive decomposition graphs:
axes[0, 0].plot(result_add.observed)
axes[0, 0].set_title('Original Time Series (Additive)')

axes[1, 0].plot(result_add.trend)
axes[1, 0].set_title('Trend (Additive)')

axes[2, 0].plot(result_add.seasonal)
axes[2, 0].set_title('Seasonality (Additive)')

axes[3, 0].plot(result_add.resid)
axes[3, 0].set_title('Residuals (Additive)')

# Multiplicative decomposition graphs
axes[0, 1].plot(result_mul.observed)
axes[0, 1].set_title('Original Time Series (Multiplicative)')

axes[1, 1].plot(result_mul.trend)
axes[1, 1].set_title('Trend (Multiplicative)')

axes[2, 1].plot(result_mul.seasonal)
axes[2, 1].set_title('Seasonality (Multiplicative)')

axes[3, 1].plot(result_mul.resid)
axes[3, 1].set_title('Residuals (Multiplicative)')

# Adjust the subplots:
plt.tight_layout()
plt.savefig("03 - Additive and Multiplicative decomposition graphs.png", dpi=300)
plt.show()


