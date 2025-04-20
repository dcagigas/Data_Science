import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import ttest_ind


###########################################
# ### 3.Extraction and transformation of the “Maximum individual wave height” time series. Data extraction.
###########################################

final_csv_file_path = 'final_dataset.csv'
range_limit_of_selected_data = '2024-12-31' # ONLY TAKE DATA UNTIL 2024
max_wave_height_var= 'Maximum individual wave height'

df_data = pd.read_csv(final_csv_file_path, index_col=0)

# Sort by index
df_data = df_data.sort_index()


# **The time series “st_Maximum_individual_wave” is defined and transformed.**
st_Maximum_individual_wave = df_data[[max_wave_height_var]]

# Sort if not sorted:
st_Maximum_individual_wave.sort_index(inplace=True)

# Convert date to 'timestamp'' format:
st_Maximum_individual_wave.index = pd.to_datetime(st_Maximum_individual_wave.index, format='%Y-%m-%d')

# Define the frequency of the time series as monthly
st_Maximum_individual_wave = st_Maximum_individual_wave.asfreq('MS')


# **It is verified that there is data in all months from January 1940 to February 2024 (or beyond)).**


# Verify that a temporary index is complete:
complete_temporal_index = (st_Maximum_individual_wave.index == pd.date_range( 
    start = st_Maximum_individual_wave.index.min(),
    end   = st_Maximum_individual_wave.index.max(),
    freq  = st_Maximum_individual_wave.index.freq)
).all()

if complete_temporal_index.all():
    print('There ARE values in all months from January 1940 through 2024.')
else:
    print('NO values exist for all months from January 1940 to February 2024.')


# IMPORTANT: GET DATA JUST UNTIL A DATE.
st_Maximum_individual_wave = st_Maximum_individual_wave[st_Maximum_individual_wave.index <= range_limit_of_selected_data]


###########################################
# ### **4.EDA. Descriptive analysis of the STATISTICAL PROCESS of the variable “Maximum_individual_wave”.**
###########################################

# first date of the time series
print(f'First date of the time series: {st_Maximum_individual_wave.index.min()}')

# last date of the time series
print(f'Last date of the time series: {st_Maximum_individual_wave.index.max()}')


# Group the data by month and year. You get a list of DataFrames, one for each month.
st_Maximum_individual_wave_per_month = [grupo for _, grupo in st_Maximum_individual_wave.groupby([st_Maximum_individual_wave.index.month])]

# Verify that there are no null values in any of the months:
dataframes_with_null_values = []

for df in st_Maximum_individual_wave_per_month:
    if df.isnull().values.any():
        dataframes_with_null_values.append(df) # if null values are present, the DataFrame is added to the list of DataFrames with null values

if len(dataframes_with_null_values) == 0: # It is checked for DataFrames with null values
    print('NO null values.')
else:
    print('There are zero values in the following months:')
    for df in dataframes_with_null_values:
        print(df)


# **Sample size of the ‘months’ variables of the stochastic process.**


for df_monthly in st_Maximum_individual_wave_per_month:
    if df_monthly.index.month[0] == 1:
        print(f'The sample size of the variable JANUARY in the time series is: {len(df_monthly)}')
    elif df_monthly.index.month[0] == 2:
        print(f'The sample size of the variable FEBRUARY in the time series is: {len(df_monthly)}')
    elif df_monthly.index.month[0] == 3:
        print(f'The sample size of the variable MARCH in the time series is: {len(df_monthly)}')
    elif df_monthly.index.month[0] == 4:
        print(f'The sample size of the variable APRIL in the time series is: {len(df_monthly)}')
    elif df_monthly.index.month[0] == 5:
        print(f'The sample size of the time series variable MAY is: {len(df_monthly)}')
    elif df_monthly.index.month[0] == 6:
        print(f'The sample size of the variable JUNE in the time series is: {len(df_monthly)}')
    elif df_monthly.index.month[0] == 7:
        print(f'The sample size of the variable JULY in the time series is: {len(df_monthly)}')
    elif df_monthly.index.month[0] == 8:
        print(f'The sample size of the time series variable AUGUST is: {len(df_monthly)}')
    elif df_monthly.index.month[0] == 9:
        print(f'The sample size of the variable SEPTEMBER in the time series is: {len(df_monthly)}')
    elif df_monthly.index.month[0] == 10:
        print(f'The sample size of the variable OCTOBER in the time series is: {len(df_monthly)}')
    elif df_monthly.index.month[0] == 11:
        print(f'The sample size of the variable NOVEMBER in the time series is: {len(df_monthly)}')
    else:
        print(f'The sample size for the time series variable DECEMBER is: {len(df_monthly)}')




# Sampling distribution of the variables ‘months’:
plt.figure(figsize=(10, 6))

for df_month in st_Maximum_individual_wave_per_month:
    plt.plot(np.array(df_month.index.strftime('%b')), df_month[max_wave_height_var].values, marker='o', linestyle='-', label=f'Month {df_month.index.month}')

#plt.title('Sampling distribution of the ‘months’ variables of the time series')
plt.title('Maximum wave heights since 1940 to 2024 grouped by month', fontsize=15)
plt.ylabel('Maximum wave height (metres)', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("02 - Maximum wave heights since 1940 to 2024 grouped by month.png", dpi=300, bbox_inches="tight")
plt.show()


# Plot Maximum wave heights by decade:
# Define the decades to analyze (from 1940s to 2020s)
decades = list(range(1940, 2030, 10))[:-1]
monthly_avg_heights = {decade: [] for decade in decades}

# Compute monthly averages for each decade
for decade in decades:
    decade_data = st_Maximum_individual_wave[
        (st_Maximum_individual_wave.index.year >= decade)
        & (st_Maximum_individual_wave.index.year < decade + 10)
    ]
    for month in range(1, 13):
        monthly_values = decade_data[decade_data.index.month == month][max_wave_height_var]
        monthly_avg_heights[decade].append(monthly_values.mean())

# Ensure that 2020s data is included (only from 2020 to 2024)
decade_2020 = 2020
decade_2020_data = st_Maximum_individual_wave[
    (st_Maximum_individual_wave.index.year >= decade_2020)
    & (st_Maximum_individual_wave.index.year <= 2024)
]

# Compute monthly averages for 2020s
monthly_avg_2020s = []
for month in range(1, 13):
    monthly_values = decade_2020_data[decade_2020_data.index.month == month][max_wave_height_var]
    monthly_avg_2020s.append(monthly_values.mean())

# Add the 2020s data to the dictionary
monthly_avg_heights[2020] = monthly_avg_2020s
decades.append(2020)  # Ensure 2020s is in the list

# Define colors and linewidths for better visualization
colors = plt.cm.plasma(np.linspace(0, 1, len(decades)))  
linewidths = np.linspace(1, 3, len(decades))  

# Plot the data with enhanced visualization
plt.figure(figsize=(12, 6))
for i, (decade, heights) in enumerate(monthly_avg_heights.items()):
    plt.plot(range(1, 13), heights, marker="o", label=f"{decade}s", 
             color=colors[i], linewidth=linewidths[i])

# Highlight recent decades with shading
plt.axhspan(min(monthly_avg_heights[2010]), max(monthly_avg_heights[2020]), 
            color='lightgray', alpha=0.3, label="Recent Decades Highlight")

# Labels and title
#plt.xlabel("Month")
plt.ylabel("Average Maximum Wave Height (m)", fontsize=14)
plt.title("Average Monthly Maximum Wave Height by Decade", fontsize=15)

# Set month labels on the X-axis
plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=14)
plt.yticks(fontsize=14)

# Move the legend to the upper-center of the graph
plt.legend(title="Decade", loc="upper center", bbox_to_anchor=(0.5, .95),
           fontsize="medium", frameon=True, facecolor="white", edgecolor="black", ncol=3)

# Enable grid for better readability
plt.grid(True)

plt.savefig("02 - Average Monthly Maximum Wave Height by Decade (1940-2024).png", dpi=300)
plt.show()


# Statistics of the ‘months’ variables of the stochastic process:

print('MONTH         AVERAGE               STANDARD DEVIATIONA    MINIMUM               MAXIMUM               PERCENTILE 0.25       PERCENTILE 0.50       PERCENTILE 0.75')
print('---------------------------------------------------------------------------------------------------------------------------------------------------------------------')

for df_monthly in st_Maximum_individual_wave_per_month:
    month = df_monthly.index.month[0]  # Get the month of the DataFrame
    month_name = df_monthly.index.month_name()[0].upper()  # Get the name of the month
    mean = round(df_monthly[max_wave_height_var].mean(), 2)
    deviation = round(df_monthly[max_wave_height_var].std(), 2)
    minimo = round(df_monthly[max_wave_height_var].min(), 2)
    maximo = round(df_monthly[max_wave_height_var].max(), 2)
    percentil_25 = round(df_monthly[max_wave_height_var].quantile(0.25), 2)
    percentil_50 = round(df_monthly[max_wave_height_var].quantile(0.50), 2)
    percentil_75 = round(df_monthly[max_wave_height_var].quantile(0.75), 2)

    print(f'{month_name:<10} {mean:<20} {deviation:<20} {minimo:<20} {maximo:<20} {percentil_25:<20} {percentil_50:<20} {percentil_75:<20}')


# **Box plots of the sampling distributions of the ‘months’ variables of the stochastic process.**

import calendar

fig, axes = plt.subplots(nrows=1, ncols=12, figsize=(20, 5)) # cuadrícula de subgráficos
fig.suptitle('Box plots of the maximum wave heights since 1940 to 2024 for each month', fontsize=20, y=1.05)

# Box plot of the distribution for each month:
for i, df_monthly in enumerate(st_Maximum_individual_wave_per_month):
    month_number = df_monthly.index.month[0]
    month_name = calendar.month_name[month_number].upper()

    df_monthly.plot.box(ax=axes[i], color='purple', fontsize=9)
    axes[i].set_ylim(0, 6)
    axes[i].set_title(month_name, fontsize=9)
    axes[i].set_xticklabels([])  # The namonth of the X-axis ticks are removed


# **Box plots with the values of the statistics of the sampling distributions of the ‘months’ variables of the stochastic process.**


fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(22, 10)) # cuadrícula de subgráficos

months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']

for i, df_monthly in enumerate(st_Maximum_individual_wave_per_month):
    row = i // 6  # we determine the row of the sub-chart
    column = i % 6  # we determine the column of the sub-chart
    month = months[i]

    # Box plot:
    df_monthly.plot.box(ax=axes[row, column], color='black', fontsize=8)
    axes[row, column].set_ylim(0, 6)
    axes[row, column].set_title(f'{month}', fontsize=10)

    # Statistics:
    median = np.median(df_monthly[max_wave_height_var])
    media = np.mean(df_monthly[max_wave_height_var])
    q1 = np.percentile(df_monthly[max_wave_height_var], 25)
    q3 = np.percentile(df_monthly[max_wave_height_var], 75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    outliers = [value for value in df_monthly[max_wave_height_var].values if value < lower_limit or value > upper_limit]

    # Annotations:
    annotations = [
        (median, f'Median: {round(median, 2)}', 'darkgreen'),
        (mean, f'Mean: {round(mean, 2)}', 'blue'),
        (q1, f'1st quartile: {round(q1, 2)}', 'brown'),
        (q3, f'3th quartile: {round(q3, 2)}', 'brown'),
        (lower_limit, f'L.boundary: {round(lower_limit, 2)}', 'purple'),
        (upper_limit, f'U.boundary: {round(upper_limit, 2)}', 'purple'),
    ]

    # Positioning of the average:
    for value, text, color in annotations:
        if value == mean:
            axes[row, column].text(0.6, value, text, ha='left', va='center', color=color, fontsize=12)
        else:
            axes[row, column].text(1.1, value, text, ha='left', va='center', color=color, fontsize=12)

    for value in outliers:
        axes[row, column].text(0.9, value, f'outlier: {round(float(value), 2)}', ha='right', color='red', fontsize=12)

    axes[row, column].set_xticklabels([]) # The names of the months on the X-axis are deleted

fig.suptitle('Box plots of the maximum wave heights since 1940 to 2024 sampling distributions and their respective statistics', fontsize=22, y=1.05)

plt.tight_layout()
plt.savefig("02 - Box plots of the maximum wave heights since 1940 to 2024 sampling distributions and their respective statistics.png", dpi=300)
plt.show()


# **Frequency distribution**

fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(15, 6)) # cuadrícula de subgráficos
fig.suptitle('Frequency distribution of the maximum wave heights since 1940 to 2024 for each month', fontsize=20, y=1.05)

months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']

# Iteration on the DataFramonth and the axes:
for i, (df_monthly, month) in enumerate(zip(st_Maximum_individual_wave_per_month, months), 1):
    row = (i - 1) // 6  # sub-chart row
    column = (i - 1) % 6  # sub-graph column

    # Histograms:
    df_monthly.hist(ax=axes[row, column], bins=20, color='blue', edgecolor='black')
    axes[row, column].set_title(f'Histogram {month}', fontsize=13)
    axes[row, column].tick_params(axis='both', labelsize=11)
    axes[row, column].grid(False)

plt.tight_layout() # Automatic adjustment to avoid overlapping
plt.savefig("02 - Frequency distribution of the maximum wave heights since 1940 to 2024 for each month (1).png", dpi=300)
plt.show()


# **In order to know the variability of each month with respect to the rest of the months, the frequency diagram is represented with the same scale on the X-axis.**



fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(15, 6)) # subgraph grid
fig.suptitle('Frequency distribution of the maximum wave heights since 1940 to 2024 for each month', fontsize=20, y=1.05)

# Define the months:
months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']

# Iteration on the DataFramonth and the axes:
for i, (df_monthly, month) in enumerate(zip(st_Maximum_individual_wave_per_month, months), 1):
    row = (i - 1) // 6  # we determine the row of the sub-chart
    column = (i - 1) % 6  # we determine the column of the sub-chart

    # Histogram:
    df_monthly.hist(ax=axes[row, column], bins=20, color='blue', edgecolor='black')
    axes[row, column].set_title(f'Histogram {month}', fontsize=13)
    axes[row, column].set_xlim(0, 6)
    axes[row, column].set_ylim(0, 15)
    axes[row, column].tick_params(axis='both', labelsize=11)
    axes[row, column].grid(False)

plt.tight_layout() # Automatic adjustment to avoid overlaps and improve readability
plt.savefig("02 - Frequency distribution of the maximum wave heights since 1940 to 2024 for each month (2).png", dpi=300)
plt.show()


# **Normal probability plots (Q-Q plot)**

fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(15, 6)) # subgraph grid
fig.subplots_adjust(hspace=0.5)  # adjustment of the vertical spacing between the rows of charts
plt.suptitle('Normal probability plots (Q-Q plot) of the maximum wave heights since 1940 for each month', fontsize=20, y=1.05)

# Iterate over the months and draw a QQ-plot on each sub-chart:
for i, (df_monthly, month) in enumerate(zip(st_Maximum_individual_wave_per_month, range(1, 13)), 1):
    row = (i - 1) // 6  # sub-chart row
    column = (i - 1) % 6  # sub-graph column

    stats.probplot(df_monthly[max_wave_height_var], plot=axes[row, column])
    axes[row, column].set_title(f'QQ-Plot {months[month - 1]}', fontsize=12)

    if column == 0:
        axes[row, column].set_ylabel('Z-Score', fontsize=11)  # It is labelled only in the first column

    axes[row, column].tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig("02 - Normal probability plots (Q-Q plot) of the maximum wave heights since 1940 for each month.png", dpi=300)
plt.show()


# **Graph of average maximum wave heights per year.**

# We filter the data to include until 2024.
df_filtered = st_Maximum_individual_wave[
    (st_Maximum_individual_wave.index < '2025-01-01' )
]


# Create a new column ‘Year’ from the datetime index:
df_filtered['Year'] = df_filtered.index.to_series().dt.year

# Group the data by year and calculate the annual mean:
annual_mean = df_filtered.groupby('Year')[max_wave_height_var].mean()

#t-test to check whether maximum wave height mean has increased since 1990:
year_1990=1990-1940 # 50 elements
before_1990 = annual_mean[0:year_1990]
after_1990 = annual_mean[(year_1990):(len(annual_mean)+1)]
annual_mean_vector=annual_mean.to_numpy()

before_1990 = before_1990.to_numpy()
after_1990 = after_1990.to_numpy()

# Perform a one-tailed t-test (alternative hypothesis: after_1990 mean > before_1990 mean)
t_stat, p_value = stats.ttest_ind(before_1990, after_1990, alternative="less")

mean_before_1990 = np.mean(before_1990)
mean_after_1990 = np.mean(after_1990)
# Results
#t_stat, p_value
print()
print("t_stat: " + str(t_stat) + " p_value: " + str(p_value) )
print("mean before_1990: " + str(np.mean(before_1990)) +" mean after_1990: " + str(np.mean(after_1990)) )
print()

"""
The mean maximum wave height is significantly higher from the 1990s to the mid-2020s (2.81 metres) 
than from the 1940s to the 1980s (2.63 metres; t=-5.0617, p=1.2297537077363426e-06).

Since the p-value is much lower than a common threshold of significance (0.05 or even 0.01), 
we can reject the null hypothesis. 
This means that there is significant evidence that the mean maximum wave height after 1990 is higher 
than the mean before 1990.

"""

print ()
# 1. Normality test (Shapiro-Wilk) for both samples
normality_before = stats.shapiro(before_1990)
normality_after = stats.shapiro(after_1990)

# 2. Homogeneity of variances test (Levene)
levene_test = stats.levene(before_1990, after_1990)

print("levene_test:")
print (levene_test)

# 3. t-test
t_test = ttest_ind(before_1990, after_1990, equal_var=True)  # Equal variance is assumed based on Levene's test
print("t_test:")
print (t_test)


# 4. Calculate trend with Kendall's Tau
tau_stat, tau_p_value = stats.kendalltau(annual_mean.index, annual_mean.values)

# Mostrar resultados
print(f"Kendall's Tau stat: {tau_stat} and p_value: {tau_p_value}")
print ()


"""

1) The Shapiro-Wilk test was performed to assess the normality of the data distributions for both time periods. 
The results indicate that the data before 1990 (W = 0.9695, p = 0.2201) and after 1990 (W = 0.9722, p = 0.5061) do not significantly deviate from a normal distribution. 
Since the p-values are greater than the conventional significance level (α = 0.05), we fail to reject the null hypothesis, suggesting that the data can be considered normally distributed.


2) To assess the homogeneity of variances between the two datasets, Levene’s test was performed. 
The results indicate that there is no significant difference in variances between the before 1990 and after 1990 datasets (F = 0.1572, p = 0.6927). 
Since the p-value is greater than the conventional significance level (α = 0.05), we fail to reject the null hypothesis, suggesting that the variances can be considered homogeneous.


3) A two-sample t-test was conducted to compare the means of the before 1990 and after 1990 datasets. 
The results indicate a statistically significant difference between the two groups (t = -5.0617, p < 0.0001). 
Since the p-value is much lower than the conventional significance level (α = 0.05), we reject the null hypothesis, suggesting that the means of the two datasets are significantly different.

4)
A Kendall’s Tau test was conducted to assess the presence of a monotonic trend within the dataset. 
The test results indicate a statistically significant increasing trend (τ = 0.3580, p < 0.0001). 
Since the p-value is well below the conventional significance level (α = 0.05), we reject the null hypothesis, suggesting a meaningful positive trend over time.

"""

# Group the data by year and calculate the annual average:
global_mean = annual_mean.mean()

# Create the time series chart:
plt.figure(figsize=(20, 6))

# Graph of the annual average with dots in yellow:
plt.plot(np.array(annual_mean.index), annual_mean.values, marker='o', linestyle='-', color='b', label='Annual Mean')
plt.scatter(annual_mean.index, annual_mean.values, color='orange')

# Overall average line:
plt.axhline(global_mean, color='orange', linestyle='--', label=f'Global Mean: {global_mean:.2f}')

# Chart customisations:
plt.title('Average maximum wave heights since 1940 to 2024', fontsize=20, pad=15)
plt.ylabel('Annual maximum wave height (metres)', fontsize=12)

# Change the border colour of the chart to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')

plt.xticks(np.arange(np.array(annual_mean.index)[0], np.array(annual_mean.index)[-1], 10), fontsize=14)
plt.yticks(fontsize=14)
# Add legend:
plt.legend(loc='upper left', fontsize=14, frameon=False)
# Show the graph:
plt.savefig("02 - Average maximum wave heights since 1940 to 2024.png", dpi=300)
plt.show()




###########################################
# #### **4.2. Correlation analysis of the variable ‘Maximum individual wave height’ with other climatic variables.**
###########################################
# **Definition of climatic variables that may influence the time series and 'Maximum individual wave height'.**

 # Rerorder the DataFrame
columns = [max_wave_height_var] + [col for col in df_data.columns if col != max_wave_height_var]
df_correlation = df_data[columns] 


# **We need to standardise the variables so that they are all in the same units. The z-score standardisation is used to transform variables with mean 0 and standard deviation 1.**

from sklearn.preprocessing import StandardScaler

# Init the StandardScaler object:
scaler = StandardScaler()

# Normalisation of variables:
df_standardised_correlation = scaler.fit_transform(df_correlation)
df_standardised_correlation = pd.DataFrame(df_standardised_correlation, columns=df_correlation.columns, index=df_correlation.index) # the normalised data in a DataFrame


# **The variables with the highest linear correlation coefficient with our study variable are obtained.**

# Calculate the linear correlation coefficients:
linear_correlation = df_standardised_correlation.corr()

# Defining the values of correlation coefficients:
lower_coefficient = -0.75
upper_coefficient = 0.75

# Print the variables with a correlation coefficient outside the desired range and create the scatter diagram:
print(f'Correlation coefficient between the variable \033[1mMaximum individual wave height\033[0m and the variables:\n')

i = 0 # counter

for j in range(1, len(linear_correlation)):
    if linear_correlation.iloc[0, j] <= lower_coefficient or linear_correlation.iloc[0, j] >= upper_coefficient:
        variables = linear_correlation.index[0], linear_correlation.columns[j]
        i += 1 # aumnentamos en uno el contador de variables que cumplen la condición
        print(f'\t{i}. \033[1m{variables[1]}\033[0m: {linear_correlation.iloc[0, j]}')



# Calculate the linear correlation coefficients:
linear_correlation = df_standardised_correlation.corr()

# Defining the values of correlation coefficients:
lower_coefficient = -0.75
upper_coefficient = 0.75

i = 0 # counter

# Print the variables with a correlation coefficient that have strong correlations (above 0.75 or beyond -0.75) and create the scatter diagram:
print('Correlation coefficient between the variable \033[1mMaximum individual wave height- media\033[0m and the variables:\n')

for j in range(1, len(linear_correlation)):
    if linear_correlation.iloc[0, j] <= lower_coefficient or linear_correlation.iloc[0, j] >= upper_coefficient:
        variables = linear_correlation.index[0], linear_correlation.columns[j]
        i += 1 # we increase by one the counter of variables that meet the condition
        print(f'\t{i}. \033[1m{variables[1]}\033[0m: {linear_correlation.iloc[0, j]:.2f}')

        fig, ax = plt.subplots() # Creating a figure for the scatter diagram
        ax.scatter(x=df_correlation[linear_correlation.index[0]], y=df_correlation[linear_correlation.index[j]], label=f'{variables[1]}')

        # Add labels and legend to the scatter diagram:
        ax.set_xlabel('Maximum individual wave height in metres (average)')
        ax.set_ylabel(linear_correlation.index[j].split(' - ')[0] + ' (average)')
        title =  'Scatter correlation plot of Maximum individual wave height in metres (average) vs ' + linear_correlation.index[j].split(' - ')[0] + ' (average)'
        ax.set_title(title)
        plt.savefig("02 - " + title + ".png")
        plt.show()


# Graph/Plot:
fig, ax = plt.subplots()
ax.scatter(x = df_correlation[max_wave_height_var], y = df_correlation['Sea surface temperature'])

# Set labels and title:
plt.title('Maximum individual wave height vs Sea surface temperature', fontsize=16, pad=15)
plt.xlabel('Maximum individual wave height (metres)', fontsize=10)
plt.ylabel('Sea surface temperature (Kelvin degrees)', fontsize=10)

# Change the border colour of the chart to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')

# Show the plot:
plt.savefig("02 - Maximum individual wave height vs Sea surface temperature.png", dpi=300)
plt.show()


