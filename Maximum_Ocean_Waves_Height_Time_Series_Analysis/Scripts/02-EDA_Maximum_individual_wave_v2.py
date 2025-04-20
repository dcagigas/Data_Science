import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import ttest_ind
import seaborn as sns


###########################################
# ### 3.Extraction and transformation of the “Maximum individual wave height” time series. Data extraction.
###########################################

final_csv_file_path = 'final_dataset.csv'
range_limit_of_selected_data = '2024-12-31' # ONLY TAKE DATA UNTIL 2024
max_wave_height_var= 'Maximum individual wave height'

df_data = pd.read_csv(final_csv_file_path, index_col=0)

# Sort by index
df_data = df_data.sort_index()

# Define the time series with the values of the 'Maximun individual wave height (average per month)':
#st_Maximum_individual_wave = df_Maximum_individual_wave[[max_wave_height_var]]
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

fig, axes = plt.subplots(nrows=1, ncols=12, figsize=(20, 5))
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

# Define figure and axes for histograms with KDE
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(18, 8))
fig.suptitle('Frequency distribution of the maximum wave heights (1940-2024) for each month', fontsize=16, y=1.05)

# Define the months
months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 
          'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']

# Plot histograms with density curves
for i, (df_monthly, month) in enumerate(zip(st_Maximum_individual_wave.resample('M').mean().groupby(st_Maximum_individual_wave.index.month), months), 1):
    row = (i - 1) // 6  # Determine the row of the sub-chart
    column = (i - 1) % 6  # Determine the column of the sub-chart

    # Extract monthly data and remove NaN values
    data = df_monthly[1].to_numpy().ravel()
    data = data[~np.isnan(data)]

    # Perform Shapiro-Wilk Test for normality
    stat, p_value = stats.shapiro(data)

    # Histogram with KDE
    #sns.histplot(df_monthly[1].to_numpy().ravel(), bins=20, kde=True, line_kws={'color':'red', 'linewidth': 2}, color='blue', edgecolor='black', ax=axes[row, column])
    sns.histplot(data, bins=20, kde=True, line_kws={'color': 'red', 'linewidth': 2}, color='blue', edgecolor='black', ax=axes[row, column])

    axes[row, column].set_title(f'{month}', fontsize=13)
    axes[row, column].set_xlabel("Wave Height")
    axes[row, column].set_ylabel("Frequency")

    # Add p-value annotation on top right
    normality_status = "Normal" if p_value >= 0.05 else "Not Normal"
    axes[row, column].text(0.95, 0.95, f'p = {p_value:.3f}\n{normality_status}', 
                           transform=axes[row, column].transAxes, ha='right', va='top', fontsize=10,
                           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.tight_layout()  # Adjust layout to avoid overlapping
plt.savefig("02 - Frequency distribution of the maximum wave heights since 1940 to 2024 for each month (1).png", dpi=300)
plt.show()


# Second figure with fixed axis limits
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(18, 8))
fig.suptitle('Frequency distribution of the maximum wave heights (1940-2024) for each month with fixed scale', fontsize=16, y=1.05)

# Plot histograms with KDE and fixed axis limits
for i, (df_monthly, month) in enumerate(zip(st_Maximum_individual_wave.resample('M').mean().groupby(st_Maximum_individual_wave.index.month), months), 1):
    row = (i - 1) // 6  # Determine the row of the sub-chart
    column = (i - 1) % 6  # Determine the column of the sub-chart

    # Extract monthly data and remove NaN values
    data = df_monthly[1].to_numpy().ravel()
    data = data[~np.isnan(data)]

    # Perform Shapiro-Wilk Test for normality
    stat, p_value = stats.shapiro(data)

    # Histogram with KDE
    #sns.histplot(df_monthly[1].to_numpy().ravel(), bins=20, kde=True, line_kws={'color':'red', 'linewidth': 2}, color='blue', edgecolor='black', ax=axes[row, column])
    sns.histplot(data, bins=20, kde=True, line_kws={'color': 'red', 'linewidth': 2}, color='blue', edgecolor='black', ax=axes[row, column])
    
    axes[row, column].set_title(f'{month}', fontsize=13)
    axes[row, column].set_xlabel("Wave Height")
    axes[row, column].set_ylabel("Frequency")
    axes[row, column].set_xlim(0, 6)  # Fixed X-axis limit
    axes[row, column].set_ylim(0, 15)  # Fixed Y-axis limit

    # Add p-value annotation on top right
    normality_status = "Normal" if p_value >= 0.05 else "Not Normal"
    axes[row, column].text(0.95, 0.95, f'p = {p_value:.3f}\n{normality_status}', 
                           transform=axes[row, column].transAxes, ha='right', va='top', fontsize=10,
                           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.tight_layout()  # Adjust layout to avoid overlapping
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

# check the trend of the two time series

from pymannkendall import original_test

# Target significance level (alpha):
significance_level = 0.05

# Perform the Mann-Kendall test:
mann_kendall_results_before_1990 = original_test(before_1990)
mann_kendall_results_after_1990 = original_test(after_1990)

# Print additional details of the results:
print('Mann-Kendall test:')
print(f'Test statistician before 1990: {round(mann_kendall_results_before_1990.z, 4)}')
print(f'p-value before 1990: {mann_kendall_results_before_1990.p}')
print()
print('Mann-Kendall test:')
print(f'Test statistician after 1990: {round(mann_kendall_results_after_1990.z, 4)}')
print(f'p-value after 1990: {mann_kendall_results_after_1990.p}')
print()

"""
Results Before 1990: 
The results indicate a statistically significant increasing trend in the data before 1990. 
The low p-value (0.0095) confirms that the observed trend is unlikely to be due to random variability.
Results After 1990:
After 1990, the analysis did not detect a significant trend. 
The high p-value (0.6701) suggests that any potential changes in the data are not statistically distinguishable from random fluctuations.

The Mann-Kendall test results suggest a clear increasing trend in the data before 1990, but no statistically significant trend after 1990. 
This shift could indicate a stabilization of the observed variable or external factors influencing its behavior over time.
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
Since the p-values are greater than the conventional significance level (α = 0.05), we fail to reject the null hypothesis, 
# suggesting that the data can be considered normally distributed.


2) To assess the homogeneity of variances between the two datasets, Levene’s test was performed. 
The results indicate that there is no significant difference in variances between the before 1990 and after 1990 datasets (F = 0.1572, p = 0.6927). 
Since the p-value is greater than the conventional significance level (α = 0.05), we fail to reject the null hypothesis, 
# suggesting that the variances can be considered homogeneous.


3) A two-sample t-test was conducted to compare the means of the before 1990 and after 1990 datasets. 
The results indicate a statistically significant difference between the two groups (t = -5.0617, p < 0.0001). 
Since the p-value is much lower than the conventional significance level (α = 0.05), we reject the null hypothesis, suggesting that the means of the two datasets are significantly different.

4)
A Kendall’s Tau test was conducted to assess the presence of a monotonic trend within the dataset. 
The test results indicate a statistically significant increasing trend (τ = 0.3580, p < 0.0001). 
Since the p-value is well below the conventional significance level (α = 0.05), we reject the null hypothesis, 
# suggesting a meaningful positive trend over time.

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
plt.title('Average maximum wave heights since 1940 to 2024', fontsize=22, pad=15)
plt.ylabel('Annual maximum wave height (metres)', fontsize=18)

# Change the border colour of the chart to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')

#plt.xticks(np.arange(1940, 2021, 10), fontsize=14)
plt.xticks(np.arange(np.array(annual_mean.index)[0], np.array(annual_mean.index)[-1], 10), fontsize=18)
plt.yticks(fontsize=18)
# Add legend:
plt.legend(loc='upper left', fontsize=18, frameon=False)
# Show the graph:
plt.savefig("02 - Average maximum wave heights since 1940 to 2024.png", dpi=300)
plt.show()



"""
###########################################
# #### **4.2. Correlation analysis of the variable ‘Maximum individual wave height’ with other climatic variables.**
###########################################

# Compute correlation using Pearson, Spearman, and Kendall methods
correlation_pearson = df_data.corr(method='pearson')[max_wave_height_var]
correlation_spearman = df_data.corr(method='spearman')[max_wave_height_var]
correlation_kendall = df_data.corr(method='kendall')[max_wave_height_var]

# Combine results into a single DataFrame
correlation_results = pd.DataFrame({
    'Pearson': correlation_pearson,
    'Spearman': correlation_spearman,
    'Kendall': correlation_kendall
}).drop(index=max_wave_height_var)  # Drop the target variable itself

# Define correlation strength categories
def categorize_correlation(value):
    if abs(value) >= 0.7:
        return "Strong"
    elif abs(value) >= 0.4:
        return "Moderate"
    elif abs(value) >= 0.2:
        return "Weak"
    else:
        return "None"

# Apply categorization
correlation_results['Pearson Category'] = correlation_results['Pearson'].apply(categorize_correlation)
correlation_results['Spearman Category'] = correlation_results['Spearman'].apply(categorize_correlation)
correlation_results['Kendall Category'] = correlation_results['Kendall'].apply(categorize_correlation)

# Sort correlation results by Pearson correlation in descending order
correlation_results_sorted = correlation_results.sort_values(by='Pearson', ascending=False)

# Display the results
#import ace_tools as tools
#tools.display_dataframe_to_user(name="Correlation Analysis", dataframe=correlation_results)

# Define the path for saving the correlation results
correlation_results_csv_path = "02 - correlation_results_sorted.csv"

# Save the sorted correlation results to a CSV file
correlation_results_sorted.to_csv(correlation_results_csv_path)


# Reorder variables based on Pearson correlation strength
ordered_vars = correlation_results_sorted.index.insert(0, max_wave_height_var)  # Insert target variable at the top

# Compute correlation matrices for Pearson and Spearman
correlation_matrix_pearson = df_data[ordered_vars].corr(method="pearson")
correlation_matrix_spearman = df_data[ordered_vars].corr(method="spearman")

# Save and display Pearson heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix_pearson, annot=True, cmap="coolwarm", fmt=".2f",
    linewidths=0.5, cbar=True
)
plt.title("Pearson Correlation Heatmap")
plt.xticks(rotation=45, ha='right')

# Highlight the target variable
idx = ordered_vars.get_loc(max_wave_height_var)
plt.gca().add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor='black', lw=3))

# Save Pearson heatmap
pearson_heatmap_path = "02 - pearson_correlation_heatmap.png"
plt.savefig(pearson_heatmap_path, bbox_inches="tight", dpi=300)
plt.show()

# Save and display Spearman heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix_spearman, annot=True, cmap="coolwarm", fmt=".2f",
    linewidths=0.5, cbar=True
)
plt.title("Spearman Correlation Heatmap")
plt.xticks(rotation=45, ha='right')

# Highlight the target variable
plt.gca().add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor='black', lw=3))

# Save Spearman heatmap
spearman_heatmap_path = "02 - spearman_correlation_heatmap.png"
plt.savefig(spearman_heatmap_path, bbox_inches="tight", dpi=300)
plt.show()
"""

