import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import levene, bartlett
from statsmodels.stats.diagnostic import het_arch

variance_results_path = "variance_analysis_results.csv"
# Load the dataset
file_path = "final_dataset.csv"
df = pd.read_csv(file_path)

# Ensure that the date column exists and convert it to datetime format
df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Define the variable name
max_wave_height_var = 'Maximum individual wave height'

# Ensure the column exists before selection
if max_wave_height_var in df.columns:
    df = df[[max_wave_height_var]]
else:
    raise KeyError(f"Column '{max_wave_height_var}' not found in the dataset.")

# Ensure data is aligned to monthly frequency
df = df.asfreq('MS')

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[max_wave_height_var], label="Maximum Wave Height")
plt.xlabel("Year")
plt.ylabel("Wave Height (m)")
plt.title("Time Series of Maximum Wave Height (1940-2024)")
plt.legend()
plt.savefig("03 - Time Series of Maximum Wave Height (1940-2024).png", dpi=300)
plt.show()

# Perform a rolling variance test
rolling_var = df[max_wave_height_var].rolling(window=60).var()  # 5-year rolling variance

plt.figure(figsize=(12, 6))
plt.plot(df.index, rolling_var, label="Rolling Variance (5 years)")
plt.xlabel("Year", fontsize=14)
plt.ylabel("Variance", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Rolling Variance of Maximum Wave Height", fontsize=14)
plt.legend(fontsize=14)
plt.savefig("03 - Rolling Variance of Maximum Wave Height.png", dpi=300)
plt.show()

# Perform Levene's test for variance homogeneity
midpoint = len(df) // 2
first_half = df[max_wave_height_var].iloc[:midpoint]
second_half = df[max_wave_height_var].iloc[midpoint:]

levene_stat, levene_p = levene(first_half, second_half)
print(f"Levene's Test Statistic: {levene_stat:.4f}, p-value: {levene_p:.4f}")

"""
The result of Levene's test for homogeneity of variance is:

Levene's statistic: 4.47
p-value: 0.0347
Since the p-value is less than 0.05, we can reject the null hypothesis of homogeneous variance. 
This indicates that the variance is not constant over time, suggesting the presence of heteroscedasticity in the maximum wave height series.
"""

# Perform Ljung-Box test for white noise
ljung_box_results = acorr_ljungbox(df[max_wave_height_var].dropna(), lags=[10, 20, 30], return_df=True)
print("Ljung-Box Test Results:")
print(ljung_box_results)

"""
The results of the Ljung-Box test to assess whether the wave height series is white noise are:

For 10 delays: p = 0.0
For 20 lags: p = 0.0
For 30 lags:  p = 0.0

Since all values of p values are 0.0 (less than 0.05), we can reject the null hypothesis that the series is white noise. 
This means that the series is autocorrelated and not purely random.

This result is important because it indicates that the series has structure and predictable patterns, 
suggesting that a model such as SARIMA may be suitable for analysis.

"""

# Split into two halves for variance testing
midpoint = len(df) // 2
first_half = df.iloc[:midpoint]
second_half = df.iloc[midpoint:]

# **1. Bartlett’s Test (Sensitive to Normality)**
bartlett_stat, bartlett_p = bartlett(first_half[max_wave_height_var], second_half[max_wave_height_var])

# **2. Brown-Forsythe Test (More Robust)**
levene_stat_bf, levene_p_bf = levene(first_half[max_wave_height_var], second_half[max_wave_height_var], center='median')

# **3. Engle’s ARCH Test (Time-Varying Variance)**
arch_test = het_arch(df[max_wave_height_var].dropna())
arch_stat, arch_p_value = arch_test[:2]

# Print Test Results
#print(f"Bartlett’s Test: Statistic = {bartlett_stat:.4f}, p-value = {bartlett_p:.4f}")
#print(f"Brown-Forsythe Test: Statistic = {levene_stat_bf:.4f}, p-value = {levene_p_bf:.4f}")
#print(f"Engle’s ARCH Test: Statistic = {arch_stat:.4f}, p-value = {arch_p_value:.4f}")

# Interpretation:
"""
- **Bartlett’s Test**: If p < 0.05, variance is changing over time (but assumes normality).
- **Brown-Forsythe Test**: If p < 0.05, variance is changing over time (better for non-normal data).
- **Engle’s ARCH Test**: If p < 0.05, variance is time-dependent, confirming heteroskedasticity.

                  Test   Statistic        p-value
0      Bartlett’s Test    4.852615   2.760425e-02
1  Brown-Forsythe Test    4.471959   3.469776e-02
2    Engle’s ARCH Test  494.524539  6.528822e-100


Both Bartlett's & Brown-Forsythe Tests Reject Homogeneous Variance

The low p-values (< 0.05) mean we reject the null hypothesis of equal variance.
This confirms that variance has increased over time.
Engle’s ARCH Test Confirms Time-Varying Variance (Heteroskedasticity)

The extremely low p-value (~0.000) strongly suggests variance changes over time.
This is typical of financial data and climate-related time series.
This means plain ARIMA/SARIMA will not work optimally.
"""

# Save results to a DataFrame
variance_analysis_results = pd.DataFrame({
    "Test": ["Bartlett’s Test", "Brown-Forsythe Test", "Engle’s ARCH Test"],
    "Statistic": [bartlett_stat, levene_stat_bf, arch_stat],
    "p-value": [bartlett_p, levene_p_bf, arch_p_value]
})

# Save to CSV
variance_analysis_results.to_csv(variance_results_path, index=False)
print()
print (variance_analysis_results)
