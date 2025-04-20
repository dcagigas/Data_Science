import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

###########################################
# ### ** 6. Forecasting models of classical statistics **
###########################################

# #### * *6.1. Deterministic trend analysis **
# 
# An ordinary least squares linear regression model (OLS) is fitted.


final_csv_file_path = 'final_dataset.csv'
range_limit_of_selected_data = '2024-12-31' # ONLY TAKE DATA UNTIL 2024
max_wave_height_var= 'Maximum individual wave height'

df_data = pd.read_csv(final_csv_file_path, index_col=0)

# Sort by index
df_data = df_data.sort_index()

# Define the time series with the values of the 'Maximun individual wave height (average per month)':
st_Maximum_individual_wave = df_data[[max_wave_height_var]]

# Sort if not sorted:
st_Maximum_individual_wave.sort_index(inplace=True)

# Convert date to 'timestamp'' format:
st_Maximum_individual_wave.index = pd.to_datetime(st_Maximum_individual_wave.index, format='%Y-%m-%d')

# Define the frequency of the time series as monthly
st_Maximum_individual_wave = st_Maximum_individual_wave.asfreq('MS')

# IMPORTANT: GET DATA JUST UNTIL A DATE.
st_Maximum_individual_wave = st_Maximum_individual_wave[st_Maximum_individual_wave.index <= range_limit_of_selected_data]


# Ensure that the index of the DataFrame is of type datetime:
st_Maximum_individual_wave.index = pd.to_datetime(st_Maximum_individual_wave.index)

# Obtain time values (x-axis) and time series (y-axis):
time = np.arange(len(st_Maximum_individual_wave))
time_series = st_Maximum_individual_wave[max_wave_height_var].values

# Create the linear regression model:
model = LinearRegression()

# Fit the model to the data:
model.fit( time.reshape(-1, 1), time_series)

# Predicting trend values:
trend = model.predict(time.reshape(-1, 1))

# Extend the prediction to 10 future values:
future_time = np.arange(len(st_Maximum_individual_wave) + 50)
future_trend = model.predict(future_time.reshape(-1, 1))

# Create a date index including future 50 periods:
frequency = st_Maximum_individual_wave.index.freq or pd.infer_freq(st_Maximum_individual_wave.index)
future_index = pd.date_range(start=st_Maximum_individual_wave.index[0], periods=len(future_time), freq=frequency)

# Plot the time series and deterministic trend:
plt.figure(figsize=(20, 6))

plt.plot(np.array(st_Maximum_individual_wave.index), time_series, label='Original Time Series', color='lightblue')
plt.plot(np.array(future_index), future_trend, label='Linear Regression forecasting', color='darkorange')

plt.ylabel('Maximum Wave Height (metres)', fontsize=18)
plt.title('Linear Regression Analysis', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)

# RMSE and R² calc:
rmse = np.sqrt(mean_squared_error(time_series, trend))
r2 = r2_score(time_series, trend)

# Calculate AIC and BIC using statsmodels:
time_series = sm.add_constant(time)
sm_model = sm.OLS(st_Maximum_individual_wave[max_wave_height_var], time_series).fit()
aic = sm_model.aic
bic = sm_model.bic

print('OLS MODEL VALIDATION PARAMETERS WITH OTHER FORECASTING MODELS:')
print(f'RMSE: {round(rmse, 4)}')
print(f'R2: {round(r2, 4)}')
print(f'AIC: {round(aic, 4)}')
print(f'BIC: {round(bic, 4)}')

# Create a DataFrame to display results in a table:
results = pd.DataFrame({
    'Train': [rmse, r2, aic, bic]
}, index=['RMSE', 'R^2', 'AIC', 'BIC'])

print()
print('OLS Model results:')
print(results)
results.to_csv('OLS Model results.csv')
print()

# TESTS OF HETEROSCEDASTICITY OF THE ERRORS:
print('Heteroscedasticity tests of the residuals between observed and predicted values:')

# Breusch-Pagan test:
residuals = sm_model.resid
X = pd.DataFrame({'constante': np.ones(len(time)), 'tiempo': time})
bp_test = het_breuschpagan(residuals, X)
bp_labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
bp_results = dict(zip(bp_labels, bp_test))

print('\nBreusch-Pagan test:')
for key in bp_results:
    print(f'{key}: {bp_results[key]:.4f}')

"""
Breusch-Pagan test:
LM Statistic: 5.9801
LM-Test p-value: 0.0145
F-Statistic: 6.0036
F-Test p-value: 0.0144
Conclusion: there is evidence of heteroskedasticity in the model errors
"""


# White test:
white_test = het_white(residuals, X)
white_labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
white_results = dict(zip(white_labels, white_test))

print('\nWhite test:')
for key in white_results:
    print(f'{key}: {white_results[key]:.4f}')

"""
White test:
Test Statistic: 6.2439
Test Statistic p-value: 0.0441
F-Statistic: 3.1319
F-Test p-value: 0.0441
Conclusion: there is evidence of heteroskedasticity in the model

"""

# Change the border color of the chart to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')

plt.savefig("04 - Linear Regression Analysis.png", dpi=300)
plt.show()



# Obtain the model summary:
print()
print('--------- OLS model summary ----------')
summary = sm_model.summary()
print(summary)
print()


slope = model.coef_[0]
intercept = model.intercept_

# Linear prediction model:
prediction_model = lambda x: slope * x + intercept

prediction_model_str = f'y = {slope:.5f}x + {intercept:.5f}' # in mathematical equation format
print(prediction_model_str)




# Function to convert month and year to a time value:
def convert_month_and_year_to_time_value(month, year):
    return (year - st_Maximum_individual_wave.index.year.min()) * 12 + month - 1

# Month and year for prediction:
prediction_month = 1  # For example, june
prediction_year = 2025  # For example, year 2024

# Converts the month and year to a time value:
prediction_time_value = convert_month_and_year_to_time_value(prediction_month, prediction_year)

# Predicts value for specified month and year:
month_and_year_forecast = prediction_model(prediction_time_value)
print(f'The forecast for {prediction_month}/{prediction_year} is: {month_and_year_forecast}')




# #### **6.2. Holt-Winters double smoothing methods.**


from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, r2_score

# Split data into training and test sets:
train_size = int(len(st_Maximum_individual_wave) * 0.85)
train, test = st_Maximum_individual_wave.iloc[:train_size], st_Maximum_individual_wave.iloc[train_size:]

# Fitting the Holt-Winters double-smoothing model to the training set:
model_hw = ExponentialSmoothing(train[max_wave_height_var],
                                 trend='additive', seasonal='multiplicative', seasonal_periods=12)
model_hw_fitted = model_hw.fit()

# Making predictions on training and test sets:
train_forecasts = model_hw_fitted.fittedvalues
test_forecasts = model_hw_fitted.predict(start=test.index[0], end=test.index[-1])

# Calculate RMSE and R² for the training and test sets:
rmse_train = np.sqrt(mean_squared_error(train[max_wave_height_var], train_forecasts))
rmse_test = np.sqrt(mean_squared_error(test[max_wave_height_var], test_forecasts))
r2_train = r2_score(train[max_wave_height_var], train_forecasts)
r2_test = r2_score(test[max_wave_height_var], test_forecasts)

# Calculate AIC and BIC for training set:
n_obs_train = len(train)
k_params_train = len(model_hw_fitted.params)
aic_train = n_obs_train * np.log(rmse_train**2) + 2 * k_params_train
bic_train = n_obs_train * np.log(rmse_train**2) + k_params_train * np.log(n_obs_train)

# Calculate AIC and BIC for test set:
n_obs_test = len(test)
k_params_test = len(model_hw_fitted.params)
aic_test = n_obs_test * np.log(rmse_test**2) + 2 * k_params_test
bic_test = n_obs_test * np.log(rmse_test**2) + k_params_test * np.log(n_obs_test)

# Create a DataFrame to display results in a table:
results = pd.DataFrame({
    'Train': [rmse_train, r2_train, aic_train, bic_train],
    'Test': [rmse_test, r2_test, aic_test, bic_test]
}, index=['RMSE', 'R^2', 'AIC', 'BIC'])

print()
print('Holt-Winters Double Smoothing Model results:')
print(results)
results.to_csv('Holt-Winters Double Smoothing Model results.csv')
print()

# Visualisation of the time series and forecasts:
plt.figure(figsize=(20, 6))
plt.plot(np.array(train.index), train[max_wave_height_var].values, label='Train', color='lightblue')
plt.plot(np.array(test.index), test[max_wave_height_var].values, label='Test', color='lightgreen')
plt.plot(np.array(train.index), np.array(train_forecasts), label='Train Predictions', color='darkorange')
plt.plot(np.array(test.index), np.array(test_forecasts), label='Test Predictions', color='orange')

plt.title('Holt-Winters Double Smoothing Model', fontsize=22)
plt.ylabel('Maximum Wave Height (metres)', fontsize=18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)

# Add vertical line at the end of the training data:
plt.axvline(x=train.index[-1], color='gray', linestyle='--')

# Change the border colour of the chart to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')

plt.savefig("04 - Holt-Winters Double Smoothing Model.png", dpi=300)
plt.show()


# **Graphical representation of the test set and its predictions.**


# Visualisation of the time series and forecasts:
plt.figure(figsize=(20, 6))
plt.plot(np.array(test.index), test[max_wave_height_var].values, label='Test', color='lightgreen')
plt.plot(np.array(test.index), np.array(test_forecasts), label='Test Predictions', color='orange')

plt.title('Holt-Winters Double Smoothing Model - Test and forecasts', fontsize=22)
plt.ylabel('Maximum wave height (metres)', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)

# Change the border colour of the chart to orange:
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('orange')

plt.savefig("04 - Holt-Winters Double Smoothing Model - Test and forecasts.png", dpi=300)
plt.show()


# **Prediction of the wave height on a specific date.**


# Forecast a specific date:
forecast_date = '2025-01-01'  # Change this to the date you want to predict
specific_forecast = model_hw_fitted.predict(start=forecast_date, end=forecast_date)

print(f'Forecasting for {forecast_date}: {specific_forecast.values[0]}')

