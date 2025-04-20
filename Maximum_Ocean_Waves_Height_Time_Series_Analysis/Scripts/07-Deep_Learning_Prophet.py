import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import joblib
import os
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score

###########################################
# ### **9. Machine Learning predictive model with Prophet**
###########################################

# ************************************************************
# **9.5. Facebook Prophet  **
# ************************************************************

df_data = pd.read_csv('final_dataset.csv', index_col=0)
range_limit_of_selected_data = '2024-12-31'  # ONLY TAKE DATA UNTIL 2024
df_data = df_data.sort_index()

Prophet_model_path = "models/Prophet_model.h5"
csv_filename = "Prophet_evaluation_metrics.csv"

max_wave_height_var = 'Maximum individual wave height'

# Define the time series with the values of the variable 'Maximum individual wave height - mean (by month)'
df_Maximum_individual_wave = df_data[[max_wave_height_var]]

# Ensure that the index of the DataFrame is of type datetime
df_Maximum_individual_wave.index = pd.to_datetime(df_Maximum_individual_wave.index)

# Ensure data is aligned to monthly frequency
df_Maximum_individual_wave = df_Maximum_individual_wave.asfreq('MS')

# Keep data only until the specified date
df_Maximum_individual_wave = df_Maximum_individual_wave[df_Maximum_individual_wave.index <= range_limit_of_selected_data]

# Convert data to Prophet format (Prophet requires 'ds' for dates and 'y' for target variable)
df_prophet = df_Maximum_individual_wave.reset_index().rename(columns={"index": "ds", max_wave_height_var: "y"})

# Split into train (85%) and test (15%)
train_size = int(len(df_prophet) * 0.85)
train_data, test_data = df_prophet.iloc[:train_size], df_prophet.iloc[train_size:]

# Function to save the model
def save_model_hdf5(model, filepath):
    temp_file = filepath + "_temp.pkl"
    joblib.dump(model, temp_file)

    with h5py.File(filepath, 'w') as h5f:
        with open(temp_file, 'rb') as f:
            model_data = f.read()
            h5f.create_dataset('model', data=np.void(model_data))

    os.remove(temp_file)
    print(f"Model saved to {filepath}")

# Function to load the model
def load_model_hdf5(filepath):
    with h5py.File(filepath, 'r') as h5f:
        model_data = h5f['model'][()]

    temp_file = filepath + "_temp.pkl"
    with open(temp_file, 'wb') as f:
        f.write(model_data.tobytes())

    model = joblib.load(temp_file)
    os.remove(temp_file)
    print(f"Model loaded from {filepath}")

    return model

# Check if the model file exists
if os.path.exists(Prophet_model_path):
    # Load the model if it exists
    Prophet_model = load_model_hdf5(Prophet_model_path)
else:
    # Initialize and train the Prophet model
    Prophet_model = Prophet(
        yearly_seasonality=True,   # Account for yearly seasonality
        weekly_seasonality=False,  # Not needed for monthly data
        daily_seasonality=False,   # Not needed for monthly data
        seasonality_mode="additive",  # Can be changed to "multiplicative" if necessary
        changepoint_prior_scale=0.05,  # Adjust trend flexibility (try values: 0.01, 0.05, 0.1)
        seasonality_prior_scale=5      # Adjust seasonality influence (try values: 1, 5, 10)
    )
    
    Prophet_model.fit(train_data)
    save_model_hdf5(Prophet_model, Prophet_model_path)

# Make future predictions (test period length)
future = Prophet_model.make_future_dataframe(periods=len(test_data), freq='MS')
forecast = Prophet_model.predict(future)

# Extract only the predicted values for the test period
y_train_pred = Prophet_model.predict(train_data[['ds']])['yhat']
y_test_pred = forecast.iloc[train_size:]['yhat'].values

# Calculate performance metrics for train set
rmse_train = np.sqrt(mean_squared_error(train_data['y'], y_train_pred))
r2_train = r2_score(train_data['y'], y_train_pred)

# Calculate performance metrics for test set
rmse_test = np.sqrt(mean_squared_error(test_data['y'], y_test_pred))
r2_test = r2_score(test_data['y'], y_test_pred)

# Compute AIC and BIC (since Prophet does not provide these, approximate them)
def compute_aic_bic(y_true, y_pred, num_params):
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    sigma2 = sse / n

    aic = n * np.log(sigma2) + 2 * num_params
    bic = n * np.log(sigma2) + num_params * np.log(n)

    return aic, bic

aic_train, bic_train = compute_aic_bic(train_data['y'], y_train_pred, 12 + 1)
aic_test, bic_test = compute_aic_bic(test_data['y'], y_test_pred, 12 + 1)

# Print Prophet results
evaluation_metrics_df = pd.DataFrame({
    "Set": ["Training", "Test"],
    "RMSE": [rmse_train, rmse_test],
    "R²": [r2_train, r2_test],
    "AIC": [aic_train, aic_test],
    "BIC": [bic_train, bic_test]
})

print("\nProphet results:")
print(evaluation_metrics_df)

# Prophet is still the best model for forecasting maximum wave height
"""
Prophet results:
        Set      RMSE        R²          AIC          BIC
0  Training  0.489125  0.662048 -1214.048949 -1152.103442
1      Test  0.530788  0.687353  -167.818143  -128.422450
"""

# Save evaluation metrics to a CSV file
evaluation_metrics_df.to_csv(csv_filename, index=False)


plt.figure(figsize=(20, 6))
plt.plot(train_data['ds'], train_data['y'], label='Train', color='lightblue')
plt.plot(test_data['ds'],  test_data['y'], label='Test', color='lightgreen')
#  Add vertical line at the end of the training data:
#plt.axvline(x=test_data['ds'].iloc[0], color='gray', linestyle='--')
plt.axvline(x=pd.to_datetime(test_data['ds'].iloc[0]), color='gray', linestyle='--')
plt.plot(test_data['ds'], y_test_pred, label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height (metres)', fontsize=18)
plt.title("Actual vs Predicted Values (Prophet)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)
plt.savefig("Prophet_Regressor.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(20, 6))
plt.plot(test_data['ds'],  test_data['y'], label='Test', color='lightgreen')
plt.plot(test_data['ds'], y_test_pred, label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height (metres)', fontsize=18)
plt.title('Prophet prediction using Top Features', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)
plt.savefig("Prophet_test_time_series_and_forecasts.png", dpi=300, bbox_inches="tight")
plt.show()


# Prophet trend and seasonality components with added labels
fig2 = Prophet_model.plot_components(forecast)

# Customize each subplot to improve readability
ax1 = fig2.axes[0]  # Trend component
ax1.set_title("Trend Over Time", fontsize=14)
ax1.set_ylabel("Wave Height Trend (m)", fontsize=12)
ax1.set_xlabel('')

ax1.legend(["Trend"])

ax2 = fig2.axes[1]  # Yearly seasonality component
ax2.set_title("Yearly Seasonality Pattern", fontsize=14)
ax2.set_ylabel("Effect on Wave Height (m)", fontsize=12)
ax2.set_xlabel('')
ax2.set_xticklabels(['January', 'March', 'May', 'July', 
                    'September', 'November', 'January'])
ax2.legend(["Seasonality"])

# Add a horizontal line at y=0 to distinguish positive and negative seasonal effects
ax2.axhline(y=0, color='orange', linestyle='--', linewidth=1)

# If there's a third subplot (holidays, etc.), add a title
if len(fig2.axes) > 2:
    ax3 = fig2.axes[2]
    ax3.set_title("Additional Seasonal Effects", fontsize=14)
    ax3.set_ylabel("Effect on Wave Height", fontsize=12)
    ax3.legend(["Seasonality (Other)"])

plt.savefig("Prophet Wave Height Trend (m) and Yearly Seasonality Pattern.png", dpi=300, bbox_inches="tight")
plt.show()

"""
TREND:
The "Trend Over Time" sub-graph in Facebook Prophet shows the underlying long-term trend of the data. It helps us understand the general direction of the time series without seasonal variations or short-term fluctuations.
 Upward Trend → Indicates that the predicted values are increasing over time (e.g., wave heights are rising).
If the Trend Over Time graph shows an increasing slope, it means wave heights are generally rising over time.

"Yearly Seasonality Pattern":
Y-Axis = Seasonal Effect (Wave Height Impact in Your Case) → Represents how much the wave height is influenced by seasonality at different times of the year.
Peaks indicate times when the wave height is typically higher.
Dips indicate times when the wave height is lower.
 How to Interpret the Sub-Graph
Peaks in the Graph

These show months where wave heights are typically higher.
If there’s a peak in December, it means waves are usually stronger in winter.
Dips in the Graph

These show months where wave heights are lower.
If there’s a dip in July/August, it suggests waves are calmer in summer.
Amplitude of the Pattern

A strong up-and-down pattern means the seasonality effect is significant.
A flat or nearly horizontal line means no strong yearly seasonality was detected.
If the graph shows peaks in winter months, this suggests wave heights are higher in winter.
If it dips in summer months, this means wave heights are lower in summer.
"""


