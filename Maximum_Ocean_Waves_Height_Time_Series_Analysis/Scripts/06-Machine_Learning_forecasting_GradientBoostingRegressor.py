import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import h5py
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

###########################################
# ### **8. Machine Learning predictive model**
###########################################

# ************************************************************
# **8.3. Gradient Boosting Regressor  **
# ************************************************************

df_data = pd.read_csv('final_dataset.csv', index_col=0)
range_limit_of_selected_data = '2024-12-31'  # ONLY TAKE DATA UNTIL 2024
df_data = df_data.sort_index()

GradientBoostingRegressor_model_path = "models/GradientBoostingRegressor_model.h5"
csv_filename = "GradientBoostingRegressor_evaluation_metrics.csv"

max_wave_height_var = 'Maximum individual wave height'

# Define the time series with the values of the variable 'Maximum individual wave height - mean (by month)':
df_Maximum_individual_wave = df_data[[max_wave_height_var]]

# Ensure that the index of the DataFrame is of type datetime:
df_Maximum_individual_wave.index = pd.to_datetime(df_Maximum_individual_wave.index)

# Ensure data is aligned to monthly frequency
df_Maximum_individual_wave = df_Maximum_individual_wave.asfreq('MS')

# Keep data only until the specified date
df_Maximum_individual_wave = df_Maximum_individual_wave[df_Maximum_individual_wave.index <= range_limit_of_selected_data]


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


# Function to generate lag features based on periodicity (lags)
def create_lag_features(data, lags=12):
    df = data.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[max_wave_height_var].shift(lag)
    df.dropna(inplace=True)  # Remove NaN values due to shifting
    return df


# Function to compute AIC and BIC
def compute_aic_bic(y_true, y_pred, num_params):
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    sigma2 = sse / n

    aic = n * np.log(sigma2) + 2 * num_params
    bic = n * np.log(sigma2) + num_params * np.log(n)

    return aic, bic


# Split into train (85%) and test (15%)
train_size = int(len(df_Maximum_individual_wave) * 0.85)
train_data, test_data = df_Maximum_individual_wave.iloc[:train_size], df_Maximum_individual_wave.iloc[train_size:]

# Define the number of lags based on periodicity
num_lags = 12

# Generate features for the model
train_features = create_lag_features(train_data, num_lags)
test_features = create_lag_features(test_data, num_lags)

# Separate independent (X) and dependent (y) variables
X_train, y_train = train_features.drop(columns=[max_wave_height_var]), train_features[max_wave_height_var]
X_test, y_test = test_features.drop(columns=[max_wave_height_var]), test_features[max_wave_height_var]

# Check if the model file exists
if os.path.exists(GradientBoostingRegressor_model_path):
    # Load the model if it exists
    GradientBoostingRegressor_model = load_model_hdf5(GradientBoostingRegressor_model_path)
else:
    # Define manually optimized hyperparameters
    best_params = {
        'n_estimators': 100,        # Number of boosting stages
        'learning_rate': 0.1,      # Step size reduction factor
        'max_depth': 3,             # Control tree complexity
        'subsample': 0.7,           # Use 80% of samples per boosting iteration
        'loss': 'squared_error'     # Squared error loss function
    }

    # Define and train the GradientBoostingRegressor model
    GradientBoostingRegressor_model = GradientBoostingRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        loss=best_params['loss'],
        random_state=42
    )
    
    GradientBoostingRegressor_model.fit(X_train, y_train)
    save_model_hdf5(GradientBoostingRegressor_model, GradientBoostingRegressor_model_path)

# Make predictions
y_train_pred = GradientBoostingRegressor_model.predict(X_train)
y_test_pred = GradientBoostingRegressor_model.predict(X_test)

# Calculate performance metrics for train set
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
aic_train, bic_train = compute_aic_bic(y_train, y_train_pred, num_lags + 1)

# Calculate performance metrics for test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)
aic_test, bic_test = compute_aic_bic(y_test, y_test_pred, num_lags + 1)

# Print GradientBoostingRegressor results
evaluation_metrics_df = pd.DataFrame({
    "Set": ["Training", "Test"],
    "RMSE": [rmse_train, rmse_test],
    "R²": [r2_train, r2_test],
    "AIC": [aic_train, aic_test],
    "BIC": [bic_train, bic_test]
})

print("\nGradientBoostingRegressor results:")
print(evaluation_metrics_df)

# GradientBoostingRegressor has overtaken AdaBoost in performance
"""
GradientBoostingRegressor results:
        Set      RMSE        R²          AIC          BIC
0  Training  0.349190  0.827959 -1773.155681 -1711.391362
1      Test  0.569891  0.642724  -132.571645   -94.237767
"""

# Save evaluation metrics to a CSV file
evaluation_metrics_df.to_csv(csv_filename, index=False)


train_data_plus_lags=df_Maximum_individual_wave.iloc[:train_size+num_lags]

plt.figure(figsize=(20, 6))
plt.plot(np.array(train_data_plus_lags.index), np.array(train_data_plus_lags), label='Train', color='lightblue')
plt.plot(np.array(X_test.index), np.array(y_test), label='Test', color='lightgreen')
#  Add vertical line at the end of the training data:
plt.axvline(x=y_test.index[0], color='gray', linestyle='--')
plt.plot(np.array(X_test.index), np.array(y_test_pred), label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height (metres)', fontsize=18)
plt.title("Actual vs Predicted Values (GradientBoosting Regressor)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=14, frameon=False)
plt.savefig("GradientBoosting_Regressor.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(20, 6))
plt.plot(np.array(X_test.index), np.array(y_test), label='Test', color='lightgreen')
plt.plot(np.array(y_test.index), np.array(y_test_pred), label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height (metres)', fontsize=18)
plt.title('GradientBoosting Regressor prediction using Top Features', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)
plt.savefig("GradientBoosting_Regressor_test_time_series_and_forecasts.png", dpi=300, bbox_inches="tight")
plt.show()
