import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import h5py
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

###########################################
# ### **8.Machine Learning predictive model**
###########################################

# ************************************************************
# **8.1. Random Forest Regressor  **
# ************************************************************


df_data = pd.read_csv('final_dataset.csv', index_col=0)
range_limit_of_selected_data = '2024-12-31' # ONLY TAKE DATA UNTIL 2024
df_data = df_data.sort_index()

RandomForestRegressor_model_path = "models/RandomForestRegressor_model.h5"
csv_filename = "RandomForestRegressor_evaluation_metrics.csv"


max_wave_height_var= 'Maximum individual wave height'


# Define the time series with the values of the variable 'Maximum individual wave height - mean (by month)':
df_Maximum_individual_wave = df_data[[max_wave_height_var]]
# Ensure that the index of the DataFrame is of type datetime:
df_Maximum_individual_wave.index = pd.to_datetime(df_Maximum_individual_wave.index)

# Asegurar que los datos estén alineados en frecuencia mensual
df_Maximum_individual_wave = df_Maximum_individual_wave.asfreq('MS')
# IMPORTANT: GET DATA JUST UNTIL A DATE.
df_Maximum_individual_wave = df_Maximum_individual_wave[df_Maximum_individual_wave.index <= range_limit_of_selected_data]




# Function to save the model
def save_model_hdf5(model, filepath):
    # Save the model in binary format using joblib
    temp_file = filepath + "_temp.pkl"
    joblib.dump(model, temp_file)
    
    # Convert to HDF5 format
    with h5py.File(filepath, 'w') as h5f:
        with open(temp_file, 'rb') as f:
            model_data = f.read()
            h5f.create_dataset('model', data=np.void(model_data))
    
    # Remove the temporary file
    os.remove(temp_file)
    print(f"Model saved to {filepath}")


# Function to load the model
def load_model_hdf5(filepath):
    with h5py.File(filepath, 'r') as h5f:
        model_data = h5f['model'][()]
    
    # Temporarily save the model to a file to load it with joblib
    temp_file = filepath + "_temp.pkl"
    with open(temp_file, 'wb') as f:
        f.write(model_data.tobytes())
    
    # Load the model from the temporary file
    model = joblib.load(temp_file)
    
    # Remove the temporary file
    os.remove(temp_file)
    print(f"Model loaded from {filepath}")
    
    return model

# Function to generate lag features based on periodicity (lags)
# In our case lags should be 12 because this is the seasonality of the time series 
def create_lag_features(data, lags=12):
    df = data.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[max_wave_height_var].shift(lag)
    df.dropna(inplace=True)  # Remove rows with NaN values due to shifting
    return df

# Function to compute AIC and BIC
def compute_aic_bic(y_true, y_pred, num_params):
    n = len(y_true)  # Number of data points
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)  # Sum of squared errors
    sigma2 = sse / n  # Estimated variance

    aic = n * np.log(sigma2) + 2 * num_params
    bic = n * np.log(sigma2) + num_params * np.log(n)

    return aic, bic

# BEGIN general RandomForestRegressor process:

# Split into train (85%) and test (15%)
train_size = int(len(df_Maximum_individual_wave) * 0.85)
# Train: 1940-01-01 to 2012-03-01 
# Test: 2012-04-01 to 2024-12-01
train_data, test_data = df_Maximum_individual_wave.iloc[:train_size], df_Maximum_individual_wave.iloc[train_size:]

# Define the number of lags based on periodicity
num_lags = 12  # Considering seasonality of 12

# Generate features for the model
train_features = create_lag_features(train_data, num_lags)
test_features = create_lag_features(test_data, num_lags)

# Separate independent (X) and dependent (y) variables
# X_Train: 1940-01-01 to 2012-03-01 
# y_train: 1941-04-01 to 2012-03-01
# X_test: 2013-04-01 to 2024-12-01 
# y_test: 2013-04-01 to 2024-12-01 (same as "y_test_pred": 141 elements)
X_train, y_train = train_features.drop(columns=[max_wave_height_var]), train_features[max_wave_height_var]
X_test, y_test = test_features.drop(columns=[max_wave_height_var]), test_features[max_wave_height_var]

# Check if the model file exists
if os.path.exists(RandomForestRegressor_model_path):
    # Load the model if it exists
    RandomForestRegressor_model = load_model_hdf5(RandomForestRegressor_model_path)
else:
    # Define manually optimized hyperparameters
    best_params = {
        'n_estimators': 100,       # Increase the number of trees for better stability
        'max_depth': 8,           # Limit tree depth to prevent overfitting
        'min_samples_split': 10,    # Minimum number of samples required to split a node
        'min_samples_leaf': 8,      # Minimum number of samples required per leaf node
        'max_features': 'sqrt'    # Randomly select features to reduce complexity
    }

    # Define and train the RandomForestRegressor model
    # Initialize and train the RandomForestRegressor model with optimized hyperparameters
    RandomForestRegressor_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        random_state=42
    )
    RandomForestRegressor_model.fit(X_train, y_train)   

    save_model_hdf5 (RandomForestRegressor_model, RandomForestRegressor_model_path)

"""
1) OVEFITTING
best_params = {
        'n_estimators': 200,       # Increase the number of trees for better stability
        'max_depth': 20,           # Limit tree depth to prevent overfitting
        'min_samples_split': 5,    # Minimum number of samples required to split a node
        'min_samples_leaf': 2      # Minimum number of samples required per leaf node
    }
        Set      RMSE        R²          AIC          BIC
0  Training  0.245679  0.914839 -2374.379485 -2312.615166
1      Test  0.583473  0.625491  -125.929413   -87.595534

2) UNDERFITTING
    best_params = {
        'n_estimators': 50,       # Increase the number of trees for better stability
        'max_depth': 10,           # Limit tree depth to prevent overfitting
        'min_samples_split': 10,    # Minimum number of samples required to split a node
        'min_samples_leaf': 5,      # Minimum number of samples required per leaf node
        'max_features': 'sqrt'    # Randomly select features to reduce complexity
    }
        Set      RMSE        R²          AIC          BIC
0  Training  0.384559  0.791343 -1608.174720 -1546.410401
1      Test  0.567814  0.645323  -133.600951   -95.267073

3) UNDERFITTING
    best_params = {
        'n_estimators': 75,       # Increase the number of trees for better stability
        'max_depth': 8,           # Limit tree depth to prevent overfitting
        'min_samples_split': 10,    # Minimum number of samples required to split a node
        'min_samples_leaf': 10,      # Minimum number of samples required per leaf node
        'max_features': 'sqrt'    # Randomly select features to reduce complexity
    }

        Set      RMSE        R²          AIC          BIC
0  Training  0.443375  0.722637 -1364.811953 -1303.047634
1      Test  0.569349  0.643403  -132.839580   -94.505701

4) BETTER --> BEST!!!!
    best_params = {
        'n_estimators': 100,       # Increase the number of trees for better stability
        'max_depth': 8,           # Limit tree depth to prevent overfitting
        'min_samples_split': 10,    # Minimum number of samples required to split a node
        'min_samples_leaf': 8,      # Minimum number of samples required per leaf node
        'max_features': 'sqrt'    # Randomly select features to reduce complexity
    }
RandomForestRegressor results:
        Set      RMSE        R²          AIC          BIC
0  Training  0.431759  0.736980 -1410.208977 -1348.444658
1      Test  0.570920  0.641432  -132.062744   -93.728866

5) WORSE THAN 4)
    best_params = {
        'n_estimators': 150,       # Increase the number of trees for better stability
        'max_depth': 8,           # Limit tree depth to prevent overfitting
        'min_samples_split': 10,    # Minimum number of samples required to split a node
        'min_samples_leaf': 10,      # Minimum number of samples required per leaf node
        'max_features': 'sqrt'    # Randomly select features to reduce complexity
    }
    
RandomForestRegressor results:
        Set      RMSE        R²          AIC          BIC
0  Training  0.444232  0.721564 -1361.509015 -1299.744695
1      Test  0.573720  0.637906  -130.682871   -92.348992
"""


# Make predictions
y_train_pred = RandomForestRegressor_model.predict(X_train)
y_test_pred = RandomForestRegressor_model.predict(X_test)

# Calculate performance metrics for train set
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
aic_train, bic_train = compute_aic_bic(y_train, y_train_pred, num_lags + 1)  # num_lags + 1 for intercept

# Calculate performance metrics for test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)
aic_test, bic_test = compute_aic_bic(y_test, y_test_pred, num_lags + 1)

# Print RandomForestRegressor results
evaluation_metrics_df = pd.DataFrame({
    "Set": ["Training", "Test"],
    "RMSE": [rmse_train, rmse_test],
    "R²": [r2_train, r2_test],
    "AIC": [aic_train, aic_test],
    "BIC": [bic_train, bic_test]
})
print()
print("RandomForestRegressor results:")
print(evaluation_metrics_df)

# Safe to a CSV file
evaluation_metrics_df.to_csv(csv_filename, index=False)


train_data_plus_lags=df_Maximum_individual_wave.iloc[:train_size+num_lags]

plt.figure(figsize=(20, 6))
plt.plot(np.array(train_data_plus_lags.index), np.array(train_data_plus_lags), label='Train', color='lightblue')
plt.plot(np.array(X_test.index), np.array(y_test), label='Test', color='lightgreen')
#  Add vertical line at the end of the training data:
plt.axvline(x=y_test.index[0], color='gray', linestyle='--')
plt.plot(np.array(X_test.index), np.array(y_test_pred), label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height in metres (monthly mean)', fontsize=12)
plt.title("Actual vs Predicted Values (Random Forest Regressor)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=14, frameon=False)
plt.savefig("Random_Forest_Regressor.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(20, 6))
plt.plot(np.array(X_test.index), np.array(y_test), label='Test', color='lightgreen')
plt.plot(np.array(y_test.index), np.array(y_test_pred), label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height in metres (monthly mean)', fontsize=12)
plt.title('Random Forest Regressor prediction using Top Features', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=14, frameon=False)
plt.savefig("Random_Forest_Regressor_test_time_series_and_forecasts.png", dpi=300, bbox_inches="tight")
plt.show()


