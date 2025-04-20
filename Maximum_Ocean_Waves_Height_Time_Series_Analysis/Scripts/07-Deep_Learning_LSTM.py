import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras import metrics

###########################################
# ### **9. Deep Learning Predictive Model with LSTM**
###########################################

# ************************************************************
# **9.1. LSTM Neural Network for Time Series Forecasting  **
# ************************************************************

df_data = pd.read_csv('final_dataset.csv', index_col=0)
df_data = df_data.sort_index()

LSTM_model_path = "models/LSTM_model.h5"
csv_filename = "LSTM_evaluation_metrics.csv"

max_wave_height_var = 'Maximum individual wave height'

# Define the time series with the values of the variable 'Maximum individual wave height - mean (by month)'
df_Maximum_individual_wave = df_data[[max_wave_height_var]]

# Ensure that the index of the DataFrame is of type datetime
df_Maximum_individual_wave.index = pd.to_datetime(df_Maximum_individual_wave.index)

# Ensure data is aligned to monthly frequency
df_Maximum_individual_wave = df_Maximum_individual_wave.asfreq('MS')

# Normalize data for better LSTM performance
scaler = MinMaxScaler(feature_range=(0, 1))
df_Maximum_individual_wave_scaled = scaler.fit_transform(df_Maximum_individual_wave)

# Convert data to supervised learning format with time lags
def create_lagged_features(data, lags=12):
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data[i - lags:i, 0])  # Past 'lags' values
        y.append(data[i, 0])  # Current value to predict
    return np.array(X), np.array(y)

# Function to generate lag features based on periodicity (lags)
def create_lag_features_index(data, lags=12):
    df = data.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[max_wave_height_var].shift(lag)
    df.dropna(inplace=True)  # Remove NaN values due to shifting
    return df

# Define lag size based on seasonality
num_lags = 12  # Since the data is monthly, this captures yearly seasonality

# Split into train (85%) and test (15%)
train_size = int(len(df_Maximum_individual_wave_scaled) * 0.85)
train_data, test_data = df_Maximum_individual_wave_scaled[:train_size], df_Maximum_individual_wave_scaled[train_size:]

# Generate lagged features (arrays: no index)
X_train, y_train = create_lagged_features(train_data, num_lags)
X_test, y_test = create_lagged_features(test_data, num_lags)

# Generate features for the model (with index).
# It is necessary for plotting.
train_size_not_scaled = int(len(df_Maximum_individual_wave) * 0.85)
train_data_not_scaled, test_data_not_scaled = df_Maximum_individual_wave.iloc[:train_size_not_scaled], df_Maximum_individual_wave.iloc[train_size_not_scaled:]
train_features = create_lag_features_index(train_data_not_scaled, num_lags)
test_features = create_lag_features_index(test_data_not_scaled, num_lags)
# Separate independent (X) and dependent (y) variables
X_train_f, y_train_f = train_features.drop(columns=[max_wave_height_var]), train_features[max_wave_height_var]
X_test_f, y_test_f = test_features.drop(columns=[max_wave_height_var]), test_features[max_wave_height_var]


# Reshape data for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Function to compute AIC and BIC
def compute_aic_bic(y_true, y_pred, num_params):
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    sigma2 = sse / n

    aic = n * np.log(sigma2) + 2 * num_params
    bic = n * np.log(sigma2) + num_params * np.log(n)

    return aic, bic

# Function to save the LSTM model
def save_model_hdf5(model, filepath):
    model.save(filepath)
    print(f"Model saved to {filepath}")

# Function to load the LSTM model
def load_model_hdf5(filepath):
    model = tf.keras.models.load_model(filepath, custom_objects={'mse': metrics.MeanSquaredError()})
    print(f"Model loaded from {filepath}")
    return model

# Check if the model file exists
if os.path.exists(LSTM_model_path):
    # Load the model if it exists
    LSTM_model = load_model_hdf5(LSTM_model_path)
else:
    # Define the LSTM model
    LSTM_model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(num_lags, 1)),  # More LSTM units
        Dropout(0.1),  # Reduce dropout slightly
        LSTM(units=100, return_sequences=False),
        Dropout(0.1),
        Dense(units=50, activation='relu'),  # Increase dense layer neurons
        Dense(units=1)  # Output layer
    ])
    # Compile the model
    LSTM_model.compile(optimizer='adam', loss='mse')

    # Train the LSTM model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    LSTM_model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

    # Save the trained model
    save_model_hdf5(LSTM_model, LSTM_model_path)

# Make predictions
y_train_pred = LSTM_model.predict(X_train)
y_test_pred = LSTM_model.predict(X_test)

# Inverse transform predictions to original scale
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate performance metrics for train set
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
aic_train, bic_train = compute_aic_bic(y_train, y_train_pred, num_lags + 1)

# Calculate performance metrics for test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)
aic_test, bic_test = compute_aic_bic(y_test, y_test_pred, num_lags + 1)

# Print LSTM results
evaluation_metrics_df = pd.DataFrame({
    "Set": ["Training", "Test"],
    "RMSE": [rmse_train, rmse_test],
    "R²": [r2_train, r2_test],
    "AIC": [aic_train, aic_test],
    "BIC": [bic_train, bic_test]
})

print("\nLSTM results:")
print(evaluation_metrics_df)

# Save evaluation metrics to a CSV file
evaluation_metrics_df.to_csv(csv_filename, index=False)


train_data_plus_lags=df_Maximum_individual_wave.iloc[:train_size+num_lags]

plt.figure(figsize=(20, 6))
plt.plot(train_data_plus_lags.index, train_data_plus_lags, label='Train', color='lightblue')
plt.plot(X_test_f.index, y_test, label='Test', color='lightgreen')
#  Add vertical line at the end of the training data:
plt.axvline(x=X_test_f.index[0], color='gray', linestyle='--')
plt.plot(X_test_f.index, y_test_pred, label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height in metres (monthly mean)', fontsize=18)
plt.title("Actual vs Predicted Values (LSTM)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)
plt.savefig("LSTM.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(20, 6))
plt.plot(X_test_f.index, y_test, label='Test', color='lightgreen')
plt.plot(X_test_f.index, y_test_pred, label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height in metres (monthly mean)', fontsize=18)
plt.title('LSTM prediction', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)
plt.savefig("LSTM_test_time_series_and_forecasts.png", dpi=300, bbox_inches="tight")
plt.show()
