import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses

###########################################
# ### **9. Deep Learning Predictive Model with Bi-LSTM**
###########################################

# ************************************************************
# **9.2. Bidirectional LSTM Neural Network for Time Series Forecasting  **
# ************************************************************

df_data = pd.read_csv('final_dataset.csv', index_col=0)
df_data = df_data.sort_index()

BiLSTM_model_path = "models/BiLSTM_model.h5"
csv_filename = "BiLSTM_evaluation_metrics.csv"

max_wave_height_var = 'Maximum individual wave height'

# Define the time series with the values of the variable 'Maximum individual wave height - mean (by month)'
df_Maximum_individual_wave = df_data[[max_wave_height_var]]

# Ensure that the index of the DataFrame is of type datetime
df_Maximum_individual_wave.index = pd.to_datetime(df_Maximum_individual_wave.index)

# Ensure data is aligned to monthly frequency
df_Maximum_individual_wave = df_Maximum_individual_wave.asfreq('MS')

# Normalize data for better Bi-LSTM performance
scaler = MinMaxScaler(feature_range=(0, 1))
df_Maximum_individual_wave_scaled = scaler.fit_transform(df_Maximum_individual_wave)

# Convert data to supervised learning format with time lags
def create_lagged_features(data, lags=12):
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data[i - lags:i, 0])  # Past 'lags' values
        y.append(data[i, 0])  # Current value to predict
    return np.array(X), np.array(y)

# Define lag size based on seasonality
num_lags = 12  # Since the data is monthly, this captures yearly seasonality

# Split into train (85%) and test (15%)
train_size = int(len(df_Maximum_individual_wave_scaled) * 0.85)
train_data, test_data = df_Maximum_individual_wave_scaled[:train_size], df_Maximum_individual_wave_scaled[train_size:]

# Generate lagged features
X_train, y_train = create_lagged_features(train_data, num_lags)
X_test, y_test = create_lagged_features(test_data, num_lags)

# Reshape data for Bi-LSTM [samples, timesteps, features]
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

# Function to save the Bi-LSTM model
def save_model_hdf5(model, filepath):
    model.save(filepath)
    print(f"Model saved to {filepath}")

# Function to load the Bi-LSTM model
def load_model_hdf5(filepath):
    #model = tf.keras.models.load_model(filepath)
    model = tf.keras.models.load_model(filepath, custom_objects={'mse': losses.MeanSquaredError()})
    print(f"Model loaded from {filepath}")
    return model

# Check if the model file exists
if os.path.exists(BiLSTM_model_path):
    # Load the model if it exists
    BiLSTM_model = load_model_hdf5(BiLSTM_model_path)
else:
    # Define the Bi-LSTM model
    BiLSTM_model = Sequential([
        Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(num_lags, 1))),  # First Bi-LSTM layer
        Dropout(0.1),  # Regularization
        Bidirectional(LSTM(units=100, return_sequences=False)),  # Second Bi-LSTM layer
        Dropout(0.1),
        Dense(units=50, activation='relu'),  # Dense layer
        Dense(units=1)  # Output layer
    ])

    # Compile the model
    BiLSTM_model.compile(optimizer='adam', loss='mse')

    # Train the Bi-LSTM model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    BiLSTM_model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

    # Save the trained model
    save_model_hdf5(BiLSTM_model, BiLSTM_model_path)

# Make predictions
y_train_pred = BiLSTM_model.predict(X_train)
y_test_pred = BiLSTM_model.predict(X_test)

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

# Print Bi-LSTM results
evaluation_metrics_df = pd.DataFrame({
    "Set": ["Training", "Test"],
    "RMSE": [rmse_train, rmse_test],
    "R²": [r2_train, r2_test],
    "AIC": [aic_train, aic_test],
    "BIC": [bic_train, bic_test]
})

print("\nBi-LSTM results:")
print(evaluation_metrics_df)

# Save evaluation metrics to a CSV file
evaluation_metrics_df.to_csv(csv_filename, index=False)

"""
Bi-LSTM results:
        Set      RMSE        R²          AIC          BIC
0  Training  0.528524  0.605874 -1064.411782 -1002.647463
1      Test  0.531657  0.687448  -153.419023  -114.993271

LSTM remains the best model for test generalization (lowest RMSE, highest R²).
Bi-LSTM slightly overfits, making it less effective for forecasting.
Prophet is still a strong contender for test predictions.
"""



import matplotlib.dates as mdates

plt.figure(figsize=(20, 6))
plt.plot(df_Maximum_individual_wave.index[train_size+num_lags:], y_test, label='Test', color='lightgreen')
plt.plot(df_Maximum_individual_wave.index[train_size+num_lags:], y_test_pred, label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height in metres (monthly mean)', fontsize=12)
plt.title('Bi-LSTM prediction', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Format x-axis with years
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()
plt.legend(loc='upper left', fontsize=14, frameon=False)
plt.savefig("07 - Bi-LSTM Time Series Forecast.png", dpi=300, bbox_inches="tight")
plt.show()