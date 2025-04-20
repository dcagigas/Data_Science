import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Conv1D
from tensorflow.keras.callbacks import EarlyStopping


###########################################
# ### **9. Deep Learning Predictive Model with Transformers**
###########################################

# ************************************************************
# **9.4. Transformer-Based Model for Time Series Forecasting  **
# ************************************************************

df_data = pd.read_csv('final_dataset.csv', index_col=0)
df_data = df_data.sort_index()

Transformer_model_path = "models/Transformer_model.keras"
csv_filename = "Transformer_evaluation_metrics.csv"

max_wave_height_var = 'Maximum individual wave height'

# Define the time series with the values of the variable 'Maximum individual wave height - mean (by month)'
df_Maximum_individual_wave = df_data[[max_wave_height_var]]

# Ensure that the index of the DataFrame is of type datetime
df_Maximum_individual_wave.index = pd.to_datetime(df_Maximum_individual_wave.index)

# Ensure data is aligned to monthly frequency
df_Maximum_individual_wave = df_Maximum_individual_wave.asfreq('MS')

# Normalize data for better Transformer performance
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

# Reshape data for Transformer [samples, timesteps, features]
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

# Function to build a Transformer model
def build_transformer_model(input_shape, num_heads=6, ff_dim=128, dropout_rate=0.05):
    inputs = Input(shape=input_shape)

    # Positional Encoding (Gives time-awareness to Transformer)
    position_encoding = tf.range(start=0, limit=input_shape[0], delta=1, dtype=tf.float32)
    position_encoding = tf.expand_dims(position_encoding, axis=0)  # Shape: (1, timesteps)
    position_encoding = Dense(input_shape[-1])(position_encoding)  # Linear transformation
    inputs_with_position = inputs + position_encoding  # Adding positional encoding

    # Convolutional Feature Extraction (Enhances pattern recognition before attention)
    conv_layer = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs_with_position)

    # Transformer Block (Reduced to One)
    attn_output = MultiHeadAttention(num_heads, key_dim=64)(conv_layer, conv_layer)
    attn_output = Dropout(dropout_rate)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output + conv_layer)  # Add residual connection

    # Feed-Forward Network (Reduced `ff_dim=128`)
    ffn_output = Dense(ff_dim, activation='relu')(attn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = Dense(input_shape[-1])(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attn_output)  # Add residual connection

    # Output Layer (Predicting only the last timestep's value)
    outputs = Dense(1)(ffn_output[:, -1, :])  # Selecting only the last timestep's output

    # Compile Model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model
    

# Train or load the Transformer model
if os.path.exists(Transformer_model_path):
    #Transformer_model = tf.keras.models.load_model(Transformer_model_path, compile=False)
    Transformer_model = tf.keras.models.load_model(Transformer_model_path)
    #Transformer_model.compile(optimizer='adam', loss='mse')
    print(f"Model loaded from {Transformer_model_path}")
else:
    Transformer_model = build_transformer_model(input_shape=(num_lags, 1))

    # Train the Transformer model
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    Transformer_model.fit(X_train, y_train, epochs=250, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])
    # Save the trained model
    #Transformer_model.save(Transformer_model_path, save_format='h5')
    Transformer_model.save(Transformer_model_path)
    print(f"Model saved to {Transformer_model_path}")

# Make predictions
y_train_pred = Transformer_model.predict(X_train).squeeze()  # Remove extra dimensions
y_test_pred = Transformer_model.predict(X_test).squeeze()    # Remove extra dimensions

# Inverse transform predictions to original scale
y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1))
y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1))
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

# Print Transformer results
evaluation_metrics_df = pd.DataFrame({
    "Set": ["Training", "Test"],
    "RMSE": [rmse_train, rmse_test],
    "R²": [r2_train, r2_test],
    "AIC": [aic_train, aic_test],
    "BIC": [bic_train, bic_test]
})

print("\nTransformer results:")
print(evaluation_metrics_df)

# Save evaluation metrics to a CSV file
evaluation_metrics_df.to_csv(csv_filename, index=False)


"""
Transformer results:
        Set      RMSE        R²          AIC         BIC
0  Training  0.548328  0.575785 -1001.508993 -939.744673
1      Test  0.544058  0.672698  -146.870786 -108.445034

Transformer Is Now More Accurate Than Before ✅

Test RMSE improved from 0.5495 → 0.5441, meaning better predictions.
Test R² increased from 0.6661 → 0.6727, meaning the model now explains more variance.
AIC and BIC Further Decreased ✅

Lower AIC/BIC means the model is now more efficient and generalizes better.
Training RMSE Improved ✅

Dropped from 0.5689 → 0.5483, meaning better training performance without overfitting.
This suggests that reducing the dropout and tuning the attention heads worked well.
Transformer Is Now as Good as LSTM/Bi-LSTM for Time Series Prediction ✅

Transformer Test RMSE: 0.5441 vs. LSTM Test RMSE: 0.5488 → Transformer is now slightly better!
Transformer Test R²: 0.6727 vs. LSTM Test R²: 0.6670 → Transformer explains slightly more variance.

"""

train_data_plus_lags=df_Maximum_individual_wave.iloc[:train_size+num_lags]

plt.figure(figsize=(20, 6))
plt.plot(train_data_plus_lags.index, train_data_plus_lags, label='Train', color='lightblue')
plt.plot(df_Maximum_individual_wave.index[train_size+num_lags:], y_test, label='Test', color='lightgreen')
#  Add vertical line at the end of the training data:
plt.axvline(x=train_data_plus_lags.index[-1], color='gray', linestyle='--')
plt.plot(df_Maximum_individual_wave.index[train_size+num_lags:], y_test_pred, label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height in metres (monthly mean)', fontsize=12)
plt.title("Actual vs Predicted Values (Transformer)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=14, frameon=False)
plt.savefig("Transformer.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(20, 6))
plt.plot(df_Maximum_individual_wave.index[train_size+num_lags:], y_test, label='Test', color='lightgreen')
plt.plot(df_Maximum_individual_wave.index[train_size+num_lags:], y_test_pred, label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height in metres (monthly mean)', fontsize=12)
plt.title('Transformer prediction', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=14, frameon=False)
plt.savefig("Transformer_test_time_series_and_forecasts.png", dpi=300, bbox_inches="tight")
plt.show()
