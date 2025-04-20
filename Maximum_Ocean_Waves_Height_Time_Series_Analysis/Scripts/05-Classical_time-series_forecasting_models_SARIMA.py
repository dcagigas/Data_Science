import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import pickle
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima  # Import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import scipy.stats as stats

###########################################
# ### **7.ARIMA/SARIMA models**
###########################################

df_data = pd.read_csv('final_dataset.csv', index_col=0)
range_limit_of_selected_data = '2024-12-31' # ONLY TAKE DATA UNTIL 2024
df_data = df_data.sort_index()

sarima_model_path = "models/sarima_model_v2.pkl"
arima_model_path = "models/arima_model_v2.pkl"

arima_csv_filename = "ARIMA_evaluation_metrics.csv"
sarima_csv_filename = "SARIMA_evaluation_metrics.csv"

max_wave_height_var= 'Maximum individual wave height'


# Define the time series with the values of the variable 'Maximum individual wave height - mean (by month)':
st_Maximum_individual_wave = df_data[[max_wave_height_var]]
# Ensure that the index of the DataFrame is of type datetime:
st_Maximum_individual_wave.index = pd.to_datetime(st_Maximum_individual_wave.index)

# Asegurar que los datos estén alineados en frecuencia mensual
st_Maximum_individual_wave = st_Maximum_individual_wave.asfreq('MS')
# IMPORTANT: GET DATA JUST UNTIL A DATE.
st_Maximum_individual_wave = st_Maximum_individual_wave[st_Maximum_individual_wave.index <= range_limit_of_selected_data]


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

# Functions to load/save the SARIMA or ARIMA models to an HDF5 file
def save_arima_model_pickle(model, order, filename):
    with open(filename, 'wb') as file:
        pickle.dump({"model": model, "order": order}, file)

def load_arima_model_pickle(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data["model"], data["order"]
    return None, None

def save_sarima_model_pickle(model, order, seasonal_order, filename):
    """Guarda un modelo SARIMA junto con su orden y orden estacional en un archivo pickle."""
    with open(filename, 'wb') as file:
        pickle.dump({"model": model, "order": order, "seasonal_order": seasonal_order}, file)

def load_sarima_model_pickle(filename):
    """Carga un modelo SARIMA y sus órdenes desde un archivo pickle."""
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data["model"], data["order"], data["seasonal_order"]
    return None, None, None


print ("Augmented Dickey—Fuller (ADF) test")
print ('If p-value < 0.05 then the time series is stationary')
print ('p-value: %f' % adfuller(st_Maximum_individual_wave)[1])

# 1) Split the series into training (85%) and testing (15%)
train_size = int(len(st_Maximum_individual_wave) * 0.85)
train, test = st_Maximum_individual_wave[:train_size], st_Maximum_individual_wave[train_size:]

# 2) Manually try a few ARIMA models and choose the best based on AIC/BIC
best_aic = float("inf")
best_order = None
best_model = None

loaded_model, loaded_order = load_arima_model_pickle(arima_model_path)
if loaded_model:
    best_model = loaded_model
    best_order = loaded_order
else:
    # Use auto_arima to find the best ARIMA model
    auto_arima_model = auto_arima(
        train, 
        seasonal=False,          # Non-seasonal ARIMA model
        stepwise=True,           # Faster optimization
        suppress_warnings=True,  # Suppress unnecessary warnings
        trace=True,               # Display model search process
        max_p=10,  # Increase the range of AR terms
        max_q=10,  # Increase the range of MA terms
        max_d=2    # Allow slightly more differentiation if needed
    )

    # Retrieve the best-found model parameters
    best_order = auto_arima_model.order
    """
    If the variance of the time series is increasing over time, a log transformation can help.
    The model assumes stationarity, but variance may be changing over time.
    This is the case of Maximum wave height time series: variance has also increased.
    See "03 - Rolling Variance of Maximum Wave Height.png"
    """
    
    # Train a new ARIMA model with the optimal parameters
    train_log = np.log(train + 1)  # Avoid log(0) issues

    best_model = ARIMA(train_log, order=best_order).fit()

    """
    # Trying a range of (p,d,q) values to find the best ARIMA model
    for p in range(6):
        for d in range(2):
        #d=1
            for q in range(3):
                try:
                    model = ARIMA(train, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                except:
                    continue
    """
    save_arima_model_pickle (best_model, best_order, arima_model_path)
    

test_log = np.log(test + 1)
test_model = ARIMA(test_log, order=best_order).fit()

# 3) Predict values in the test set using the best model
forecast_log = best_model.forecast(steps=len(test))
# Convert back to original scale
forecast = np.exp(forecast_log) - 1

# 4) Evaluate the model
train_pred = np.exp(best_model.fittedvalues) - 1
test_pred = forecast

# Compute RMSE and R² for train and test sets
train_rmse = np.sqrt(mean_squared_error(train, train_pred))
test_rmse = np.sqrt(mean_squared_error(test, test_pred))
train_r2 = r2_score(train, train_pred)
test_r2 = r2_score(test, test_pred)

# Get train AIC and BIC
train_aic = best_model.aic
train_bic = best_model.bic
# Get test AIC and BIC
test_aic = test_model.aic
test_bic = test_model.bic

arima_evaluation_metrics_df = pd.DataFrame({
    "Set": ["Training", "Test"],
    "RMSE": [train_rmse, test_rmse],
    "R²": [train_r2, test_r2],
    "AIC": [train_aic, test_aic],
    "BIC": [train_bic, test_bic]
})

print("\ARIMA results:")
print(arima_evaluation_metrics_df)

# Save evaluation metrics to a CSV file
arima_evaluation_metrics_df.to_csv(arima_csv_filename, index=False)

"""
Set	        RMSE	            R²	                 AIC	             BIC
Training	0.6847509784688756	0.3376605593229083	-590.1967016235517	-566.3772770807395
Test	    0.7895220543839894	0.30826312387412547	-69.32927636744395	-54.20987376321257
"""


# Plot the actual vs predicted values
plt.figure(figsize=(20, 6))
plt.plot(train.index, train, label='Train', color='lightblue')
plt.plot(test.index, test, label='Test', color='lightgreen')
plt.plot(test.index, test_pred, label='Test forecasts', color='orange')
#  Add vertical line at the end of the training data:
plt.axvline(x=train.index[-1], color='gray', linestyle='--')
plt.ylabel('Maximum wave height (metres)', fontsize=18)
plt.title(f"Actual vs Predicted Values (ARIMA{best_model.model.order})", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)
plt.savefig("05 - Arima Model.png", dpi=300)
plt.show()

plt.figure(figsize=(20, 6))
plt.plot(test.index, test, label='Test', color='lightgreen')
plt.plot(test.index, test_pred, label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height (metres)', fontsize=18)
plt.title(f'ARIMA {best_model.model.order} test time series and forecasts', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)
plt.savefig("05 - Arima Model - Test and forecasts.png", dpi=300)
plt.show()



# 5) Residual Analysis 

residuals = best_model.resid  # Log-scale residuals

# Histogram and QQ-Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True, bins=30)
plt.title("Histogram of Residuals")

plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot of Residuals")
plt.savefig("05 - ARIMA Residual analysis (Histogram and QQ-Plot).png", dpi=300)
plt.show()

# ACF and PACF plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_acf(residuals, lags=40, ax=plt.gca())
plt.subplot(1, 2, 2)
plot_pacf(residuals, lags=40, ax=plt.gca())
plt.savefig("05 - ARIMA Residual ACF and PACF (autocorrelation and partical autocorrelation).png", dpi=300)
plt.show()

# Ljung-Box Test for White Noise
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("\nLjung-Box Test for White Noise:")
print(lb_test)

"""
Ljung-Box Test for White Noise:
       lb_stat     lb_pvalue
10  172.343052  9.074384e-32

Conclusion: Residuals are not white noise. 
They contain patterns, meaning the model did not fully capture the time series dynamics.
"""

# 6) Confidence Interval Analysis

pred = best_model.get_forecast(steps=len(test))
pred_ci_log = pred.conf_int()
forecast_log = pred.predicted_mean  

# Convert forecast and confidence intervals back to original scale
forecast = np.exp(forecast_log) - 1  
pred_ci = np.exp(pred_ci_log) - 1  

# Plot Confidence Intervals
plt.figure(figsize=(10, 5))
plt.title(f'ARIMA {best_model.model.order} Confidence Intervals in Forecasting', fontsize=16)

plt.plot(test.index, test, label="Actual", color="blue")
plt.plot(test.index, forecast, label="Forecast", color="orange")

plt.fill_between(test.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='orange', alpha=0.2)
plt.ylabel('Maximum wave height (metres)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig("05 - ARIMA Model - Confidence Intervals in Forecasting.png", dpi=300)
plt.show()





from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1) Split the series into training (85%) and testing (15%)
train_size = int(len(st_Maximum_individual_wave) * 0.85)
train, test = st_Maximum_individual_wave[:train_size], st_Maximum_individual_wave[train_size:]

# 2) Try different SARIMA models and choose the best based on AIC/BIC
best_sarima_aic = float("inf")
best_sarima_order = None
best_sarima_seasonal_order = None
best_sarima_model = None

# Define a seasonal period of 12 months
seasonal_periods = 12

loaded_model, loaded_order, loaded_seasonal_order = load_sarima_model_pickle(sarima_model_path)
if loaded_model:
    best_sarima_model = loaded_model
    best_sarima_order = loaded_order
    best_sarima_seasonal_order = loaded_seasonal_order
else:
    # Use auto_arima to find the best SARIMA model
    auto_sarima_model = auto_arima(
        train, 
        seasonal=True,           # Seasonal model
        m=12,                    # Monthly seasonality
        stepwise=True,           # Faster optimization
        suppress_warnings=True,  # Suppress unnecessary warnings
        trace=True,              # Display model search process
        max_p=10,  
        max_q=10,  
        max_d=2,   
        max_P=5,   # More seasonal AR terms
        max_Q=5,   # More seasonal MA terms
        max_D=2,   # Allow up to 2 seasonal differences
        max_order=20  # Allow more complex models    
        )
    # Retrieve the best-found model parameters
    best_sarima_order = auto_sarima_model.order
    best_sarima_seasonal_order = auto_sarima_model.seasonal_order
    # Apply log transformation
    train_log = np.log(train + 1)  # Avoid log(0)

    # Train a new SARIMA model with the optimal parameters
    sarima_model = SARIMAX(train_log, order=best_sarima_order, seasonal_order=best_sarima_seasonal_order)
    best_sarima_model = sarima_model.fit()
    # Iterate over possible (p, d, q) and seasonal (P, D, Q, s) values
    """
    for p in range(6):
        for d in range(2):
        #d=1
            for q in range(3):
                for P in range(2):
                    for D in range(2):
                        for Q in range(2):
                            s = seasonal_periods
                            try:
                                model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
                                fitted_model = model.fit(disp=False)
                                if fitted_model.aic < best_sarima_aic:
                                    best_sarima_aic = fitted_model.aic
                                    best_sarima_order = (p, d, q)
                                    best_sarima_seasonal_order = (P, D, Q, s)
                                    best_sarima_model = fitted_model
                            except:
                                continue
"""
    save_sarima_model_pickle (best_sarima_model, best_sarima_order, best_sarima_seasonal_order, sarima_model_path)


test_sarima_log = np.log(test + 1)
test_sarima_model = SARIMAX(test_sarima_log, order=best_sarima_order, seasonal_order=best_sarima_seasonal_order).fit()

# 2) Predict values in the test set using the best SARIMA model
sarima_forecast_log = best_sarima_model.forecast(steps=len(test))
# Convert forecasts back to original scale
sarima_forecast = np.exp(sarima_forecast_log) - 1

# 3) Evaluate the SARIMA model
#sarima_train_pred = best_sarima_model.fittedvalues
sarima_train_pred = np.exp(best_sarima_model.fittedvalues) - 1  # Convert back to original scale
sarima_test_pred = sarima_forecast

# Compute RMSE and R² for train and test sets
sarima_train_rmse = np.sqrt(mean_squared_error(train, sarima_train_pred))
sarima_test_rmse = np.sqrt(mean_squared_error(test, sarima_test_pred))
sarima_train_r2 = r2_score(train, sarima_train_pred)
sarima_test_r2 = r2_score(test, sarima_test_pred)

# Get AIC and BIC
train_sarima_aic = best_sarima_model.aic
train_sarima_bic = best_sarima_model.bic

test_sarima_aic = test_sarima_model.aic
test_sarima_bic = test_sarima_model.bic

sarima_evaluation_metrics_df = pd.DataFrame({
    "Set": ["Training", "Test"],
    "RMSE": [sarima_train_rmse, sarima_test_rmse],
    "R²": [sarima_train_r2, sarima_test_r2],
    "AIC": [train_sarima_aic, test_sarima_aic],
    "BIC": [train_sarima_bic, test_sarima_bic]
})

print("\SARIMA results:")
print(sarima_evaluation_metrics_df)

# Save evaluation metrics to a CSV file
sarima_evaluation_metrics_df.to_csv(sarima_csv_filename, index=False)

# 4) Plot the actual vs predicted values for SARIMA
plt.figure(figsize=(20, 6))
plt.plot(train.index, train, label='Train', color='lightblue')
plt.plot(test.index,  test, label='Test', color='lightgreen')
#  Add vertical line at the end of the training data:
plt.axvline(x=train.index[-1], color='gray', linestyle='--')
plt.plot(test.index, sarima_test_pred, label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height (metres)', fontsize=18)
plt.title(f"Actual vs Predicted Values (SARIMA{best_sarima_model.model.order} x {best_sarima_model.model.seasonal_order})", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)
plt.savefig("05 - Sarima Model.png", dpi=300)
plt.show()

plt.figure(figsize=(20, 6))
plt.plot(test.index, test, label='Test', color='lightgreen')
plt.plot(test.index, sarima_test_pred, label='Test forecasts', color='orange')
plt.ylabel('Maximum wave height (metres)', fontsize=18)
plt.title(f'SARIMA {best_sarima_model.model.order} x {best_sarima_model.model.seasonal_order} test time series and forecasts', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize=18, frameon=False)
plt.savefig("05 - Sarima Model - Test and forecasts.png", dpi=300)
plt.show()

# 5) Residual Analysis 

residuals = best_sarima_model.resid  # Log-scale residuals

# Histogram and QQ-Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True, bins=30)
plt.title("Histogram of Residuals")

plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot of Residuals")
plt.savefig("05 - SARIMA Residual analysis (Histogram and QQ-Plot).png", dpi=300)
plt.show()

# ACF and PACF plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_acf(residuals, lags=40, ax=plt.gca())
plt.subplot(1, 2, 2)
plot_pacf(residuals, lags=40, ax=plt.gca())
plt.savefig("05 - SARIMA Residual ACF and PACF (autocorrelation and partical autocorrelation).png", dpi=300)
plt.show()

# Ljung-Box Test for White Noise
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("\nLjung-Box Test for White Noise:")
print(lb_test)

"""
Ljung-Box Test for White Noise:
     lb_stat  lb_pvalue
10  8.244267   0.604991

Conclusion: Residuals are uncorrelated (white noise), meaning SARIMA effectively captures seasonality and trends.
"""

# 6) Confidence Interval Analysis

pred = best_sarima_model.get_forecast(steps=len(test))
pred_ci_log = pred.conf_int()
forecast_log = pred.predicted_mean  

# Convert forecast and confidence intervals back to original scale
forecast = np.exp(forecast_log) - 1  
pred_ci = np.exp(pred_ci_log) - 1  

# Plot Confidence Intervals
plt.figure(figsize=(10, 5))
plt.title(f'SARIMA {best_sarima_model.model.order} x {best_sarima_model.model.seasonal_order} Confidence Intervals in Forecasting', fontsize=16)

plt.plot(test.index, test, label="Test", color="lightgreen")
plt.plot(test.index, forecast, label="Test forecast", color="orange")

plt.fill_between(test.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='orange', alpha=0.2)
plt.ylabel('Maximum wave height (metres)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig("05 - SARIMA Model - Confidence Intervals in Forecasting.png", dpi=300)
plt.show()


"""
SARIMA Clearly Outperforms ARIMA 

Lower RMSE (Train: 0.5197, Test: 0.5451) vs. ARIMA (Train: 0.6848, Test: 0.7895) → More accurate predictions.
Higher R² (Train: 0.6185, Test: 0.6703) vs. ARIMA (Train: 0.3377, Test: 0.3083) → Better explanation of variance.
Better AIC/BIC values → SARIMA is a significantly better fit for the data.
ARIMA Has Poor Generalization 

High test RMSE (0.7895) → ARIMA fails to capture seasonality.
Low test R² (0.3083) → Model explains very little of the variance in test data.
Test AIC/BIC are much worse than SARIMA, confirming it is an inferior model.
SARIMA Captures Seasonality Well 

Lower RMSE and higher R² indicate SARIMA captures the wave height patterns better.
Test RMSE (0.5451) is close to Train RMSE (0.5197) → No overfitting.
Significantly lower AIC and BIC confirm SARIMA is the better model.
"""