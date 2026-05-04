import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


airline_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
airline = pd.read_csv(airline_url, parse_dates=["Month"])
airline.set_index("Month", inplace=True)
data = airline["Passengers"].astype(float).values
time = np.arange(len(data))

plt.plot(time, data)
plt.title("Non-Stationary Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()


def check_stationarity(ts):
    result = adfuller(ts)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])


stationary_data = np.diff(data, n=1)

print("\nDickey-Fuller Test for Stationarity:")
check_stationarity(stationary_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(stationary_data, ax=ax1, lags=20)
plot_pacf(stationary_data, ax=ax2, lags=20)
plt.show()

order = (1, 1, 1)
model = ARIMA(data, order=order)
results = model.fit()

forecast_steps = 10
forecast_values = results.get_forecast(steps=forecast_steps).predicted_mean

plt.plot(time, data, label="Original Data")
plt.plot(np.arange(len(data), len(data) + forecast_steps), forecast_values, label="Forecast")
plt.title("Time Series Forecasting with ARIMA")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
