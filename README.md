# **Ex.No: 08 – Moving Average Model and Exponential Smoothing**

**Date:** 25/10/2025  

## **Aim**
To implement Moving Average Model and Exponential Smoothing using Python for the BMW Car Sales dataset.

---

## **Algorithm**

1. Import the necessary libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`, and `statsmodels`.  
2. Load the BMW car sales dataset from the CSV file.  
3. Convert the `Year` column to datetime and set it as the index.  
4. Display the shape and first 10 rows of the dataset.  
5. Plot the original sales data (`Sales_Volume`) to visualize yearly sales trends.  
6. Compute the **Rolling Mean**:
   - Calculate the mean with a **window size of 5** and display the first 10 values.  
   - Calculate the mean with a **window size of 10** and display the first 20 values.  
   - Plot both rolling means along with the original data.  
7. Normalize the yearly sales data using **MinMaxScaler**.  
8. Split the data into **training (80%)** and **testing (20%)** subsets.  
9. Apply the **Exponential Smoothing** model (additive trend, multiplicative seasonality).  
10. Plot training, testing, and predicted data for visual evaluation.  
11. Compute **Root Mean Square Error (RMSE)** for performance evaluation.  
12. Display the variance and mean of the scaled dataset.  
13. Train the model on the full dataset and forecast future sales.  
14. Plot the future prediction of BMW car sales volume.

---

## **Program**

```python

#Name: Hiruthik Sudhakar
#Regno : 212223240054
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Read dataset
data = pd.read_csv("/content/BMW_Car_Sales_Classification.csv")

# Convert 'Year' to datetime and set as index
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

# Extract Sales_Volume column
sales_data = data[['Sales_Volume']]
print("Shape of the dataset:", sales_data.shape)
print("First 10 rows of the dataset:")
print(sales_data.head(10))

# Plot original data
plt.figure(figsize=(14, 7))
plt.plot(sales_data['Sales_Volume'], label='Original BMW Sales Volume', color='blue', alpha=0.7)
plt.title('Original BMW Car Sales Data Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Sales Volume', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Rolling Mean with window = 5
rolling_mean_5 = sales_data['Sales_Volume'].rolling(window=5).mean()
print("\nFirst 10 values of Rolling Mean (window=5):")
print(rolling_mean_5.head(10))

# Rolling Mean with window = 10
rolling_mean_10 = sales_data['Sales_Volume'].rolling(window=10).mean()
print("\nFirst 20 values of Rolling Mean (window=10):")
print(rolling_mean_10.head(20))

# Plot Rolling Means
plt.figure(figsize=(14, 7))
plt.plot(sales_data['Sales_Volume'], label='Original', color='blue', alpha=0.5)
plt.plot(rolling_mean_5, label='Rolling Mean (5)', color='orange', linewidth=2)
plt.plot(rolling_mean_10, label='Rolling Mean (10)', color='green', linewidth=2)
plt.title('Moving Average (Rolling Mean) of BMW Sales Volume Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Sales Volume', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Resample yearly and scale
sales_yearly = sales_data.resample('YS').mean()
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(sales_yearly.values.reshape(-1,1)).flatten(),
    index=sales_yearly.index
)
scaled_data = scaled_data + 1

# Train-test split
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# Exponential Smoothing
model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=3).fit()
predictions = model.forecast(steps=len(test_data))

# Plot Train/Test/Predictions
ax = train_data.plot(figsize=(14,7), label="Train Data")
predictions.plot(ax=ax, label="Predictions")
test_data.plot(ax=ax, label="Test Data")
ax.set_title('BMW Car Sales Forecast', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Scaled Sales Volume', fontsize=12)
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(test_data, predictions))
print("\nRoot Mean Square Error (RMSE):", rmse)

# Variance and Mean
print("Variance:", np.sqrt(scaled_data.var()), "Mean:", scaled_data.mean())

# Future Forecast
model_full = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=3).fit()
future_predictions = model_full.forecast(steps=5)

ax = scaled_data.plot(figsize=(14,7), label="Yearly Sales Data")
future_predictions.plot(ax=ax, label="Future Predictions")
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Scaled Sales Volume', fontsize=12)
ax.set_title('Future Prediction of BMW Car Sales Volume', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
```


### OUTPUT:
Original Data:

<br>

```
Shape of the dataset: (50000, 1)
First 10 rows of the dataset:
            Sales_Volume
Year                    
2016-01-01          8300
2013-01-01          3428
2022-01-01          6994
2024-01-01          4047
2020-01-01          3080
2017-01-01          1232
2022-01-01          7949
2014-01-01           632
2016-01-01          8944
2019-01-01          4411
```

<img width="1389" height="690" alt="download" src="https://github.com/user-attachments/assets/d9f0586d-3e15-4a04-8d8f-94d0a0bd7a18" />



Moving Average:
<br>
```
First 10 values of Rolling Mean (window=5):
Year
2016-01-01       NaN
2013-01-01       NaN
2022-01-01       NaN
2024-01-01       NaN
2020-01-01    5169.8
2017-01-01    3756.2
2022-01-01    4660.4
2014-01-01    3388.0
2016-01-01    4367.4
2019-01-01    4633.6
Name: Sales_Volume, dtype: float64
```
```
First 20 values of Rolling Mean (window=10):
Year
2016-01-01       NaN
2013-01-01       NaN
2022-01-01       NaN
2024-01-01       NaN
2020-01-01       NaN
2017-01-01       NaN
2022-01-01       NaN
2014-01-01       NaN
2016-01-01       NaN
2019-01-01    4901.7
2012-01-01    4121.7
2016-01-01    4604.1
2020-01-01    4715.8
2020-01-01    4777.9
2017-01-01    5199.0
2014-01-01    5852.3
2013-01-01    5613.5
2017-01-01    6525.8
2017-01-01    6119.4
2012-01-01    6388.7
Name: Sales_Volume, dtype: float64
<br>
```
<img width="1389" height="690" alt="image" src="https://github.com/user-attachments/assets/36297831-6a51-4a16-b514-22e1a92fd8ba" />


Plot Transform Dataset:

<img width="1389" height="690" alt="image" src="https://github.com/user-attachments/assets/310e621f-1ca8-41b7-93e9-3c36cafaf030" />


Exponential Smoothing:

<img width="1029" height="509" alt="image" src="https://github.com/user-attachments/assets/e410d2da-e3e3-4ee8-a682-da2930a3b2c5" />

Performance metrics:

<img width="447" height="46" alt="image" src="https://github.com/user-attachments/assets/201c2b14-f0a3-4dd7-936f-301d8859460e" />



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
