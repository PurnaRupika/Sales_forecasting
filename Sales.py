# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
train = pd.read_csv("train.csv")
features = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")

# %%
print("Train Shape:", train.shape)
print(train.head())

# %%
print("\nFeatures Shape:", features.shape)
print(features.head())

# %%
print("\nStores Shape:", stores.shape)
print(stores.head())

# %%
print("Missing values in train:\n", train.isnull().sum())
print("\nMissing values in features:\n", features.isnull().sum())
print("\nMissing values in stores:\n", stores.isnull().sum())

# %%
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
features[markdown_cols] = features[markdown_cols].fillna(0)

# %%
# Sort by Store and Date to maintain time order
features = features.sort_values(['Store', 'Date'])

# Forward fill CPI and Unemployment per Store
features[['CPI', 'Unemployment']] = features.groupby('Store')[['CPI', 'Unemployment']].ffill().bfill()

# %%
print("Missing values after cleaning:\n", features.isnull().sum())


# %%
print("Train Description:")
print(train.describe())

print("\nFeatures Description:")
print(features.describe())


# %%
train['Date'] = pd.to_datetime(train['Date'])
features['Date'] = pd.to_datetime(features['Date'])


# %%
# Merge train with features and stores
data = pd.merge(train, features, on=["Store", "Date"], how='left')
data = pd.merge(data, stores, on="Store", how='left')

print("Merged Data Shape:", data.shape)
data.head()


# %%
# Example: Sales trend for one store and department
sample = data[(data['Store'] == 1) & (data['Dept'] == 1)]
sample = sample.sort_values("Date")

plt.figure(figsize=(12, 6))
plt.plot(sample['Date'], sample['Weekly_Sales'])
plt.title('Weekly Sales Trend - Store 1, Dept 1')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.grid(True)
plt.show()


# %%
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week


# %%
sales_per_store = data.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sales_per_store.plot(kind='bar')
plt.title('Total Sales per Store')
plt.xlabel('Store')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()


# %%
monthly_sales = data.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_sales, x='Month', y='Weekly_Sales', hue='Year', marker='o')
plt.title('Monthly Sales Trend by Year')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()


# %%
dept_sales = data.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
dept_sales.plot(kind='bar', color='skyblue')
plt.title('Total Sales by Department')
plt.xlabel('Department')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()


# %%
# Filter data
store_dept_sales = data[(data['Store'] == 1) & (data['Dept'] == 1)]
store_dept_sales = store_dept_sales.sort_values('Date')

# Plot
plt.figure(figsize=(14, 6))
plt.plot(store_dept_sales['Date'], store_dept_sales['Weekly_Sales'], marker='o')
plt.title('Weekly Sales Trend - Store 1, Department 1')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
overall_weekly_sales = data.groupby('Date')['Weekly_Sales'].sum().reset_index()

plt.figure(figsize=(14, 6))
plt.plot(overall_weekly_sales['Date'], overall_weekly_sales['Weekly_Sales'], color='green')
plt.title('Total Weekly Sales Trend (All Stores & Departments)')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# %%
# Filter data
new_data = data[(data['Store'] == 1) & (data['Dept'] == 1)]

# Sort and group by Date (aggregate sales per week)
ts = new_data.groupby('Date')['Weekly_Sales'].sum().sort_index()


# %%
result = adfuller(ts)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] > 0.05:
    print("Series is NOT stationary – differencing needed.")
else:
    print("Series is stationary – no differencing needed.")


# %%
ts_diff = ts.diff().dropna()

# Re-run ADF test
result_diff = adfuller(ts_diff)
print("After differencing - p-value:", result_diff[1])


# %%
model = ARIMA(ts, order=(1, 1, 1))
model_fit = model.fit()

# Summary
print(model_fit.summary())


# %%
# Forecast next 12 weeks
forecast = model_fit.forecast(steps=12)

# Plot actual + forecast
plt.figure(figsize=(14, 6))
plt.plot(ts, label='Actual Sales')
plt.plot(pd.date_range(ts.index[-1], periods=12, freq='W'), forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast - Store 1, Dept 1')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Load test dataset
test = pd.read_csv("test.csv")
test['Date'] = pd.to_datetime(test['Date'])

# Filter for Store 1, Dept 1 (same as trained model)
test_filtered = test[(test['Store'] == 1) & (test['Dept'] == 1)].sort_values('Date')


# %%
# Use Store 1, Dept 1 again
ts = data[(data['Store'] == 1) & (data['Dept'] == 1)]
ts = ts.groupby('Date')['Weekly_Sales'].sum().sort_index()

# Use last 12 weeks as test, rest as train
train_ts = ts[:-12]
test_ts = ts[-12:]



# %%
model = ARIMA(train_ts, order=(1, 1, 1))
model_fit = model.fit()

# %%
forecast = model_fit.forecast(steps=12)

# Align forecast with test dates
forecast.index = test_ts.index

# %%
plt.figure(figsize=(14, 6))
plt.plot(train_ts.index, train_ts, label='Train')
plt.plot(test_ts.index, test_ts, label='Actual')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Model Forecast vs Actual (Backtest - Last 12 Weeks)')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test_ts, forecast)
rmse = np.sqrt(mean_squared_error(test_ts, forecast))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


# %%
# Create a working DataFrame
df = data.copy()

# Create time-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

# Encode IsHoliday
df['IsHoliday_x'] = df['IsHoliday_x'].astype(int)

# Target
y = df['Weekly_Sales']

# Select useful features
features = [
    'Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
    'Type', 'Size', 'IsHoliday_x', 'Year', 'Month', 'Week'
]

# Convert categorical 'Type' to numeric
df['Type'] = df['Type'].map({'A': 0, 'B': 1, 'C': 2})

X = df[features]


# %%
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_val)

# Metrics
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


# %%
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()


# %%
top_features = feature_importance_df['Feature'].head(8).tolist()

# Re-train with top features
X_train_top = X_train[top_features]
X_val_top = X_val[top_features]

rf_top = RandomForestRegressor(n_estimators=100, random_state=42)
rf_top.fit(X_train_top, y_train)
y_pred_top = rf_top.predict(X_val_top)

# Metrics
mae_top = mean_absolute_error(y_val, y_pred_top)
rmse_top = np.sqrt(mean_squared_error(y_val, y_pred_top))

print(f"[Optimized] MAE: {mae_top:.2f}")
print(f"[Optimized] RMSE: {rmse_top:.2f}")


# %%
# Assume you’ve cleaned and prepared `df` with feature columns and target `y`
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)  # Train on full available data


# %%
# Get unique stores and departments
stores = df['Store'].unique()
depts = df['Dept'].unique()

# Simulate weekly dates (next year)
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(weeks=1), periods=52, freq='W')

# Build future prediction DataFrame
future_data = []

for store in stores:
    for dept in depts:
        for date in future_dates:
            future_data.append({'Store': store, 'Dept': dept, 'Date': date})

future_df = pd.DataFrame(future_data)


# %%
# Load original supporting data
features = pd.read_csv("features.csv")
stores_info = pd.read_csv("stores.csv")
features['Date'] = pd.to_datetime(features['Date'])

# Merge features and store info
future_df = pd.merge(future_df, features, on=['Store', 'Date'], how='left')
future_df = pd.merge(future_df, stores_info, on='Store', how='left')

# Fill missing markdowns as 0 (likely for future)
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in markdown_cols:
    if col in future_df.columns:
        future_df[col] = future_df[col].fillna(0)

# Fill economic indicators
future_df = future_df.sort_values(['Store', 'Date'])
future_df[['CPI', 'Unemployment']] = future_df.groupby('Store')[['CPI', 'Unemployment']].ffill().bfill()

# Add CPI and Unemployment if missing after merge
if 'CPI' not in future_df.columns:
    future_df['CPI'] = np.nan
if 'Unemployment' not in future_df.columns:
    future_df['Unemployment'] = np.nan

# Fill them using forward/backward fill per Store (or global median fallback)
future_df = future_df.sort_values(['Store', 'Date'])
future_df[['CPI', 'Unemployment']] = (
    future_df.groupby('Store')[['CPI', 'Unemployment']].ffill().bfill()
)

# If still missing (e.g., entire store missing values), fill with median
future_df['CPI'] = future_df['CPI'].fillna(df['CPI'].median())
future_df['Unemployment'] = future_df['Unemployment'].fillna(df['Unemployment'].median())


# Encode 'Type'
future_df['Type'] = future_df['Type'].map({'A': 0, 'B': 1, 'C': 2})

# Time features
future_df['Year'] = future_df['Date'].dt.year
future_df['Month'] = future_df['Date'].dt.month
future_df['Week'] = future_df['Date'].dt.isocalendar().week.astype(int)

# Fill missing values in IsHoliday with 0 (not a holiday)
future_df['IsHoliday'] = future_df['IsHoliday'].fillna(0).astype(int)


# Encode holiday flag
future_df['IsHoliday'] = future_df['IsHoliday'].astype(int)


# %%
print(type(features))
print(features)


# %%
# Redefine features correctly
features = [
    'Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
    'Type', 'Size', 'IsHoliday_x', 'Year', 'Month', 'Week'
]


# %%
future_X = future_df[features]
future_df['Predicted_Sales'] = rf_model.predict(future_X)


# %%
# Example: Total predicted sales for next year
predicted_total = future_df['Predicted_Sales'].sum()
print(f"Predicted Total Sales for Next Year: {predicted_total:.2f}")

# Save to CSV
future_df[['Store', 'Dept', 'Date', 'Predicted_Sales']].to_csv("next_year_sales_predictions.csv", index=False)


# %%



