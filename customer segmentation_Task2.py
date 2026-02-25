"""
SALES DATA ANALYSIS AND FORECASTING SYSTEM
============================================
This system analyzes past sales trends and predicts future revenue
for retail/e-commerce businesses to optimize inventory and marketing.
"""

# ============================================
# STEP 1: IMPORT LIBRARIES
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import itertools

warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("talk")

print("="*60)
print("SALES DATA ANALYSIS AND FORECASTING SYSTEM")
print("="*60)
print("✅ Libraries imported successfully")

# ============================================
# STEP 2: DATA LOADING
# ============================================
print("\n" + "="*60)
print("STEP 2: LOADING DATASET")
print("="*60)

# Use the same Online Retail dataset
file_path = r"C:\Users\Dell\customer segmentation\online_retail_II.csv"
print(f"📂 Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ File not found. Please check the path.")
    # Alternative: Create sample data if file not found
    print("\n📌 Creating sample sales data for demonstration...")
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    df = pd.DataFrame({
        'InvoiceDate': np.random.choice(dates, 10000),
        'Quantity': np.random.randint(1, 20, 10000),
        'Price': np.random.uniform(5, 100, 10000).round(2),
        'Country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany'], 10000)
    })
    df['TotalSales'] = df['Quantity'] * df['Price']
    df['Invoice'] = [f'INV-{i:06d}' for i in range(10000)]
    print("✅ Sample dataset created!")

# ============================================
# STEP 3: DATA CLEANING & PREPROCESSING
# ============================================
print("\n" + "="*60)
print("STEP 3: DATA CLEANING & PREPROCESSING")
print("="*60)

# Make a copy
sales_df = df.copy()
initial_rows = len(sales_df)

# Standardize column names
column_mapping = {}
for col in sales_df.columns:
    col_lower = col.lower().strip()
    if 'invoice' in col_lower and 'date' in col_lower:
        column_mapping[col] = 'InvoiceDate'
    elif 'quantity' in col_lower:
        column_mapping[col] = 'Quantity'
    elif 'price' in col_lower or 'unit' in col_lower:
        column_mapping[col] = 'Price'
    elif 'country' in col_lower:
        column_mapping[col] = 'Country'
    elif 'description' in col_lower:
        column_mapping[col] = 'Description'

if column_mapping:
    sales_df.rename(columns=column_mapping, inplace=True)
    print("✅ Standardized column names")

# Convert InvoiceDate to datetime
if 'InvoiceDate' in sales_df.columns:
    try:
        sales_df['InvoiceDate'] = pd.to_datetime(sales_df['InvoiceDate'])
        print("✅ Converted InvoiceDate to datetime")
    except:
        print("⚠️ Could not convert InvoiceDate, using index as date")
        sales_df['InvoiceDate'] = pd.date_range(start='2023-01-01', periods=len(sales_df), freq='H')

# Remove invalid data
if 'Quantity' in sales_df.columns and 'Price' in sales_df.columns:
    sales_df = sales_df[sales_df['Quantity'] > 0]
    sales_df = sales_df[sales_df['Price'] > 0]
    print(f"✅ Removed invalid quantities/prices: {initial_rows - len(sales_df)} rows removed")

# Create TotalSales column if not exists
if 'TotalSales' not in sales_df.columns:
    sales_df['TotalSales'] = sales_df['Quantity'] * sales_df['Price']
    print("✅ Created TotalSales column")

print(f"\n📊 Cleaned dataset shape: {sales_df.shape}")

# ============================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================
print("\n" + "="*60)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("="*60)

# Basic statistics
print("\n📊 Sales Statistics:")
print(sales_df[['Quantity', 'Price', 'TotalSales']].describe())

# Sales by country
if 'Country' in sales_df.columns:
    print("\n📊 Top 10 Countries by Sales:")
    country_sales = sales_df.groupby('Country')['TotalSales'].sum().sort_values(ascending=False).head(10)
    print(country_sales)

# Create time-based features
sales_df['Year'] = sales_df['InvoiceDate'].dt.year
sales_df['Month'] = sales_df['InvoiceDate'].dt.month
sales_df['Day'] = sales_df['InvoiceDate'].dt.day
sales_df['DayOfWeek'] = sales_df['InvoiceDate'].dt.dayofweek
sales_df['Hour'] = sales_df['InvoiceDate'].dt.hour
sales_df['Weekday'] = sales_df['DayOfWeek'].map({
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
    4: 'Friday', 5: 'Saturday', 6: 'Sunday'
})

print("✅ Created time-based features")

# ============================================
# STEP 5: VISUALIZATION 1 - SALES OVERVIEW
# ============================================
print("\n" + "="*60)
print("STEP 5: CREATING SALES VISUALIZATIONS")
print("="*60)

fig = plt.figure(figsize=(20, 12))

# 1. Daily Sales Trend
ax1 = plt.subplot(3, 3, 1)
daily_sales = sales_df.groupby(sales_df['InvoiceDate'].dt.date)['TotalSales'].sum()
daily_sales.plot(ax=ax1, color='blue', alpha=0.7)
ax1.set_title('Daily Sales Trend', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Sales ($)')
ax1.tick_params(axis='x', rotation=45)

# 2. Sales by Month
ax2 = plt.subplot(3, 3, 2)
monthly_sales = sales_df.groupby('Month')['TotalSales'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax2.bar(months[:len(monthly_sales)], monthly_sales.values, color='orange', alpha=0.7)
ax2.set_title('Average Sales by Month', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Avg Sales ($)')
ax2.tick_params(axis='x', rotation=45)

# 3. Sales by Day of Week
ax3 = plt.subplot(3, 3, 3)
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_sales = sales_df.groupby('Weekday')['TotalSales'].mean().reindex(weekday_order)
ax3.bar(weekday_sales.index, weekday_sales.values, color='green', alpha=0.7)
ax3.set_title('Average Sales by Day of Week', fontsize=14, fontweight='bold')
ax3.set_xlabel('Day')
ax3.set_ylabel('Avg Sales ($)')
ax3.tick_params(axis='x', rotation=45)

# 4. Sales Distribution
ax4 = plt.subplot(3, 3, 4)
sales_df['TotalSales'].hist(bins=50, ax=ax4, color='purple', alpha=0.7, edgecolor='black')
ax4.set_title('Sales Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Sales Amount ($)')
ax4.set_ylabel('Frequency')

# 5. Top Products (if Description exists)
ax5 = plt.subplot(3, 3, 5)
if 'Description' in sales_df.columns:
    top_products = sales_df.groupby('Description')['TotalSales'].sum().nlargest(10)
    ax5.barh(range(len(top_products)), top_products.values, color='red', alpha=0.7)
    ax5.set_yticks(range(len(top_products)))
    ax5.set_yticklabels([p[:20] + '...' if len(p) > 20 else p for p in top_products.index])
    ax5.set_title('Top 10 Products by Sales', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Total Sales ($)')
else:
    ax5.text(0.5, 0.5, 'Product data not available', ha='center', va='center', fontsize=14)
    ax5.set_title('Top Products', fontsize=14, fontweight='bold')

# 6. Sales by Hour
ax6 = plt.subplot(3, 3, 6)
hourly_sales = sales_df.groupby('Hour')['TotalSales'].mean()
ax6.plot(hourly_sales.index, hourly_sales.values, marker='o', color='brown', linewidth=2)
ax6.set_title('Average Sales by Hour', fontsize=14, fontweight='bold')
ax6.set_xlabel('Hour of Day')
ax6.set_ylabel('Avg Sales ($)')
ax6.grid(True, alpha=0.3)

# 7. Country Sales (if Country exists)
ax7 = plt.subplot(3, 3, 7)
if 'Country' in sales_df.columns:
    country_sales_top = country_sales.head(8)
    ax7.pie(country_sales_top.values, labels=country_sales_top.index, autopct='%1.1f%%', 
            startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, 8)))
    ax7.set_title('Sales by Country (Top 8)', fontsize=14, fontweight='bold')
else:
    ax7.text(0.5, 0.5, 'Country data not available', ha='center', va='center', fontsize=14)
    ax7.set_title('Sales by Country', fontsize=14, fontweight='bold')

# 8. Quantity vs Price Scatter
ax8 = plt.subplot(3, 3, 8)
sample_df = sales_df.sample(min(1000, len(sales_df)))
ax8.scatter(sample_df['Quantity'], sample_df['Price'], alpha=0.5, c='teal')
ax8.set_title('Quantity vs Price', fontsize=14, fontweight='bold')
ax8.set_xlabel('Quantity')
ax8.set_ylabel('Price ($)')

# 9. Sales Correlation Heatmap
ax9 = plt.subplot(3, 3, 9)
numeric_cols = sales_df.select_dtypes(include=[np.number]).columns
corr_matrix = sales_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax9, fmt='.2f')
ax9.set_title('Feature Correlations', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('sales_eda_dashboard.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ EDA Dashboard saved as 'sales_eda_dashboard.png'")

# ============================================
# STEP 6: TIME SERIES AGGREGATION
# ============================================
print("\n" + "="*60)
print("STEP 6: TIME SERIES AGGREGATION")
print("="*60)

# Aggregate sales by different time periods
daily_sales_ts = sales_df.groupby(sales_df['InvoiceDate'].dt.date)['TotalSales'].sum().reset_index()
daily_sales_ts.columns = ['Date', 'Sales']
daily_sales_ts['Date'] = pd.to_datetime(daily_sales_ts['Date'])
daily_sales_ts = daily_sales_ts.set_index('Date').sort_index()

# Weekly aggregation
weekly_sales = daily_sales_ts.resample('W').sum()

# Monthly aggregation
monthly_sales_ts = daily_sales_ts.resample('M').sum()

print(f"📊 Time Series Data:")
print(f"   Daily data: {len(daily_sales_ts)} days")
print(f"   Weekly data: {len(weekly_sales)} weeks")
print(f"   Monthly data: {len(monthly_sales_ts)} months")

print("\n📊 Daily Sales Statistics:")
print(daily_sales_ts.describe())

# Plot time series at different frequencies
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Daily
daily_sales_ts.plot(ax=axes[0], color='blue', alpha=0.7)
axes[0].set_title('Daily Sales', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Sales ($)')
axes[0].grid(True, alpha=0.3)

# Weekly
weekly_sales.plot(ax=axes[1], color='green', alpha=0.7)
axes[1].set_title('Weekly Sales', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Sales ($)')
axes[1].grid(True, alpha=0.3)

# Monthly
monthly_sales_ts.plot(ax=axes[2], color='red', alpha=0.7)
axes[2].set_title('Monthly Sales', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Sales ($)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sales_time_series.png', dpi=100)
plt.show()
print("✅ Time series plot saved as 'sales_time_series.png'")

# ============================================
# STEP 7: TIME SERIES DECOMPOSITION
# ============================================
print("\n" + "="*60)
print("STEP 7: TIME SERIES DECOMPOSITION")
print("="*60)

# Decompose time series (using monthly data for better visualization)
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Ensure we have at least 2 years of data for decomposition
    if len(monthly_sales_ts) >= 24:
        decomposition = seasonal_decompose(monthly_sales_ts, model='additive', period=12)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Original
        decomposition.observed.plot(ax=axes[0])
        axes[0].set_title('Original Monthly Sales', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Sales ($)')
        
        # Trend
        decomposition.trend.plot(ax=axes[1])
        axes[1].set_title('Trend Component', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Sales ($)')
        
        # Seasonal
        decomposition.seasonal.plot(ax=axes[2])
        axes[2].set_title('Seasonal Component', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Sales ($)')
        
        # Residual
        decomposition.resid.plot(ax=axes[3])
        axes[3].set_title('Residual Component', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('Sales ($)')
        
        plt.tight_layout()
        plt.savefig('time_series_decomposition.png', dpi=100)
        plt.show()
        print("✅ Time series decomposition saved as 'time_series_decomposition.png'")
    else:
        print("⚠️ Insufficient data for decomposition (need at least 24 months)")
except Exception as e:
    print(f"⚠️ Could not perform time series decomposition: {e}")

# ============================================
# STEP 8: FEATURE ENGINEERING FOR FORECASTING
# ============================================
print("\n" + "="*60)
print("STEP 8: FEATURE ENGINEERING FOR FORECASTING")
print("="*60)

# Create features for ML models
forecast_df = daily_sales_ts.copy()
forecast_df['DayOfWeek'] = forecast_df.index.dayofweek
forecast_df['Month'] = forecast_df.index.month
forecast_df['Quarter'] = forecast_df.index.quarter
forecast_df['Year'] = forecast_df.index.year
forecast_df['DayOfYear'] = forecast_df.index.dayofyear
forecast_df['WeekOfYear'] = forecast_df.index.isocalendar().week.astype(int)
forecast_df['IsWeekend'] = forecast_df['DayOfWeek'].isin([5, 6]).astype(int)

# Create lag features
for lag in [1, 2, 3, 7, 14, 30]:
    forecast_df[f'Lag_{lag}'] = forecast_df['Sales'].shift(lag)

# Create rolling statistics
for window in [7, 14, 30]:
    forecast_df[f'RollingMean_{window}'] = forecast_df['Sales'].rolling(window=window).mean()
    forecast_df[f'RollingStd_{window}'] = forecast_df['Sales'].rolling(window=window).std()

# Drop NaN values created by lag/rolling features
forecast_df = forecast_df.dropna()

print(f"✅ Created forecasting features: {list(forecast_df.columns)}")
print(f"📊 Final dataset shape: {forecast_df.shape}")

# ============================================
# STEP 9: TRAIN-TEST SPLIT
# ============================================
print("\n" + "="*60)
print("STEP 9: TRAIN-TEST SPLIT")
print("="*60)

# Split data chronologically
train_size = int(len(forecast_df) * 0.8)
train_data = forecast_df.iloc[:train_size]
test_data = forecast_df.iloc[train_size:]

print(f"📊 Training data: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
print(f"📊 Testing data: {len(test_data)} days ({test_data.index[0].date()} to {test_data.index[-1].date()})")

# Prepare features for ML
feature_columns = [col for col in forecast_df.columns if col not in ['Sales']]
X_train = train_data[feature_columns]
y_train = train_data['Sales']
X_test = test_data[feature_columns]
y_test = test_data['Sales']

# Scale features for ML
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled for ML models")

# ============================================
# STEP 10: ML MODELS FOR FORECASTING
# ============================================
print("\n" + "="*60)
print("STEP 10: TRAINING ML MODELS")
print("="*60)

# Dictionary to store models and their predictions
models = {}
predictions = {}
metrics = {}

# 1. Linear Regression
print("\n📊 Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
models['Linear Regression'] = lr_model
predictions['Linear Regression'] = lr_pred

# Calculate metrics
mae = mean_absolute_error(y_test, lr_pred)
mse = mean_squared_error(y_test, lr_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, lr_pred)
metrics['Linear Regression'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
print(f"   MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R2: {r2:.4f}")

# 2. Random Forest
print("\n📊 Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
models['Random Forest'] = rf_model
predictions['Random Forest'] = rf_pred

# Calculate metrics
mae = mean_absolute_error(y_test, rf_pred)
mse = mean_squared_error(y_test, rf_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, rf_pred)
metrics['Random Forest'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
print(f"   MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R2: {r2:.4f}")

# Feature importance for Random Forest
if hasattr(rf_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))

# ============================================
# STEP 11: TIME SERIES MODELS (ARIMA & ETS)
# ============================================
print("\n" + "="*60)
print("STEP 11: TRAINING TIME SERIES MODELS")
print("="*60)

# Prepare time series data
ts_train = train_data['Sales']
ts_test = test_data['Sales']

# 3. Exponential Smoothing (ETS)
print("\n📊 Training Exponential Smoothing...")
try:
    ets_model = ExponentialSmoothing(
        ts_train, 
        seasonal_periods=7, 
        trend='add', 
        seasonal='add'
    ).fit()
    ets_pred = ets_model.forecast(len(ts_test))
    models['ETS'] = ets_model
    predictions['ETS'] = ets_pred.values
    
    # Calculate metrics
    mae = mean_absolute_error(ts_test, ets_pred)
    rmse = np.sqrt(mean_squared_error(ts_test, ets_pred))
    r2 = r2_score(ts_test, ets_pred)
    metrics['ETS'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"   MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R2: {r2:.4f}")
except Exception as e:
    print(f"⚠️ ETS model failed: {e}")

# 4. ARIMA
print("\n📊 Training ARIMA...")
try:
    # Find best ARIMA parameters using AIC
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    
    best_aic = float('inf')
    best_order = None
    best_arima_model = None
    
    for order in pdq:
        try:
            model = ARIMA(ts_train, order=order)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
                best_arima_model = model_fit
        except:
            continue
    
    if best_arima_model:
        arima_pred = best_arima_model.forecast(len(ts_test))
        models['ARIMA'] = best_arima_model
        predictions['ARIMA'] = arima_pred.values
        
        # Calculate metrics
        mae = mean_absolute_error(ts_test, arima_pred)
        rmse = np.sqrt(mean_squared_error(ts_test, arima_pred))
        r2 = r2_score(ts_test, arima_pred)
        metrics['ARIMA'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"   Best ARIMA{best_order}: MAE=${mae:.2f}, RMSE=${rmse:.2f}, R2={r2:.4f}")
except Exception as e:
    print(f"⚠️ ARIMA model failed: {e}")

# ============================================
# STEP 12: MODEL COMPARISON
# ============================================
print("\n" + "="*60)
print("STEP 12: MODEL COMPARISON")
print("="*60)

# Create comparison DataFrame
comparison_df = pd.DataFrame(metrics).T
print("\n📊 Model Performance Comparison:")
print(comparison_df.round(2))

# Find best model based on RMSE
best_model = comparison_df['RMSE'].idxmin()
print(f"\n✅ Best Model: {best_model} (RMSE: ${comparison_df.loc[best_model, 'RMSE']:.2f})")

# Plot predictions vs actual
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (model_name, pred_values) in enumerate(predictions.items()):
    if idx < 4:
        ax = axes[idx]
        ax.plot(test_data.index, y_test, label='Actual', color='blue', linewidth=2)
        ax.plot(test_data.index, pred_values, label='Predicted', color='red', linestyle='--', linewidth=2)
        ax.set_title(f'{model_name} Predictions\nRMSE: ${metrics[model_name]["RMSE"]:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_predictions_comparison.png', dpi=100)
plt.show()
print("✅ Model predictions comparison saved as 'model_predictions_comparison.png'")

# ============================================
# STEP 13: FUTURE FORECASTING
# ============================================
print("\n" + "="*60)
print("STEP 13: FUTURE SALES FORECASTING")
print("="*60)

# Forecast next 30 days using the best model
forecast_days = 30
last_date = test_data.index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

print(f"\n📊 Forecasting next {forecast_days} days from {last_date.date()} to {future_dates[-1].date()}")

# Use best model for forecasting
if best_model == 'ETS' and 'ETS' in models:
    future_forecast = models['ETS'].forecast(forecast_days)
    forecast_values = future_forecast.values
elif best_model == 'ARIMA' and 'ARIMA' in models:
    future_forecast = models['ARIMA'].forecast(forecast_days)
    forecast_values = future_forecast.values
elif best_model == 'Random Forest':
    # For RF, we need to create future features
    print("⚠️ Random Forest requires feature engineering for future dates")
    print("   Using simple trend extrapolation instead")
    # Simple linear trend for demonstration
    z = np.polyfit(range(len(ts_test)), ts_test.values, 1)
    trend = np.poly1d(z)
    last_trend = trend(len(ts_test) - 1)
    forecast_values = [last_trend * (1 + i*0.01) for i in range(forecast_days)]
else:
    # Default to linear regression trend
    print("   Using linear trend for forecast")
    z = np.polyfit(range(len(ts_test)), ts_test.values, 1)
    trend = np.poly1d(z)
    forecast_values = [trend(len(ts_test) + i) for i in range(forecast_days)]

# Plot historical and forecast
plt.figure(figsize=(15, 7))

# Plot historical data (last 90 days)
historical_days = 90
plt.plot(daily_sales_ts.index[-historical_days:], 
         daily_sales_ts.values[-historical_days:], 
         label='Historical Sales', color='blue', linewidth=2)

# Plot forecast
plt.plot(future_dates, forecast_values, 
         label=f'{best_model} Forecast', color='red', linestyle='--', linewidth=2)

# Add confidence interval (simulated)
plt.fill_between(future_dates, 
                 [v * 0.85 for v in forecast_values], 
                 [v * 1.15 for v in forecast_values], 
                 alpha=0.2, color='red', label='Confidence Interval (85-115%)')

plt.title(f'Sales Forecast - Next {forecast_days} Days', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('future_sales_forecast.png', dpi=100)
plt.show()
print("✅ Future forecast saved as 'future_sales_forecast.png'")

# ============================================
# STEP 14: FORECAST SUMMARY
# ============================================
print("\n" + "="*60)
print("STEP 14: FORECAST SUMMARY")
print("="*60)

forecast_summary = pd.DataFrame({
    'Date': future_dates,
    'Forecasted_Sales': forecast_values,
    'Lower_Bound': [v * 0.85 for v in forecast_values],
    'Upper_Bound': [v * 1.15 for v in forecast_values]
})

print("\n📊 Next 7 Days Forecast:")
print(forecast_summary.head(7).to_string(index=False))

print(f"\n📊 Forecast Statistics:")
print(f"   Total Forecasted Revenue (30 days): ${sum(forecast_values):,.2f}")
print(f"   Average Daily Forecast: ${np.mean(forecast_values):,.2f}")
print(f"   Peak Day: {future_dates[np.argmax(forecast_values)].date()} (${max(forecast_values):,.2f})")
print(f"   Lowest Day: {future_dates[np.argmin(forecast_values)].date()} (${min(forecast_values):,.2f})")

# ============================================
# STEP 15: BUSINESS INSIGHTS & RECOMMENDATIONS
# ============================================
print("\n" + "="*60)
print("STEP 15: BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*60)

# Generate insights
print("\n📈 Key Insights:")
print("   " + "-" * 40)

# Best performing day
best_day = weekday_sales.idxmax()
print(f"   • Best performing day: {best_day} (${weekday_sales.max():,.2f} avg)")

# Best performing month
best_month_idx = monthly_sales.idxmax()
best_month = months[best_month_idx - 1]
print(f"   • Best performing month: {best_month} (${monthly_sales.max():,.2f} avg)")

# Peak hour
peak_hour = hourly_sales.idxmax()
print(f"   • Peak sales hour: {peak_hour}:00 (${hourly_sales.max():,.2f} avg)")

# Seasonal pattern
if len(monthly_sales) >= 12:
    q4_avg = monthly_sales.loc[[10, 11, 12]].mean() if 12 in monthly_sales.index else monthly_sales.tail(3).mean()
    q2_avg = monthly_sales.loc[[4, 5, 6]].mean() if 6 in monthly_sales.index else monthly_sales.iloc[3:6].mean()
    seasonal_boost = ((q4_avg - q2_avg) / q2_avg * 100)
    print(f"   • Q4 sales boost: {seasonal_boost:.1f}% higher than Q2")

print("\n💡 Business Recommendations:")
print("   " + "-" * 40)
print("   1. Inventory Planning:")
print(f"      • Stock up for {best_month} sales peak")
if 'seasonal_boost' in locals():
    print(f"      • Plan for {seasonal_boost:.1f}% increase in Q4")
else:
    print("      • Monitor seasonal patterns")
print("      • Maintain safety stock for high-demand periods")

print("\n   2. Marketing Strategy:")
print(f"      • Run promotions on {best_day}s to maximize sales")
print(f"      • Target marketing during peak hour ({peak_hour}:00)")
print("      • Create special campaigns for holiday seasons")

print("\n   3. Staff Scheduling:")
print(f"      • Schedule more staff on {best_day}s")
print(f"      • Increase coverage during peak hours ({peak_hour}:00)")
print("      • Plan time-offs during slow periods")

print("\n   4. Revenue Optimization:")
print(f"      • Expected revenue next 30 days: ${sum(forecast_values):,.2f}")
growth_target = ((np.mean(forecast_values) / np.mean(daily_sales_ts.values[-30:]) - 1) * 100)
print(f"      • Growth target: {growth_target:.1f}%")
print("      • Focus on upselling and cross-selling")

# ============================================
# STEP 16: EXPORT RESULTS
# ============================================
print("\n" + "="*60)
print("STEP 16: EXPORTING RESULTS")
print("="*60)

# Save forecast to CSV
forecast_file = 'sales_forecast_results.csv'
forecast_summary.to_csv(forecast_file, index=False)
print(f"✅ Forecast saved to '{forecast_file}'")

# Save model comparison
comparison_file = 'model_comparison.csv'
comparison_df.to_csv(comparison_file)
print(f"✅ Model comparison saved to '{comparison_file}'")

# Save daily sales data
daily_sales_file = 'daily_sales_data.csv'
daily_sales_ts.to_csv(daily_sales_file)
print(f"✅ Daily sales data saved to '{daily_sales_file}'")

# Save recommendations
with open('business_recommendations.txt', 'w') as f:
    f.write("SALES FORECASTING - BUSINESS RECOMMENDATIONS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Analysis Period: {daily_sales_ts.index[0].date()} to {daily_sales_ts.index[-1].date()}\n")
    f.write(f"Best Model: {best_model}\n")
    f.write(f"Forecast Period: {forecast_days} days\n\n")
    
    f.write("KEY METRICS:\n")
    f.write(f"• Average Daily Sales: ${daily_sales_ts.mean():,.2f}\n")
    f.write(f"• Total Revenue: ${daily_sales_ts.sum():,.2f}\n")
    f.write(f"• Best Day: {best_day}\n")
    f.write(f"• Best Month: {best_month}\n")
    f.write(f"• Peak Hour: {peak_hour}:00\n\n")
    
    f.write("FORECAST SUMMARY:\n")
    f.write(f"• Next 30 Days Revenue: ${sum(forecast_values):,.2f}\n")
    f.write(f"• Average Daily Forecast: ${np.mean(forecast_values):,.2f}\n")
    f.write(f"• Expected Growth: {growth_target:.1f}%\n\n")
    
    f.write("RECOMMENDATIONS:\n")
    f.write("1. Increase inventory before peak seasons\n")
    f.write("2. Schedule marketing campaigns on high-sales days\n")
    f.write("3. Optimize staff scheduling for peak hours\n")
    f.write("4. Monitor at-risk periods for promotions\n")

print(f"✅ Recommendations saved to 'business_recommendations.txt'")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("🎉 SALES ANALYSIS & FORECASTING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📁 Files generated:")
print("   1. sales_eda_dashboard.png - Sales overview visualizations")
print("   2. sales_time_series.png - Time series at different frequencies")
print("   3. time_series_decomposition.png - Trend/seasonality analysis")
print("   4. model_predictions_comparison.png - Model performance comparison")
print("   5. future_sales_forecast.png - 30-day sales forecast")
print("   6. sales_forecast_results.csv - Detailed forecast data")
print("   7. model_comparison.csv - Model performance metrics")
print("   8. daily_sales_data.csv - Historical daily sales")
print("   9. business_recommendations.txt - Actionable insights")

print("\n" + "="*60)