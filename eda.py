import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Read the data
file_path = 'data_YesBank_StockPrices.csv'
df = pd.read_csv(file_path)

# Convert 'Date' to datetime (assuming format 'Mon-YY')
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
df = df.sort_values('Date').reset_index(drop=True)

# Basic info
print('--- Data Head ---')
print(df.head())
print('\n--- Data Info ---')
print(df.info())
print('\n--- Data Description ---')
print(df.describe())

# Check for missing values
print('\n--- Missing Values ---')
print(df.isnull().sum())

# Handle missing values (if any)
df = df.dropna()

# Outlier detection and removal (Z-score method)
def remove_outliers_zscore(data, columns, threshold=3):
    z_scores = np.abs(stats.zscore(data[columns]))
    mask = (z_scores < threshold).all(axis=1)
    return data[mask]

num_cols = ['Open', 'High', 'Low', 'Close']
df = remove_outliers_zscore(df, num_cols)

# 1. Line plot of Closing Price over Time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], marker='o')
plt.title('Monthly Closing Price of Yes Bank Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. OHLC (Open, High, Low, Close) Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Open'], label='Open', alpha=0.7)
plt.plot(df['Date'], df['High'], label='High', alpha=0.7)
plt.plot(df['Date'], df['Low'], label='Low', alpha=0.7)
plt.plot(df['Date'], df['Close'], label='Close', alpha=0.7)
plt.title('OHLC Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Distribution of Closing Prices
plt.figure(figsize=(8, 5))
sns.histplot(df['Close'], bins=30, kde=True)
plt.title('Distribution of Monthly Closing Prices')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Monthly Returns (Percentage Change)
df['Monthly_Return'] = df['Close'].pct_change() * 100
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Monthly_Return'], marker='o', color='purple')
plt.title('Monthly Returns (%) Over Time')
plt.xlabel('Date')
plt.ylabel('Monthly Return (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Open', 'High', 'Low', 'Close']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Price Features')
plt.tight_layout()
plt.show()

# 6. Boxplot of Closing Price by Year
df['Year'] = df['Date'].dt.year
plt.figure(figsize=(12, 6))
sns.boxplot(x='Year', y='Close', data=df)
plt.title('Yearly Distribution of Closing Prices')
plt.xlabel('Year')
plt.ylabel('Closing Price')
plt.tight_layout()
plt.show()



