import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import joblib

# 1. Load Data
file_path = 'data_YesBank_StockPrices.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
df = df.sort_values('Date').reset_index(drop=True)

# 2. Handle Missing Values
print('--- Missing Values Before ---')
print(df.isnull().sum())
# Drop any rows with missing values
if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print('Missing values found and dropped.')
else:
    print('No missing values found.')
print('--- Missing Values After ---')
print(df.isnull().sum())

# 3. Outlier Detection and Handling (Z-score method for numeric columns)
def remove_outliers_zscore(data, columns, threshold=3):
    z_scores = np.abs(stats.zscore(data[columns]))
    mask = (z_scores < threshold).all(axis=1)
    return data[mask]

num_cols = ['Open', 'High', 'Low', 'Close']
df_clean = remove_outliers_zscore(df, num_cols)
print(f"Rows before outlier removal: {len(df)}")
print(f"Rows after outlier removal: {len(df_clean)}")

# 4. Correlation Analysis
plt.figure(figsize=(8, 6))
sns.heatmap(df_clean[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Price Features')
plt.tight_layout()
plt.show()

# 5. Feature Engineering (add lag features, month, year)
df_clean['Month'] = df_clean['Date'].dt.month
df_clean['Year'] = df_clean['Date'].dt.year
df_clean['Prev_Close'] = df_clean['Close'].shift(1)
df_clean = df_clean.dropna().reset_index(drop=True)

# 6. Feature Selection
features = ['Open', 'High', 'Low', 'Month', 'Year', 'Prev_Close']
target = 'Close'

# 7. Train-Test Split
X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 8. Model Training & Evaluation Functions
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    """Train and evaluate a regression model, print metrics and plot results."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f'\n--- {name} ---')
    print('Train RMSE:', np.sqrt(mean_squared_error(y_train, y_pred_train)))
    print('Test RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print('Test MAE:', mean_absolute_error(y_test, y_pred_test))
    print('Test R2:', r2_score(y_test, y_pred_test))
    plt.figure(figsize=(8, 4))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred_test, label='Predicted')
    plt.title(f'{name} - Actual vs Predicted Closing Price')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return np.sqrt(mean_squared_error(y_test, y_pred_test)), r2_score(y_test, y_pred_test)

# 9. Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_rmse, lr_r2 = evaluate_model(lr, X_train, y_train, X_test, y_test, 'Linear Regression')

# 10. Model 2: Random Forest Regressor (with hyperparameter tuning)
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, None]}
rf = RandomForestRegressor(random_state=42)
gs_rf = GridSearchCV(rf, rf_params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
gs_rf.fit(X_train, y_train)
print('Best RF Params:', gs_rf.best_params_)
rf_best = gs_rf.best_estimator_
rf_rmse, rf_r2 = evaluate_model(rf_best, X_train, y_train, X_test, y_test, 'Random Forest Regressor')

# 11. Model 3: Ridge Regression (with hyperparameter tuning)
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
gs_ridge = GridSearchCV(ridge, ridge_params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
gs_ridge.fit(X_train, y_train)
print('Best Ridge Params:', gs_ridge.best_params_)
ridge_best = gs_ridge.best_estimator_
ridge_rmse, ridge_r2 = evaluate_model(ridge_best, X_train, y_train, X_test, y_test, 'Ridge Regression')

# 12. Final Model Selection
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Ridge Regression'],
    'Test RMSE': [lr_rmse, rf_rmse, ridge_rmse],
    'Test R2': [lr_r2, rf_r2, ridge_r2]
})
print('\n--- Model Comparison ---')
print(results)

best_model_idx = results['Test RMSE'].idxmin()
print(f"\nBest Model: {results.loc[best_model_idx, 'Model']} (Lowest Test RMSE)")

# 13. Final Summary
print("""
Summary:
- Data was cleaned for missing values and outliers.
- Feature engineering included lag features and date parts.
- Three models were tested: Linear Regression, Random Forest, and Ridge Regression (with hyperparameter tuning).
- The best model was selected based on lowest Test RMSE.
- All code is modular, commented, and outputs are clearly formatted for interpretation.
""")

def save_best_model(model, filename='best_model.pkl'):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f'Best model saved to {filename}')

# Save the best model
if results.loc[best_model_idx, 'Model'] == 'Random Forest':
    save_best_model(rf_best)
elif results.loc[best_model_idx, 'Model'] == 'Ridge Regression':
    save_best_model(ridge_best)
else:
    save_best_model(lr)

# --- Prediction Utility ---
def predict_new_closing_price(open_price, high_price, low_price, month, year, prev_close, model_path='best_model.pkl'):
    """
    Predict the closing price for a new month using the saved best model.
    Args:
        open_price (float): Opening price for the month
        high_price (float): Highest price for the month
        low_price (float): Lowest price for the month
        month (int): Month as integer (1-12)
        year (int): Year as integer (e.g., 2021)
        prev_close (float): Previous month's closing price
        model_path (str): Path to the saved model file
    Returns:
        float: Predicted closing price
    """
    model = joblib.load(model_path)
    new_data = pd.DataFrame([{
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Month': month,
        'Year': year,
        'Prev_Close': prev_close
    }])
    pred = model.predict(new_data)
    return pred[0]

# --- Optional: Command-line interface ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict Yes Bank monthly closing price.')
    parser.add_argument('--open', type=float, required=False, help='Opening price for the month')
    parser.add_argument('--high', type=float, required=False, help='Highest price for the month')
    parser.add_argument('--low', type=float, required=False, help='Lowest price for the month')
    parser.add_argument('--month', type=int, required=False, help='Month as integer (1-12)')
    parser.add_argument('--year', type=int, required=False, help='Year as integer (e.g., 2021)')
    parser.add_argument('--prev_close', type=float, required=False, help="Previous month's closing price")
    args = parser.parse_args()
    if all([args.open, args.high, args.low, args.month, args.year, args.prev_close]):
        pred = predict_new_closing_price(
            open_price=args.open,
            high_price=args.high,
            low_price=args.low,
            month=args.month,
            year=args.year,
            prev_close=args.prev_close
        )
        print(f'Predicted Closing Price: {pred:.2f}')
    else:
        print('To predict, provide all arguments: --open --high --low --month --year --prev_close')
        print('Or use the function predict_new_closing_price() in your own script.') 