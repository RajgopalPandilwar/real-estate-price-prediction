# -*- coding: utf-8 -*-
"""
Real Estate Price Prediction
============================
A machine learning project for predicting real estate prices based on property features.
Uses regression models with feature engineering and cross-validation.

Author: Rajgopal Pandilwar
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_and_explore_data(file_path='house_data.csv'):
    """
    Load real estate data and perform initial exploration.
    
    Args:
        file_path (str): Path to the dataset CSV
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print("Loading real estate data...")
    df = pd.read_csv(file_path)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Names and Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nPrice Statistics:")
    print(df['price'].describe() if 'price' in df.columns else "No 'price' column found")
    
    return df


def feature_engineering(df):
    """
    Create new features and transform existing ones for better model performance.
    
    Args:
        df (pandas.DataFrame): Raw dataset
    
    Returns:
        pandas.DataFrame: Dataset with engineered features
    """
    df_fe = df.copy()
    
    # Create price per square foot feature
    if 'sqft_living' in df_fe.columns and 'price' in df_fe.columns:
        df_fe['price_per_sqft'] = df_fe['price'] / df_fe['sqft_living']
    
    # Create total square footage feature
    if 'sqft_living' in df_fe.columns and 'sqft_lot' in df_fe.columns:
        df_fe['total_sqft'] = df_fe['sqft_living'] + df_fe['sqft_lot']
    
    # Create age of property feature
    if 'yr_built' in df_fe.columns:
        df_fe['property_age'] = 2024 - df_fe['yr_built']
    
    # Create renovation feature
    if 'yr_renovated' in df_fe.columns:
        df_fe['is_renovated'] = (df_fe['yr_renovated'] > 0).astype(int)
    
    # Encode categorical variables
    categorical_cols = ['zipcode', 'condition', 'grade', 'view', 'waterfront']
    le = LabelEncoder()
    
    for col in categorical_cols:
        if col in df_fe.columns:
            df_fe[col] = le.fit_transform(df_fe[col].astype(str))
    
    print("Feature engineering complete.")
    print(f"New features: {list(df_fe.columns)}")
    
    return df_fe


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple regression models and evaluate performance.
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing target
    
    Returns:
        dict: Model performance results
    """
    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': (LinearRegression(), X_train_scaled, X_test_scaled),
        'Ridge Regression': (Ridge(alpha=1.0), X_train_scaled, X_test_scaled),
        'Lasso Regression': (Lasso(alpha=1.0), X_train_scaled, X_test_scaled),
        'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), X_train, X_test),
    }
    
    results = {}
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    for name, (model, X_tr, X_te) in models.items():
        print(f"\n--- {name} ---")
        
        # Train the model
        model.fit(X_tr, y_train)
        
        # Make predictions
        y_pred = model.predict(X_te)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE:  {mae:.2f}")
        print(f"R2:   {r2:.4f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"{'Model':<25} {'RMSE':<15} {'MAE':<15} {'R2':<10}")
    print("-" * 65)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['RMSE']):
        print(f"{name:<25} {metrics['RMSE']:<15.2f} {metrics['MAE']:<15.2f} {metrics['R2']:<10.4f}")
    
    return results


def main():
    """Main execution function."""
    print("=" * 60)
    print("REAL ESTATE PRICE PREDICTION SYSTEM")
    print("Author: Rajgopal Pandilwar")
    print("=" * 60)
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    
    # Step 2: Feature engineering
    df_fe = feature_engineering(df)
    
    # Step 3: Split features and target
    target_col = 'price'
    if target_col not in df_fe.columns:
        # Try to find a price-like column
        price_cols = [c for c in df_fe.columns if 'price' in c.lower()]
        if price_cols:
            target_col = price_cols[0]
        else:
            raise ValueError("No target column found")
    
    X = df_fe.drop(target_col, axis=1)
    y = df_fe[target_col]
    
    # Step 4: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Step 5: Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 60)
    print("Prediction system ready.")
    print("=" * 60)


if __name__ == '__main__':
    main()
