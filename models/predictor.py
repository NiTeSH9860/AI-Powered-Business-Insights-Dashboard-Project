import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import xgboost as xgb
from datetime import datetime, timedelta
import pickle
import os

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_sales_data(self, sales_df):
        """Preparing sales data for forecasting"""
        # Aggregating daily sales
        daily_sales = sales_df.groupby('date')['sales_amount'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        
        # Adding features
        daily_sales['day_of_week'] = daily_sales['ds'].dt.dayofweek
        daily_sales['month'] = daily_sales['ds'].dt.month
        daily_sales['quarter'] = daily_sales['ds'].dt.quarter
        daily_sales['day_of_month'] = daily_sales['ds'].dt.day
        
        return daily_sales
    
    def forecast_with_prophet(self, sales_df, periods=30):
        """Using Prophet for time series forecasting"""
        try:
            # Preparing data
            df = sales_df[['date', 'sales_amount']].copy()
            df.columns = ['ds', 'y']
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Initializing and training Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            # Adding custom seasonalities
            model.add_country_holidays(country_name='US')
            
            model.fit(df)
            
            # Making future predictions
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Extracting relevant predictions
            predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
            return {
                'forecast_dates': predictions['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'forecast_values': predictions['yhat'].tolist(),
                'lower_bound': predictions['yhat_lower'].tolist(),
                'upper_bound': predictions['yhat_upper'].tolist(),
                'model': model
            }
        except Exception as e:
            print(f"Prophet forecasting error: {e}")
            return None
    
    def forecast_with_xgboost(self, sales_df, periods=30):
        """Using XGBoost for regression-based forecasting"""
        try:
            # Creating features
            df = sales_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Feature engineering
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df['quarter'] = df['date'].dt.quarter
            df['weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Creating lag features
            for lag in [1, 7, 14, 30]:
                df[f'lag_{lag}'] = df['sales_amount'].shift(lag)
            
            # Rolling statistics
            df['rolling_mean_7'] = df['sales_amount'].rolling(window=7).mean()
            df['rolling_std_7'] = df['sales_amount'].rolling(window=7).std()
            
            # Dropping NaN values
            df = df.dropna()
            
            # Preparing features
            feature_columns = ['day_of_week', 'day_of_month', 'month', 'quarter', 'weekend',
                             'lag_1', 'lag_7', 'rolling_mean_7']
            
            X = df[feature_columns]
            y = df['sales_amount']
            
            # Training model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X, y)
            
            # Creating future dates for prediction
            last_date = df['date'].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, periods+1)]
            
            # Preparing future features (simplified)
            future_df = pd.DataFrame({'date': future_dates})
            future_df['day_of_week'] = future_df['date'].dt.dayofweek
            future_df['day_of_month'] = future_df['date'].dt.day
            future_df['month'] = future_df['date'].dt.month
            future_df['quarter'] = future_df['date'].dt.quarter
            future_df['weekend'] = (future_df['day_of_week'] >= 5).astype(int)
            
            # Using last known values for lag features
            future_df['lag_1'] = df['sales_amount'].iloc[-1]
            future_df['lag_7'] = df['sales_amount'].iloc[-7] if len(df) > 7 else df['sales_amount'].iloc[-1]
            future_df['rolling_mean_7'] = df['sales_amount'].iloc[-7:].mean()
            
            # Making predictions
            predictions = model.predict(future_df[feature_columns])
            
            return {
                'forecast_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'forecast_values': predictions.tolist(),
                'feature_importance': dict(zip(feature_columns, model.feature_importances_)),
                'model': model
            }
        except Exception as e:
            print(f"XGBoost forecasting error: {e}")
            return None

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_churn_data(self, churn_df):
        """Prepare churn data for prediction"""
        features = ['monthly_spend', 'total_purchases', 'days_since_signup']
        
        # Calculating days since signup
        churn_df['days_since_signup'] = (pd.Timestamp.now() - pd.to_datetime(churn_df['signup_date'])).dt.days
        
        # Creating segment encoding
        segment_dummies = pd.get_dummies(churn_df['customer_segment'], prefix='segment')
        
        # Combining features
        X = churn_df[features].copy()
        for col in segment_dummies.columns:
            X[col] = segment_dummies[col]
            
        y = churn_df['churn_flag'] if 'churn_flag' in churn_df.columns else None
        
        return X, y
    
    def predict_churn_risk(self, churn_df):
        """Predict churn risk for customers"""
        try:
            X, _ = self.prepare_churn_data(churn_df)
            
            # Training Random Forest for churn prediction
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # If we have historical churn data, train on it
            if 'churn_flag' in churn_df.columns:
                y = churn_df['churn_flag']
                model.fit(X, y)
            else:
                # Otherwise use a rule-based approach
                churn_df['churn_risk'] = (
                    (churn_df['monthly_spend'] < churn_df['monthly_spend'].median()) * 0.3 +
                    (churn_df['total_purchases'] < churn_df['total_purchases'].median()) * 0.3 +
                    (churn_df['days_since_signup'] < 30) * 0.4
                )
                return churn_df[['customer_id', 'churn_risk']].to_dict('records')
            
            # Predicting probabilities
            churn_probabilities = model.predict_proba(X)[:, 1]
            
            # Getting feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            # Identifying high-risk customers
            results = []
            for idx, prob in enumerate(churn_probabilities):
                results.append({
                    'customer_id': churn_df.iloc[idx]['customer_id'],
                    'churn_risk': float(prob),
                    'segment': churn_df.iloc[idx]['customer_segment'],
                    'monthly_spend': float(churn_df.iloc[idx]['monthly_spend'])
                })
            
            # Sorting by risk (highest first)
            results.sort(key=lambda x: x['churn_risk'], reverse=True)
            
            return {
                'customer_risks': results[:20],  # Top 20 highest risk
                'feature_importance': feature_importance,
                'average_risk': float(np.mean(churn_probabilities)),
                'model': model
            }
            
        except Exception as e:
            print(f"Churn prediction error: {e}")
            return None