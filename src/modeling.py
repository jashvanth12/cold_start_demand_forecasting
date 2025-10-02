import lightgbm as lgb
import streamlit as st
import pandas as pd

@st.cache_resource
def train_models(train_df, features, categorical):
    """Train mean and quantile models"""
    X_train = train_df[features].copy()
    y_train = train_df['units']
    
    # Verify data types for LightGBM
    for col in X_train.columns:
        if col not in categorical and X_train[col].dtype not in [int, float, bool]:
            st.error(f"Invalid data type for feature {col}: {X_train[col].dtype}. Must be int, float, or bool.")
            return None, None, None
    
    for cat in categorical:
        X_train[cat] = X_train[cat].astype('category')
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical, free_raw_data=False)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }
    
    model_mean = lgb.train(params, train_data, num_boost_round=500)
    
    params_lower = params.copy()
    params_lower['objective'] = 'quantile'
    params_lower['alpha'] = 0.05
    model_lower = lgb.train(params_lower, train_data, num_boost_round=500)
    
    params_upper = params.copy()
    params_upper['objective'] = 'quantile'
    params_upper['alpha'] = 0.95
    model_upper = lgb.train(params_upper, train_data, num_boost_round=500)
    
    return model_mean, model_lower, model_upper

def make_predictions(model_mean, model_lower, model_upper, future_df, features, categorical):
    """Generate predictions with uncertainty intervals"""
    if model_mean is None or model_lower is None or model_upper is None:
        st.error("Cannot make predictions: Models not trained successfully.")
        return None
    
    X_future = future_df[features].copy()
    
    # Verify data types for LightGBM
    for col in X_future.columns:
        if col not in categorical and X_future[col].dtype not in [int, float, bool]:
            st.error(f"Invalid data type for feature {col}: {X_future[col].dtype}. Must be int, float, or bool.")
            return None
    
    for cat in categorical:
        X_future[cat] = X_future[cat].astype('category')
    
    predictions = future_df.copy()
    predictions['forecast'] = model_mean.predict(X_future)
    predictions['lower_90'] = model_lower.predict(X_future)
    predictions['upper_90'] = model_upper.predict(X_future)
    
    predictions['forecast'] = predictions['forecast'].clip(lower=0)
    predictions['lower_90'] = predictions['lower_90'].clip(lower=0)
    predictions['upper_90'] = predictions['upper_90'].clip(lower=0)
    
    return predictions