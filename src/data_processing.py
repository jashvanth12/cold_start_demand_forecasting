import pandas as pd
import numpy as np
import streamlit as st
import os

@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        data_dir = r"D:\colddd\data"
        panel = pd.read_csv(os.path.join(data_dir, "panel_train.csv"))
        price_plan_fut = pd.read_csv(os.path.join(data_dir, "price_plan_future.csv"))
        promos_fut = pd.read_csv(os.path.join(data_dir, "promos_future.csv"))
        weather_fut = pd.read_csv(os.path.join(data_dir, "weather_future.csv"))
        calendar_fut = pd.read_csv(os.path.join(data_dir, "calendar_future.csv"))
        return panel, price_plan_fut, promos_fut, weather_fut, calendar_fut
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

@st.cache_data
def parse_dates(panel, price_plan_fut, promos_fut, weather_fut, calendar_fut):
    """Parse date columns"""
    date_format = '%d-%m-%Y'
    panel['week_start'] = pd.to_datetime(panel['week_start'], format=date_format)
    price_plan_fut['week_start'] = pd.to_datetime(price_plan_fut['week_start'], format=date_format)
    calendar_fut['week_start'] = pd.to_datetime(calendar_fut['week_start'], format=date_format)
    weather_fut['week_start'] = pd.to_datetime(weather_fut['week_start'], format=date_format)
    promos_fut['week_start'] = pd.to_datetime(promos_fut['week_start'], format=date_format)
    promos_fut['week_end'] = pd.to_datetime(promos_fut['week_end'], format=date_format)
    return panel, price_plan_fut, promos_fut, weather_fut, calendar_fut

def has_promo(row, promos):
    """Check if SKU-Market has promo in given week"""
    matching = promos[
        (promos['market'] == row['market']) & 
        (promos['sku_id'] == row['sku_id']) &
        (promos['week_start'] <= row['week_start']) & 
        (promos['week_end'] >= row['week_start'])
    ]
    return 1 if not matching.empty else 0

@st.cache_data
def prepare_future_data(calendar_fut, price_plan_fut, weather_fut, promos_fut):
    """Prepare future forecast dataframe"""
    weeks = calendar_fut['week_start'].unique()
    markets = price_plan_fut['market'].unique()
    skus = price_plan_fut['sku_id'].unique()
    
    future_df = pd.DataFrame(
        [(w, m, s) for w in weeks for m in markets for s in skus], 
        columns=['week_start', 'market', 'sku_id']
    )
    
    future_df = future_df.merge(
        price_plan_fut, 
        on=['market', 'sku_id', 'week_start'], 
        how='left'
    ).rename(columns={'planned_price': 'price'})
    
    future_df = future_df.merge(
        weather_fut, 
        on=['market', 'week_start'], 
        how='left'
    )
    
    future_df = future_df.merge(
        calendar_fut[['week_start', 'holiday_flag']], 
        on='week_start', 
        how='left'
    )
    
    future_df['promo_flag'] = future_df.apply(
        lambda row: has_promo(row, promos_fut), 
        axis=1
    )
    
    return future_df

def engineer_features(all_df):
    """Create time-based features"""
    if all_df['week_start'].isnull().any():
        st.error("Missing values detected in week_start column")
        return all_df
    
    unique_weeks = sorted(all_df['week_start'].unique())
    all_df['time_index'] = all_df['week_start'].map({w: i for i, w in enumerate(unique_weeks)}).astype(float)
    all_df['week_of_year'] = all_df['week_start'].dt.isocalendar().week.astype(float)
    all_df['sin_week'] = np.sin(2 * np.pi * all_df['week_of_year'].fillna(0) / 52).astype(float)
    all_df['cos_week'] = np.cos(2 * np.pi * all_df['week_of_year'].fillna(0) / 52).astype(float)
    all_df['month'] = all_df['week_start'].dt.month.astype(float)
    all_df['quarter'] = all_df['week_start'].dt.quarter.astype(float)
    return all_df