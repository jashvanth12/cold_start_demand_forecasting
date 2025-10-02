import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from datetime import datetime
from data_processing import load_data, parse_dates, prepare_future_data, engineer_features
from modeling import train_models, make_predictions
from visualization import (
    plot_forecast_interactive, plot_driver_attribution, 
    plot_weather_impact, plot_shock_analysis, plot_uncertainty_width
)

st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    h1 {color: #1f77b4; padding-bottom: 20px;}
    h2 {color: #2c3e50; padding-top: 20px;}
    .reportview-container .main .block-container {max-width: 1400px;}
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("📊 Shock-Aware Demand Forecasting Dashboard")
    st.markdown("### 13-Week Sales Forecast with Uncertainty Quantification")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        if st.button("🔄 Load & Process Data"):
            st.session_state.data_loaded = True
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This dashboard provides:
        - 13-week demand forecasts
        - 90% prediction intervals
        - Driver attribution
        - Shock impact analysis
        - Weather sensitivity
        """)
    
    if 'data_loaded' not in st.session_state:
        st.info("👈 Click 'Load & Process Data' in the sidebar to begin")
        with st.expander("📋 Expected Data Structure"):
            st.markdown("""
            **Required CSV files in D:\\colddd\\data\\:**
            - panel_train.csv: Historical sales data
            - price_plan_future.csv: Future pricing plans
            - promos_future.csv: Planned promotions
            - weather_future.csv: Weather forecasts
            - calendar_future.csv: Calendar with holidays
            """)
        return
    
    with st.spinner("Loading data..."):
        panel, price_plan_fut, promos_fut, weather_fut, calendar_fut = load_data()
        if panel is None:
            st.error("Failed to load data. Please ensure CSV files are in D:\\colddd\\data\\.")
            return
        panel, price_plan_fut, promos_fut, weather_fut, calendar_fut = parse_dates(
            panel, price_plan_fut, promos_fut, weather_fut, calendar_fut
        )
    
    st.header("📈 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Markets", panel['market'].nunique())
    with col2:
        st.metric("SKUs", panel['sku_id'].nunique())
    with col3:
        st.metric("Historical Weeks", panel['week_start'].nunique())
    with col4:
        st.metric("Forecast Weeks", calendar_fut['week_start'].nunique())
    
    with st.spinner("Preparing features..."):
        future_df = prepare_future_data(calendar_fut, price_plan_fut, weather_fut, promos_fut)
        all_df = pd.concat([panel, future_df], ignore_index=True).sort_values('week_start')
        all_df = engineer_features(all_df)
        train_df = all_df[all_df['units'].notnull()].copy()
        future_df = all_df[all_df['units'].isnull()].copy()
    
    features = ['market', 'sku_id', 'time_index', 'sin_week', 'cos_week', 
                'price', 'promo_flag', 'holiday_flag', 'temp_c', 'rain_mm']
    categorical = ['market', 'sku_id', 'promo_flag', 'holiday_flag']
    
    with st.spinner("Training models (mean + quantile regression)..."):
        model_mean, model_lower, model_upper = train_models(train_df, features, categorical)
        if model_mean is None:
            st.error("Model training failed. Check data types and try again.")
            return
    
    st.success("✅ Models trained successfully!")
    
    with st.spinner("Generating forecasts..."):
        predictions = make_predictions(model_mean, model_lower, model_upper, future_df, features, categorical)
        if predictions is None:
            st.error("Prediction failed. Check data types and try again.")
            return
    
    st.header("🎯 Forecast Explorer")
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_market = st.selectbox("Select Market", options=sorted(predictions['market'].unique()))
    with col2:
        available_skus = sorted(predictions[predictions['market'] == selected_market]['sku_id'].unique())
        selected_skus = st.multiselect("Select SKUs", options=available_skus, 
                                     default=available_skus[:3] if len(available_skus) >= 3 else available_skus)
    
    if not selected_skus:
        st.warning("Please select at least one SKU")
        return
    
    st.subheader(f"📊 Forecast with 90% Prediction Intervals")
    fig_forecast = plot_forecast_interactive(predictions, selected_skus, selected_market)
    st.plotly_chart(fig_forecast, use_container_width=True)
    os.makedirs(r"D:\colddd\plots", exist_ok=True)
    fig_forecast.write_html(r"D:\colddd\plots\forecast_plot.html")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Driver Attribution")
        fig_importance = plot_driver_attribution(model_mean, features)
        st.plotly_chart(fig_importance, use_container_width=True)
        fig_importance.write_html(r"D:\colddd\plots\driver_attribution.html")
    
    with col2:
        st.subheader("⚡ Shock Impact Analysis")
        fig_shock = plot_shock_analysis(predictions, selected_market, selected_skus)
        st.plotly_chart(fig_shock, use_container_width=True)
        fig_shock.write_html(r"D:\colddd\plots\shock_analysis.html")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌤️ Weather Impact")
        fig_weather = plot_weather_impact(predictions, selected_market)
        st.plotly_chart(fig_weather, use_container_width=True)
        fig_weather.write_html(r"D:\colddd\plots\weather_impact.html")
    
    with col2:
        st.subheader("📏 Uncertainty Analysis")
        fig_uncertainty = plot_uncertainty_width(predictions, selected_market)
        st.plotly_chart(fig_uncertainty, use_container_width=True)
        fig_uncertainty.write_html(r"D:\colddd\plots\uncertainty_analysis.html")
    
    st.header("📋 Detailed Forecast Table")
    forecast_table = predictions[
        (predictions['market'] == selected_market) &
        (predictions['sku_id'].isin(selected_skus))
    ][['week_start', 'sku_id', 'forecast', 'lower_90', 'upper_90', 
       'price', 'promo_flag', 'holiday_flag', 'temp_c', 'rain_mm']].copy()
    
    forecast_table['week_start'] = forecast_table['week_start'].dt.strftime('%Y-%m-%d')
    forecast_table = forecast_table.round(2)
    forecast_table = forecast_table.sort_values(['sku_id', 'week_start'])
    os.makedirs(r"D:\colddd\data", exist_ok=True)
    forecast_table.to_csv(r"D:\colddd\data\forecast.csv", index=False)
    
    st.dataframe(forecast_table, use_container_width=True, height=400)
    
    csv = forecast_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f"forecast_{selected_market}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    st.header("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    market_forecast = predictions[predictions['market'] == selected_market]
    
    with col1:
        st.metric("Total Forecasted Units", f"{market_forecast['forecast'].sum():,.0f}")
    with col2:
        st.metric("Avg Weekly Demand", f"{market_forecast['forecast'].mean():,.0f}")
    with col3:
        avg_uncertainty = (market_forecast['upper_90'] - market_forecast['lower_90']).mean()
        st.metric("Avg Uncertainty Width", f"{avg_uncertainty:,.0f}")
    with col4:
        promo_weeks = (market_forecast['promo_flag'] == 1).sum()
        st.metric("Promo Weeks", f"{promo_weeks}")

if __name__ == "__main__":
    main()