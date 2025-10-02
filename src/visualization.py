import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

def plot_forecast_interactive(predictions, selected_skus, selected_market):
    """Interactive forecast plot with Plotly"""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    
    for idx, sku in enumerate(selected_skus):
        sku_data = predictions[
            (predictions['sku_id'] == sku) & 
            (predictions['market'] == selected_market)
        ].sort_values('week_start')
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=sku_data['week_start'],
            y=sku_data['forecast'],
            mode='lines+markers',
            name=f'{sku} Forecast',
            line=dict(color=color, width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=sku_data['week_start'].tolist() + sku_data['week_start'].tolist()[::-1],
            y=sku_data['upper_90'].tolist() + sku_data['lower_90'].tolist()[::-1],
            fill='toself',
            fillcolor=color,
            opacity=0.2,
            line=dict(width=0),
            showlegend=True,
            name=f'{sku} 90% CI',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f'13-Week Demand Forecast for {selected_market}',
        xaxis_title='Week Starting',
        yaxis_title='Forecasted Units',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    return fig

def plot_driver_attribution(model, features):
    """Plot feature importance"""
    importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=True).tail(15)
    
    fig = go.Figure(go.Bar(
        x=importance['importance'],
        y=importance['feature'],
        orientation='h',
        marker=dict(
            color=importance['importance'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Importance")
        )
    ))
    
    fig.update_layout(
        title='Top 15 Feature Importance (Driver Attribution)',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=500,
        template='plotly_white'
    )
    return fig

def plot_weather_impact(predictions, selected_market):
    """Weather impact heatmap"""
    market_data = predictions[predictions['market'] == selected_market].copy()
    market_data['temp_bin'] = pd.cut(market_data['temp_c'], bins=5)
    market_data['rain_bin'] = pd.cut(market_data['rain_mm'], bins=5)
    
    pivot_data = market_data.pivot_table(
        values='forecast',
        index='temp_bin',
        columns='rain_bin',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[str(col) for col in pivot_data.columns],
        y=[str(idx) for idx in pivot_data.index],
        colorscale='YlOrRd',
        colorbar=dict(title="Avg Forecast")
    ))
    
    fig.update_layout(
        title='Weather Impact on Demand (Temperature vs Rain)',
        xaxis_title='Rain (mm)',
        yaxis_title='Temperature (°C)',
        height=500,
        template='plotly_white'
    )
    return fig

def plot_shock_analysis(predictions, selected_market, selected_skus):
    """Analyze shock impact (promos, holidays)"""
    market_data = predictions[
        (predictions['market'] == selected_market) &
        (predictions['sku_id'].isin(selected_skus))
    ].copy()
    
    market_data['shock_type'] = 'Normal'
    market_data.loc[market_data['promo_flag'] == 1, 'shock_type'] = 'Promo'
    market_data.loc[market_data['holiday_flag'] == 1, 'shock_type'] = 'Holiday'
    market_data.loc[
        (market_data['promo_flag'] == 1) & (market_data['holiday_flag'] == 1),
        'shock_type'
    ] = 'Promo+Holiday'
    
    shock_summary = market_data.groupby('shock_type')['forecast'].agg(['mean', 'std', 'count'])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shock_summary.index,
        y=shock_summary['mean'],
        error_y=dict(type='data', array=shock_summary['std']),
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        text=shock_summary['count'],
        texttemplate='n=%{text}',
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Impact of Shocks on Forecasted Demand - {selected_market}',
        xaxis_title='Shock Type',
        yaxis_title='Average Forecasted Units',
        height=400,
        template='plotly_white'
    )
    return fig

def plot_uncertainty_width(predictions, selected_market):
    """Analyze uncertainty width across time"""
    market_data = predictions[predictions['market'] == selected_market].copy()
    
    weekly_uncertainty = market_data.groupby('week_start').agg({
        'forecast': 'mean',
        'upper_90': 'mean',
        'lower_90': 'mean'
    }).reset_index()
    
    weekly_uncertainty['uncertainty_width'] = (
        weekly_uncertainty['upper_90'] - weekly_uncertainty['lower_90']
    )
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=weekly_uncertainty['week_start'],
            y=weekly_uncertainty['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Bar(
            x=weekly_uncertainty['week_start'],
            y=weekly_uncertainty['uncertainty_width'],
            name='90% CI Width',
            marker_color='lightcoral',
            opacity=0.6
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f'Forecast Uncertainty Over Time - {selected_market}',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    fig.update_xaxes(title_text="Week Starting")
    fig.update_yaxes(title_text="Forecasted Units", secondary_y=False)
    fig.update_yaxes(title_text="Uncertainty Width", secondary_y=True)
    return fig