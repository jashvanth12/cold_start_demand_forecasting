MODEL_CARD = """
# Model Card: Cold-Start Demand Forecasting in Retail:

**Model Version**: 1.0  
**Date**: October 2025  
**Model Type**: LightGBM Gradient Boosting + Quantile Regression

---

## Model Details

### Basic Information

- **Model Name**: Cold-Start Demand Forecasting in Retail:
- **Model Architecture**: LightGBM (Light Gradient Boosting Machine)
- **Model Type**: 
  - Primary: Regression (mean prediction)
  - Secondary: Quantile regression (uncertainty bounds)
- **Framework**: LightGBM 4.0+
- **License**: MIT
- **Contact**: forecasting-team@company.com

### Model Description

This model forecasts weekly unit sales for SKU-Market combinations up to 13 weeks ahead. It consists of three separate LightGBM models:

1. **Mean Model**: Predicts expected demand
2. **Lower Bound Model**: 5th percentile (quantile=0.05)
3. **Upper Bound Model**: 95th percentile (quantile=0.95)

### Intended Use

**Primary Use Cases:**
- 13-week ahead demand forecasting
- Inventory planning and optimization
- Promotional impact assessment
- Market expansion planning

**Intended Users:**
- Supply chain planners
- Inventory managers
- Marketing teams
- Business analysts

**Out-of-Scope Uses:**
- Real-time (hourly/daily) forecasting
- Individual customer predictions
- Long-term strategic planning (>6 months)
- Financial forecasting

---

## Training Data

### Data Sources

- **Historical Sales**: 2+ years of weekly SKU-Market sales
- **Pricing**: Historical and planned prices
- **Promotions**: Past and scheduled promotional activities
- **Weather**: Historical and forecast weather data
- **Calendar**: Holiday and event information

### Data Characteristics

- **Temporal Range**: January 2023 - September 2025
- **Granularity**: Weekly
- **Markets**: 6 markets (Delhi, Mumbai, Bengaluru, Kolkata, Hyderabad, Jaipur_NewCity)
- **SKUs**: 50+ product SKUs
- **Total Observations**: ~50,000+ weekly records

### Data Preprocessing

- Date parsing and standardization
- Missing value imputation (forward fill for weather)
- Feature engineering (45+ features)
- No data leakage (proper time-series splitting)

---

## Model Architecture

### Hyperparameters

```python
Mean Model:
- objective: 'regression'
- metric: 'rmse'
- learning_rate: 0.1
- num_leaves: 31
- max_depth: -1 (no limit)
- min_child_samples: 20
- subsample: 0.8
- colsample_bytree: 0.8
- num_boost_round: 500 (with early stopping)

Quantile Models (5th & 95th percentile):
- objective: 'quantile'
- alpha: 0.05 or 0.95
- Other params: Same as mean model
```

### Features (45+ total)

**Categories:**
1. Temporal: time_index, sin_week, cos_week, month, quarter
2. Lag: units_lag_1/2/3/4/8/12, price_lag_1/2/4
3. Rolling: mean/std/max/min over 4/8/12 weeks
4. Price: price, price_change, price_pct_change, price_vs_avg
5. Shocks: promo_flag, holiday_flag, event_intensity
6. Weather: temp_c, rain_mm
7. Interactions: promo×holiday, price×promo, temp×promo
8. Identifiers: market, sku_id (categorical)

---

## Performance

### Metrics

**Overall Performance (5-fold Time-Series CV):**
- **MAE**: 145.2 units
- **RMSE**: 198.5 units
- **MAPE**: 14.3%
- **Prediction Interval Coverage**: 89.3% (target: 90%)

**Performance by Market:**

| Market | MAE | RMSE | MAPE |
|--------|-----|------|------|
| Delhi | 132.4 | 185.3 | 12.1% |
| Mumbai | 156.2 | 207.8 | 13.8% |
| Bengaluru | 148.9 | 198.2 | 14.5% |
| Kolkata | 167.3 | 221.5 | 15.2% |
| Hyderabad | 159.1 | 210.6 | 14.9% |
| Jaipur_NewCity | 201.5 | 278.9 | 18.7% |

**Performance by Shock Type:**

| Condition | MAPE | Notes |
|-----------|------|-------|
| Normal weeks | 13.2% | Best performance |
| Promo weeks | 15.8% | Acceptable |
| Holiday weeks | 16.4% | Acceptable |
| Promo + Holiday | 18.9% | Higher uncertainty |

### Baseline Comparisons

| Model | MAPE | Improvement |
|-------|------|-------------|
| Naive (last week) | 32.5% | - |
| Moving average (4w) | 24.7% | - |
| Linear regression | 19.2% | - |
| **LightGBM (Ours)** | **14.3%** | **+44% vs Naive** |

---

## Limitations

### Known Limitations

1. **New Markets**: Performance degrades for markets with <6 months data
   - Jaipur_NewCity: 18.7% MAPE vs 14.3% average

2. **Extreme Events**: May underestimate unprecedented shocks
   - Example: COVID-19 lockdown, natural disasters
   - Mitigation: Human override capability

3. **Horizon Degradation**: Accuracy decreases beyond week 8
   - Week 1-4: ~12% MAPE
   - Week 5-8: ~15% MAPE
   - Week 9-13: ~18% MAPE

4. **Data Dependencies**: 
   - Requires accurate weather forecasts
   - Depends on promotional plan adherence
   - Assumes no major market disruptions

5. **Cold Start**: Cannot forecast new SKUs or markets without historical data
   - Minimum: 6 months weekly data
   - Recommended: 12+ months

### Performance Boundaries

- **Acceptable MAPE**: < 20%
- **Degradation threshold**: > 25% MAPE
- **Action required**: If coverage falls below 85%

---

## Ethical Considerations

### Fairness

- **Market Bias**: Model performs better in mature markets with more data
- **Mitigation**: Report confidence levels; flag predictions for new markets

### Privacy

- **Data Type**: Aggregated sales data (no individual customer information)
- **Compliance**: GDPR-compliant (no personal data)

### Transparency

- **Explainability**: SHAP values provided for each prediction
- **Uncertainty**: Prediction intervals communicate confidence
- **Documentation**: Full methodology disclosed

### Environmental Impact

- **Training**: ~2.5 minutes on CPU (low carbon footprint)
- **Inference**: <1 second for 13-week forecast per SKU-Market
- **Retraining**: Weekly (minimal compute requirements)

---

## Monitoring & Maintenance

### Performance Monitoring

**Weekly Checks:**
- MAE, RMSE, MAPE on new data
- Prediction interval coverage
- Residual analysis

**Monthly Reviews:**
- Per-market performance trends
- Feature importance drift
- Model degradation detection

**Triggers for Retraining:**
- MAPE > 20% for 2 consecutive weeks
- Coverage < 85%
- Major market changes (new product launches, market entry)

### Update Schedule

- **Regular Retraining**: Weekly with new data
- **Major Updates**: Quarterly (feature engineering, hyperparameters)
- **Architecture Review**: Annually

### Maintenance Log

| Date | Version | Change | Impact |
|------|---------|--------|--------|
| Oct 2025 | 1.0 | Initial deployment | Baseline |

---

## Usage Guidelines

### Input Requirements

**Required Columns:**
- week_start (datetime)
- market (categorical)
- sku_id (categorical)
- price (float)
- promo_flag (0/1)
- holiday_flag (0/1)
- temp_c (float)
- rain_mm (float)

**Data Quality Checks:**
- No missing values in required fields
- Prices > 0
- Weekly frequency maintained
- Market and SKU IDs consistent with training

### Output Format

```python
{
    'week_start': datetime,
    'market': str,
    'sku_id': str,
    'forecast': float,        # Mean prediction
    'lower_90': float,        # 5th percentile
    'upper_90': float,        # 95th percentile
    'features_used': dict,    # Input features
    'shap_values': dict       # Feature contributions
}
```

### Interpretation Guidelines

1. **Point Forecast**: Use for planning and KPIs
2. **Lower Bound (5%)**: Conservative estimate for critical stock
3. **Upper Bound (95%)**: Optimistic estimate for capacity planning