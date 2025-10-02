```markdown
# Cold-Start Demand Forecasting in Retail:

## Overview
This project is an interactive Streamlit dashboard for 13-week demand forecasting with uncertainty quantification, incorporating external factors like promotions, holidays, and weather.

## Prerequisites
- Python 3.8+
- Conda (optional, for environment.yml)
- Git

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd shock_aware_forecasting
   ```

2. **Create Environment**
   Using conda:
   ```bash
   conda env create -f environment.yml
   conda activate forecasting_env
   ```
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**
   Place the following CSV files in the `data/` directory:
   - `panel_train.csv`: Historical sales data
   - `price_plan_future.csv`: Future pricing plans
   - `promos_future.csv`: Planned promotions
   - `weather_future.csv`: Weather forecasts
   - `calendar_future.csv`: Calendar with holidays

4. **Run the Application**
   ```bash
   streamlit run src/app.py
   ```

5. **Access the Dashboard**
   Open your browser to `http://localhost:8501`.

## Project Structure
- `src/`: Python modules for data processing, modeling, visualization, and app logic
- `reports/`: Project report, model card, and explainability report
- `data/`: Output forecast CSV
- `plots/`: Exported Plotly visualizations
- `environment.yml`: Conda environment file
- `requirements.txt`: Pip requirements file

## Deliverables
- Reproducible environment (`environment.yml`, `requirements.txt`)
- Project report (`reports/project_report.md`)
- Model card (`reports/model_card.md`)
- Explainability report (`reports/explainability_report.md`)
- Forecast output (`data/forecast.csv`)
- Visualization plots (`plots/`)
```