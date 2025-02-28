# NFHS-4 Data Analysis Dashboard

## Overview
This repository contains a Streamlit dashboard that analyzes data from the National Family Health Survey (NFHS-4) conducted in India. The dashboard explores relationships between education, fertility patterns, and demographic characteristics across urban and rural areas.

## Features
- **Interactive Dashboard**: Multi-tab interface with key statistics and visualizations
- **Education Analysis**: Visualizations of education levels and distribution by residence
- **Fertility Patterns**: Analysis of the relationship between education and number of children
- **Demographics**: Age distribution and BMI analysis (where available)
- **Data Explorer**: Interactive tool to filter data and create custom visualizations

## Data
The dashboard uses data from the National Family Health Survey (NFHS-4), which includes information on:
- Type of residence (urban/rural)
- Education levels (in years)
- Number of children
- Age of respondents
- BMI (Body Mass Index) where available

## Key Insights
- Significant education gap between urban and rural areas
- Strong negative correlation between education and number of children
- Rural women have higher fertility rates compared to urban women
- Education levels are a strong predictor of fertility patterns

## Technical Details
The dashboard is built using:
- Streamlit for the web application framework
- Pandas for data manipulation
- Plotly for interactive visualizations
- Matplotlib and Seaborn for statistical visualizations

## Deployment Instructions

### Local Deployment
1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run nfhs_dashboard.py
   ```

### Streamlit Cloud Deployment
1. Fork this repository to your GitHub account
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and select the forked repository
4. Set the main file path to `nfhs_dashboard.py`
5. Deploy!

## File Structure
- `nfhs_dashboard.py`: Main Streamlit application
- `NFHS 4 Data.csv`: Dataset containing NFHS-4 survey data
- `requirements.txt`: Required Python packages
- `README.md`: This documentation

## License
This project is available for educational and research purposes.
