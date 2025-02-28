import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="NFHS-4 Data Analyzer", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f8ff;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.1);
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1E88E5;
}
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("NFHS 4 Data.csv")
        df.columns = ['residence', 'education_years', 'children', 'age', 'bmi']
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Descriptive statistics function
def get_descriptive_stats(df, column):
    return {
        'Mean': df[column].mean(),
        'Median': df[column].median(),
        'Standard Deviation': df[column].std(),
        'Minimum': df[column].min(),
        'Maximum': df[column].max()
    }

# Regression analysis
def perform_regression(X, y):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return {
        'Coefficients': model.coef_,
        'Intercept': model.intercept_,
        'R-squared': model.score(X_scaled, y)
    }

# Main app
def main():
    st.markdown('<div class="main-header">NFHS-4 Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar for navigation
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Descriptive Statistics", 
            "Distribution Analysis", 
            "Regression Analysis"
        ]
    )
    
    # Descriptive Statistics
    if analysis_type == "Descriptive Statistics":
        st.header("Descriptive Statistics")
        
        # Select columns for analysis
        columns_to_analyze = st.multiselect(
            "Select Columns for Analysis",
            ['education_years', 'children', 'age', 'bmi'],
            default=['education_years', 'children']
        )
        
        # Display statistics for selected columns
        for col in columns_to_analyze:
            st.subheader(f"Descriptive Statistics for {col}")
            stats = get_descriptive_stats(df, col)
            
            # Create columns for metrics
            cols = st.columns(len(stats))
            for i, (stat_name, stat_value) in enumerate(stats.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{stat_value:.2f}</div>
                        <div>{stat_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Distribution Analysis
    elif analysis_type == "Distribution Analysis":
        st.header("Distribution Analysis")
        
        # Select column for distribution
        dist_column = st.selectbox(
            "Select Column for Distribution",
            ['education_years', 'children', 'age', 'bmi']
        )
        
        # Color by residence
        color_by_residence = st.checkbox("Color by Residence", value=True)
        
        # Create distribution plot
        if color_by_residence:
            fig = px.histogram(
                df, 
                x=dist_column, 
                color='residence', 
                marginal='box',
                title=f'Distribution of {dist_column}',
                labels={dist_column: dist_column.replace('_', ' ').title()}
            )
        else:
            fig = px.histogram(
                df, 
                x=dist_column, 
                marginal='box',
                title=f'Distribution of {dist_column}',
                labels={dist_column: dist_column.replace('_', ' ').title()}
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Descriptive statistics
        stats = get_descriptive_stats(df, dist_column)
        st.subheader("Descriptive Statistics")
        st.json(stats)
    
    # Regression Analysis
    elif analysis_type == "Regression Analysis":
        st.header("Regression Analysis")
        
        # Select target variable
        target_var = st.selectbox(
            "Select Target Variable",
            ['children', 'education_years', 'bmi']
        )
        
        # Select predictor variables
        predictor_vars = st.multiselect(
            "Select Predictor Variables",
            [col for col in ['education_years', 'children', 'age', 'bmi'] if col != target_var],
            default=['education_years', 'age']
        )
        
        # Perform regression
        if predictor_vars:
            X = df[predictor_vars]
            y = df[target_var]
            
            regression_results = perform_regression(X, y)
            
            st.subheader("Regression Results")
            
            # Display coefficients
            coef_df = pd.DataFrame({
                'Predictor': predictor_vars,
                'Coefficient': regression_results['Coefficients']
            })
            st.dataframe(coef_df)
            
            # R-squared
            st.metric("R-squared", f"{regression_results['R-squared']:.4f}")
            
            # Scatter plot of actual vs predicted
            X_scaled = StandardScaler().fit_transform(X)
            y_pred = LinearRegression().fit(X_scaled, y).predict(X_scaled)
            
            scatter_fig = go.Figure()
            scatter_fig.add_trace(go.Scatter(
                x=y, 
                y=y_pred, 
                mode='markers',
                name='Actual vs Predicted'
            ))
            scatter_fig.add_trace(go.Scatter(
                x=[y.min(), y.max()], 
                y=[y.min(), y.max()], 
                mode='lines',
                name='Perfect Prediction Line'
            ))
            
            scatter_fig.update_layout(
                title=f'Actual vs Predicted {target_var}',
                xaxis_title=f'Actual {target_var}',
                yaxis_title=f'Predicted {target_var}'
            )
            
            st.plotly_chart(scatter_fig, use_container_width=True)

if __name__ == "__main__":
    main()
