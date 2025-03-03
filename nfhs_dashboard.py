import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
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
.stat-card {
    background-color: #f0f8ff;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.stat-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1E88E5;
}
.stat-label {
    font-size: 0.9rem;
    color: #666;
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

# Comprehensive descriptive statistics
def get_detailed_stats(df, column):
    return {
        'Mean': df[column].mean(),
        'Median': df[column].median(),
        'Standard Deviation': df[column].std(),
        'Minimum': df[column].min(),
        'Maximum': df[column].max(),
        'Skewness': df[column].skew(),
        'Kurtosis': df[column].kurtosis()
    }

# Create scatter plot with regression line
def create_scatter_with_regression(df, x_col, y_col):
    # Prepare data
    X = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values
    
    # Perform linear regression
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Calculate R-squared
    r_squared = reg.score(X, y)
    
    # Create scatter plot with regression line
    fig = go.Figure()
    
    # Scatter plot points
    fig.add_trace(go.Scatter(
        x=df[x_col], 
        y=df[y_col],
        mode='markers',
        name='Data Points',
        marker=dict(
            color=df['residence'].map({'urban': '#1E88E5', 'rural': '#FFC107'}),
            opacity=0.7
        )
    ))
    
    # Regression line
    fig.add_trace(go.Scatter(
        x=df[x_col], 
        y=y_pred,
        mode='lines',
        name='Regression Line',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Scatter Plot: {x_col} vs {y_col} with Regression Line',
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'R² = {r_squared:.4f}',
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    return fig

# Correlation heatmap
def create_correlation_heatmap(df):
    # Select numeric columns
    numeric_cols = ['education_years', 'children', 'age', 'bmi']
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1, 
        zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={'size':10},
    ))
    
    fig.update_layout(
        title='Correlation Heatmap of Numeric Variables',
        height=500
    )
    
    return fig

# Box plot by residence
def create_boxplot_by_residence(df, column):
    fig = px.box(
        df, 
        x='residence', 
        y=column, 
        title=f'Distribution of {column} by Residence',
        color='residence',
        color_discrete_sequence=['#1E88E5', '#FFC107']
    )
    return fig

# Main Streamlit app
def main():
    st.markdown('<div class="main-header">NFHS-4 Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar for analysis type
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Descriptive Statistics", 
            "Scatter & Regression Analysis",
            "Correlation Analysis",
            "Distribution Comparison"
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
        
        # Create columns for stats display
        for column in columns_to_analyze:
            st.subheader(f"Descriptive Statistics for {column}")
            
            # Get detailed statistics
            stats = get_detailed_stats(df, column)
            
            # Display stats in a grid
            cols = st.columns(len(stats))
            for i, (stat_name, stat_value) in enumerate(stats.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{stat_value:.2f}</div>
                        <div class="stat-label">{stat_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Scatter & Regression Analysis
    elif analysis_type == "Scatter & Regression Analysis":
        st.header("Scatter Plots with Regression Analysis")
        
        # Select variables for scatter plot
        x_column = st.selectbox("Select X-axis Variable", 
            ['education_years', 'children', 'age', 'bmi'])
        y_column = st.selectbox("Select Y-axis Variable", 
            ['children', 'education_years', 'age', 'bmi'], 
            index=0 if x_column != 'children' else 1)
        
        # Ensure different variables are selected
        if x_column == y_column:
            st.warning("Please select different variables for X and Y axes.")
        else:
            # Create scatter plot with regression line
            scatter_fig = create_scatter_with_regression(df, x_column, y_column)
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Additional statistical insights
            X = df[x_column].values.reshape(-1, 1)
            y = df[y_column].values
            
            # Perform linear regression
            reg = LinearRegression().fit(X, y)
            
            # Calculate correlation
            correlation, p_value = stats.pearsonr(df[x_column], df[y_column])
            
            st.subheader("Regression Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Correlation Coefficient", f"{correlation:.4f}")
            with col2:
                st.metric("P-value", f"{p_value:.4f}")
    
    # Correlation Analysis
    elif analysis_type == "Correlation Analysis":
        st.header("Correlation Analysis")
        
        # Create correlation heatmap
        corr_fig = create_correlation_heatmap(df)
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Interpretation of correlations
        st.subheader("Correlation Interpretation")
        corr_matrix = df[['education_years', 'children', 'age', 'bmi']].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
        
        # Sort by absolute correlation
        sorted_corr = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
        
        # Display top correlations
        st.write("Top Correlations:")
        for var1, var2, corr_value in sorted_corr[:3]:
            st.markdown(f"- **{var1}** and **{var2}**: {corr_value:.4f}")
    
    # Distribution Comparison
    elif analysis_type == "Distribution Comparison":
        st.header("Distribution Comparison by Residence")
        
        # Select column for distribution comparison
        dist_column = st.selectbox(
            "Select Variable to Compare", 
            ['education_years', 'children', 'age', 'bmi']
        )
        
        # Create boxplot
        boxplot_fig = create_boxplot_by_residence(df, dist_column)
        st.plotly_chart(boxplot_fig, use_container_width=True)
        
        # Compute and display summary statistics by residence
        st.subheader("Summary Statistics by Residence")
        residence_stats = df.groupby('residence')[dist_column].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        st.dataframe(residence_stats)

if __name__ == "__main__":
    main()
