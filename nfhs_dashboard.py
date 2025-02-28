import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
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

# Create correlation heatmap
def create_correlation_heatmap(df):
    # Select numeric columns
    numeric_cols = ['education_years', 'children', 'age', 'bmi']
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Numeric Variables')
    return plt.gcf()

# Create pairplot
def create_pairplot(df):
    # Select numeric columns
    numeric_cols = ['education_years', 'children', 'age', 'bmi']
    
    # Create figure
    plt.figure(figsize=(12, 10))
    plot_df = df[numeric_cols + ['residence']]
    
    # Use seaborn pairplot
    g = sns.pairplot(plot_df, hue='residence', 
                     plot_kws={'alpha': 0.5},
                     diag_kws={'alpha': 0.7})
    g.fig.suptitle('Pairplot of Numeric Variables by Residence', y=1.02)
    return g.fig

# Boxplot comparison
def create_boxplot_comparison(df):
    # Prepare figure
    plt.figure(figsize=(12, 6))
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Boxplots of Variables by Residence', fontsize=16)
    
    # Variables to plot
    variables = ['education_years', 'children', 'age', 'bmi']
    
    # Create boxplots
    for i, var in enumerate(variables):
        row = i // 2
        col = i % 2
        
        sns.boxplot(x='residence', y=var, data=df, ax=axes[row, col])
        axes[row, col].set_title(f'{var.replace("_", " ").title()} by Residence')
    
    plt.tight_layout()
    return fig

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
            "Regression Analysis",
            "Correlation & Visualization",
            "Comparative Graphs"
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
    
    # Correlation & Visualization
    elif analysis_type == "Correlation & Visualization":
        st.header("Correlation & Advanced Visualizations")
        
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        corr_heatmap = create_correlation_heatmap(df)
        st.pyplot(corr_heatmap)
        
        # Pairplot
        st.subheader("Pairplot of Variables")
        pairplot = create_pairplot(df)
        st.pyplot(pairplot)
    
    # Comparative Graphs
    elif analysis_type == "Comparative Graphs":
        st.header("Comparative Visualizations")
        
        # Boxplot Comparison
        st.subheader("Boxplots by Residence")
        boxplot_fig = create_boxplot_comparison(df)
        st.pyplot(boxplot_fig)
        
        # Bar chart comparing means
        st.subheader("Mean Comparison by Residence")
        
        # Variables to compare
        compare_vars = ['education_years', 'children', 'age', 'bmi']
        
        # Prepare data for comparison
        means_by_residence = df.groupby('residence')[compare_vars].mean()
        
        # Create bar chart
        fig = px.bar(
            means_by_residence.reset_index(), 
            x='residence', 
            y=compare_vars,
            title='Mean Comparison of Variables by Residence',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the means
        st.dataframe(means_by_residence)

if __name__ == "__main__":
    main()
