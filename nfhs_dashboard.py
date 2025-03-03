import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Flexible Data Analysis Dashboard", page_icon="📊", layout="wide")

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

# Comprehensive descriptive statistics
def get_detailed_stats(series):
    return {
        'Mean': series.mean(),
        'Median': series.median(),
        'Standard Deviation': series.std(),
        'Minimum': series.min(),
        'Maximum': series.max(),
        'Skewness': series.skew(),
        'Kurtosis': series.kurtosis()
    }

# Create scatter plot with regression line
def create_scatter_with_regression(df, x_col, y_col, color_col=None):
    # Prepare data
    X = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values
    
    # Perform linear regression
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Calculate R-squared and other metrics
    r_squared = reg.score(X, y)
    
    # Create scatter plot with regression line
    fig = go.Figure()
    
    # Scatter plot points
    scatter_kwargs = {
        'x': df[x_col], 
        'y': df[y_col],
        'mode': 'markers',
        'name': 'Data Points',
        'marker': {'opacity': 0.7}
    }
    
    # Add color if specified
    if color_col and color_col in df.columns:
        scatter_kwargs['color'] = df[color_col]
        scatter_kwargs['color_discrete_sequence'] = px.colors.qualitative.Plotly
    
    fig.add_trace(go.Scatter(**scatter_kwargs))
    
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
        xaxis_title=x_col,
        yaxis_title=y_col,
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
    
    return fig, reg, r_squared

# Correlation heatmap
def create_correlation_heatmap(df):
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
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
        height=600,
        width=800
    )
    
    return fig

# Boxplot by a categorical column
def create_boxplot(df, numeric_col, category_col):
    fig = px.box(
        df, 
        x=category_col, 
        y=numeric_col, 
        title=f'Distribution of {numeric_col} by {category_col}',
        color=category_col
    )
    return fig

# Main Streamlit app
def main():
    st.markdown('<div class="main-header">Flexible Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for file upload and configuration
    st.sidebar.header("Data Configuration")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        try:
            df = pd.read_csv(uploaded_file)
            
            # Allow column renaming
            st.sidebar.header("Rename Columns (Optional)")
            renamed_columns = {}
            for col in df.columns:
                new_name = st.sidebar.text_input(f"Rename '{col}' to:", col)
                renamed_columns[col] = new_name
            
            # Rename columns if needed
            df.columns = [renamed_columns.get(col, col) for col in df.columns]
            
            # Select analysis type
            analysis_type = st.sidebar.selectbox(
                "Select Analysis Type",
                [
                    "Descriptive Statistics", 
                    "Regression Analysis",
                    "Correlation Analysis",
                    "Distribution Comparison"
                ]
            )
            
            # Identify numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Descriptive Statistics
            if analysis_type == "Descriptive Statistics":
                st.header("Descriptive Statistics")
                
                # Select numeric columns for analysis
                columns_to_analyze = st.multiselect(
                    "Select Columns for Analysis",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                # Display statistics for selected columns
                for column in columns_to_analyze:
                    st.subheader(f"Descriptive Statistics for {column}")
                    
                    # Get detailed statistics
                    stats = get_detailed_stats(df[column])
                    
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
            
            # Regression Analysis
            elif analysis_type == "Regression Analysis":
                st.header("Regression Analysis")
                
                # Select variables for regression
                st.subheader("Select Variables for Regression")
                
                # X variables (predictors)
                x_columns = st.multiselect(
                    "Select Predictor Variables (X)",
                    numeric_cols,
                    default=numeric_cols[:len(numeric_cols)//2] if len(numeric_cols) > 1 else numeric_cols
                )
                
                # Y variable (target)
                y_column = st.selectbox(
                    "Select Target Variable (Y)",
                    numeric_cols,
                    index=len(numeric_cols)//2 if len(numeric_cols) > 1 else 0
                )
                
                # Color by categorical variable (optional)
                color_column = st.selectbox(
                    "Optional: Color by Categorical Variable",
                    ['None'] + categorical_cols,
                    index=0
                )
                color_column = color_column if color_column != 'None' else None
                
                # Perform regression if variables are selected
                if x_columns and y_column and x_columns != [y_column]:
                    # Prepare data
                    X = df[x_columns]
                    y = df[y_column]
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Perform regression
                    reg = LinearRegression().fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = reg.predict(X_test)
                    
                    # Evaluation metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r_squared = r2_score(y_test, y_pred)
                    
                    # Display regression results
                    st.subheader("Regression Results")
                    
                    # Coefficients
                    coef_df = pd.DataFrame({
                        'Predictor': x_columns,
                        'Coefficient': reg.coef_
                    })
                    st.dataframe(coef_df)
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R-squared", f"{r_squared:.4f}")
                    with col2:
                        st.metric("Mean Squared Error", f"{mse:.4f}")
                    
                    # Create scatter plots for each predictor
                    for x_col in x_columns:
                        # Create scatter plot
                        scatter_fig, _, r_sq = create_scatter_with_regression(
                            df, x_col, y_column, 
                            color_col=color_column
                        )
                        st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Correlation Analysis
            elif analysis_type == "Correlation Analysis":
                st.header("Correlation Analysis")
                
                # Create correlation heatmap
                corr_fig = create_correlation_heatmap(df[numeric_cols])
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Interpretation of correlations
                st.subheader("Correlation Insights")
                corr_matrix = df[numeric_cols].corr()
                
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
                for var1, var2, corr_value in sorted_corr[:5]:
                    st.markdown(f"- **{var1}** and **{var2}**: {corr_value:.4f}")
            
            # Distribution Comparison
            elif analysis_type == "Distribution Comparison":
                st.header("Distribution Comparison")
                
                # Select numeric column to compare
                numeric_column = st.selectbox(
                    "Select Numeric Variable", 
                    numeric_cols
                )
                
                # Select categorical column for comparison
                if categorical_cols:
                    category_column = st.selectbox(
                        "Compare by Categorical Variable", 
                        categorical_cols
                    )
                    
                    # Create boxplot
                    boxplot_fig = create_boxplot(df, numeric_column, category_column)
                    st.plotly_chart(boxplot_fig, use_container_width=True)
                    
                    # Compute and display summary statistics
                    st.subheader("Summary Statistics")
                    grouped_stats = df.groupby(category_column)[numeric_column].agg([
                        'mean', 'median', 'std', 'min', 'max'
                    ]).round(2)
                    st.dataframe(grouped_stats)
                else:
                    st.warning("No categorical columns found for comparison.")
        
        except Exception as e:
            st.error(f"Error processing the file: {e}")
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()
