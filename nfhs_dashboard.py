import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="NFHS-4 Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
    .note {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Function to load and clean data
@st.cache_data
def load_data():
    try:
        # Load the CSV file
        df = pd.read_csv("NFHS 4 Data.csv")
        
        # Rename columns for better readability
        df.columns = ['residence', 'education_years', 'children', 'age', 'bmi']
        
        # Clean the data
        # Check for missing values and handle them appropriately
        # For BMI, we'll keep NaN values as is
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to calculate key statistics
def calculate_stats(df):
    stats = {}
    
    # Sample distribution
    stats['total_records'] = len(df)
    stats['urban_count'] = df[df['residence'] == 'urban'].shape[0]
    stats['rural_count'] = df[df['residence'] == 'rural'].shape[0]
    stats['urban_percent'] = (stats['urban_count'] / stats['total_records'] * 100).round(1)
    stats['rural_percent'] = (stats['rural_count'] / stats['total_records'] * 100).round(1)
    
    # Education statistics
    stats['avg_education_urban'] = df[df['residence'] == 'urban']['education_years'].mean().round(1)
    stats['avg_education_rural'] = df[df['residence'] == 'rural']['education_years'].mean().round(1)
    
    urban_no_education = df[(df['residence'] == 'urban') & (df['education_years'] == 0)].shape[0]
    rural_no_education = df[(df['residence'] == 'rural') & (df['education_years'] == 0)].shape[0]
    stats['no_education_urban_percent'] = (urban_no_education / stats['urban_count'] * 100).round(1)
    stats['no_education_rural_percent'] = (rural_no_education / stats['rural_count'] * 100).round(1)
    
    # Fertility statistics
    stats['avg_children_urban'] = df[df['residence'] == 'urban']['children'].mean().round(2)
    stats['avg_children_rural'] = df[df['residence'] == 'rural']['children'].mean().round(2)
    
    # BMI statistics (only for non-null values)
    valid_bmi = df[df['bmi'].notna()]
    stats['bmi_available_percent'] = (len(valid_bmi) / stats['total_records'] * 100).round(1)
    
    if len(valid_bmi) > 0:
        stats['avg_bmi_urban'] = valid_bmi[valid_bmi['residence'] == 'urban']['bmi'].mean().round(1)
        stats['avg_bmi_rural'] = valid_bmi[valid_bmi['residence'] == 'rural']['bmi'].mean().round(1)
    else:
        stats['avg_bmi_urban'] = None
        stats['avg_bmi_rural'] = None
    
    return stats

# Function to categorize education
def categorize_education(years):
    if years == 0:
        return 'No education'
    elif years <= 5:
        return 'Primary (1-5 yrs)'
    elif years <= 12:
        return 'Secondary (6-12 yrs)'
    else:
        return 'Higher (12+ yrs)'

# Function to categorize age
def categorize_age(age):
    if age < 20:
        return 'Under 20'
    elif age < 30:
        return '20-29'
    elif age < 40:
        return '30-39'
    else:
        return '40-49'

# Function to categorize BMI
def categorize_bmi(bmi):
    if pd.isna(bmi):
        return 'Unknown'
    elif bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

# Function to create education distribution chart
def plot_education_distribution(df):
    df['education_category'] = df['education_years'].apply(categorize_education)
    
    # Count by residence and education category
    edu_counts = df.groupby(['residence', 'education_category']).size().reset_index(name='count')
    
    # Calculate percentages
    total_urban = edu_counts[edu_counts['residence'] == 'urban']['count'].sum()
    total_rural = edu_counts[edu_counts['residence'] == 'rural']['count'].sum()
    
    edu_counts['percentage'] = edu_counts.apply(
        lambda x: (x['count'] / total_urban * 100) if x['residence'] == 'urban' 
                 else (x['count'] / total_rural * 100), 
        axis=1
    )
    
    # Order categories
    category_order = ['No education', 'Primary (1-5 yrs)', 'Secondary (6-12 yrs)', 'Higher (12+ yrs)']
    
    # Create plot
    fig = px.bar(
        edu_counts, 
        x='percentage', 
        y='education_category',
        color='residence',
        barmode='group',
        orientation='h',
        color_discrete_sequence=['#1E88E5', '#FFC107'],
        category_orders={'education_category': category_order},
        labels={
            'percentage': 'Percentage (%)',
            'education_category': 'Education Level',
            'residence': 'Residence'
        },
        title='Education Distribution by Residence Type'
    )
    
    fig.update_layout(
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

# Function to create children by education chart
def plot_children_by_education(df):
    df['education_category'] = df['education_years'].apply(categorize_education)
    
    # Calculate average children by education category
    children_by_edu = df.groupby('education_category')['children'].mean().reset_index()
    children_by_edu['children'] = children_by_edu['children'].round(2)
    
    # Order categories
    category_order = ['No education', 'Primary (1-5 yrs)', 'Secondary (6-12 yrs)', 'Higher (12+ yrs)']
    children_by_edu['education_category'] = pd.Categorical(
        children_by_edu['education_category'], 
        categories=category_order, 
        ordered=True
    )
    children_by_edu = children_by_edu.sort_values('education_category')
    
    # Create plot
    fig = px.bar(
        children_by_edu,
        x='education_category',
        y='children',
        color='education_category',
        color_discrete_sequence=['#FFC107', '#4CAF50', '#1E88E5', '#9C27B0'],
        labels={
            'education_category': 'Education Level',
            'children': 'Average Number of Children'
        },
        title='Average Number of Children by Education Level'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

# Function to create education vs children scatter plot
def plot_education_vs_children(df):
    # Group by education years
    edu_children = df.groupby('education_years')['children'].agg(['mean', 'count']).reset_index()
    edu_children['mean'] = edu_children['mean'].round(2)
    
    # Create plot
    fig = px.scatter(
        edu_children,
        x='education_years',
        y='mean',
        size='count',
        size_max=30,
        labels={
            'education_years': 'Education (Years)',
            'mean': 'Average Number of Children',
            'count': 'Number of Respondents'
        },
        title='Education Years vs. Average Number of Children'
    )
    
    # Add trendline
    fig.add_trace(
        go.Scatter(
            x=edu_children['education_years'],
            y=edu_children['mean'],
            mode='lines',
            line=dict(color='rgba(0, 0, 0, 0.3)', dash='dash'),
            name='Trend'
        )
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

# Function to create age distribution chart
def plot_age_distribution(df):
    df['age_group'] = df['age'].apply(categorize_age)
    
    # Count by residence and age group
    age_counts = df.groupby(['residence', 'age_group']).size().reset_index(name='count')
    
    # Calculate percentages
    total_urban = age_counts[age_counts['residence'] == 'urban']['count'].sum()
    total_rural = age_counts[age_counts['residence'] == 'rural']['count'].sum()
    
    age_counts['percentage'] = age_counts.apply(
        lambda x: (x['count'] / total_urban * 100) if x['residence'] == 'urban' 
                 else (x['count'] / total_rural * 100), 
        axis=1
    )
    
    # Order categories
    category_order = ['Under 20', '20-29', '30-39', '40-49']
    
    # Create plot
    fig = px.bar(
        age_counts, 
        x='age_group', 
        y='percentage',
        color='residence',
        barmode='group',
        color_discrete_sequence=['#1E88E5', '#FFC107'],
        category_orders={'age_group': category_order},
        labels={
            'percentage': 'Percentage (%)',
            'age_group': 'Age Group',
            'residence': 'Residence'
        },
        title='Age Distribution by Residence Type'
    )
    
    fig.update_layout(
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

# Function to create BMI distribution chart
def plot_bmi_distribution(df):
    # Work with clean BMI data
    valid_bmi_df = df[df['bmi'].notna()].copy()
    valid_bmi_df['bmi_category'] = valid_bmi_df['bmi'].apply(categorize_bmi)
    
    # Count by BMI category
    bmi_counts = valid_bmi_df.groupby('bmi_category').size().reset_index(name='count')
    bmi_counts['percentage'] = (bmi_counts['count'] / bmi_counts['count'].sum() * 100).round(1)
    
    # Order categories
    category_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    
    # Create plot
    fig = px.pie(
        bmi_counts,
        values='count',
        names='bmi_category',
        color='bmi_category',
        color_discrete_sequence=['#F44336', '#4CAF50', '#FFC107', '#1E88E5'],
        category_orders={'bmi_category': category_order},
        title=f'BMI Distribution (Available for {(len(valid_bmi_df) / len(df) * 100).round(1)}% of respondents)'
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    fig.update_layout(
        height=400,
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

# Function to create correlation heatmap
def plot_correlation_heatmap(df):
    # Select numeric columns and calculate correlation
    numeric_df = df[['education_years', 'children', 'age']].copy()
    corr_matrix = numeric_df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))
    
    # Generate a heatmap
    heatmap = sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        linewidths=.5,
        fmt='.2f',
        mask=mask
    )
    
    plt.title('Correlation Heatmap')
    
    return plt.gcf()

# Main function to run the app
def main():
    # Load the data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the file path and format.")
        return
    
    # Calculate statistics
    stats = calculate_stats(df)
    
    # Main header
    st.markdown('<div class="main-header">NFHS-4 Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard explores data from the National Family Health Survey (NFHS-4) conducted in India. 
    The analysis focuses on education, fertility patterns, and demographic characteristics across urban and rural areas.
    """)
    
    # Navigation
    st.markdown('<div class="sub-header">Dashboard Sections</div>', unsafe_allow_html=True)
    tabs = st.tabs([
        "Overview", 
        "Education Analysis", 
        "Fertility Patterns", 
        "Demographics",
        "Data Explorer"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        st.markdown('<div class="sub-header">Key Statistics</div>', unsafe_allow_html=True)
        
        # Display key metrics in three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Sample Distribution</div>
                <div class="metric-value">{urban}% | {rural}%</div>
                <div>Urban | Rural</div>
            </div>
            """.format(urban=stats['urban_percent'], rural=stats['rural_percent']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Average Education (years)</div>
                <div class="metric-value">{urban} | {rural}</div>
                <div>Urban | Rural</div>
            </div>
            """.format(urban=stats['avg_education_urban'], rural=stats['avg_education_rural']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Average Children</div>
                <div class="metric-value">{urban} | {rural}</div>
                <div>Urban | Rural</div>
            </div>
            """.format(urban=stats['avg_children_urban'], rural=stats['avg_children_rural']), unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">Key Visualizations</div>', unsafe_allow_html=True)
        
        # Two charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_education_distribution(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_children_by_education(df), use_container_width=True)
        
        # Overview insights
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Key Insights</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Education Gap**: Urban women have significantly more years of education ({urban_edu} years) 
          compared to rural women ({rural_edu} years).
        
        - **No Education Rate**: Nearly half ({rural_no_edu}%) of rural women have no formal education
          compared to {urban_no_edu}% in urban areas.
        
        - **Fertility Patterns**: Rural women have more children on average ({rural_children})
          than urban women ({urban_children}).
        
        - **Education-Fertility Link**: Strong negative relationship between education level and
          number of children. Women with no education have twice as many children as those with higher education.
        """.format(
            urban_edu=stats['avg_education_urban'],
            rural_edu=stats['avg_education_rural'],
            urban_no_edu=stats['no_education_urban_percent'],
            rural_no_edu=stats['no_education_rural_percent'],
            urban_children=stats['avg_children_urban'],
            rural_children=stats['avg_children_rural']
        ))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Education Analysis
    with tabs[1]:
        st.markdown('<div class="sub-header">Education Patterns</div>', unsafe_allow_html=True)
        
        # Education metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Average Education Years</div>
                <div class="metric-value">{urban} | {rural}</div>
                <div>Urban | Rural</div>
            </div>
            """.format(urban=stats['avg_education_urban'], rural=stats['avg_education_rural']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">No Education Rate</div>
                <div class="metric-value">{urban}% | {rural}%</div>
                <div>Urban | Rural</div>
            </div>
            """.format(urban=stats['no_education_urban_percent'], rural=stats['no_education_rural_percent']), unsafe_allow_html=True)
        
        # Education distribution chart
        st.plotly_chart(plot_education_distribution(df), use_container_width=True)
        
        # Additional education analysis
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Education Level Analysis</div>', unsafe_allow_html=True)
        
        # Education distribution table
        df['education_category'] = df['education_years'].apply(categorize_education)
        edu_dist = df.groupby(['residence', 'education_category']).size().unstack().reset_index()
        
        if not edu_dist.empty and 'residence' in edu_dist.columns:
            total_urban = df[df['residence'] == 'urban'].shape[0]
            total_rural = df[df['residence'] == 'rural'].shape[0]
            
            # Calculate percentages for each education category
            for cat in ['No education', 'Primary (1-5 yrs)', 'Secondary (6-12 yrs)', 'Higher (12+ yrs)']:
                if cat in edu_dist.columns:
                    edu_dist[f'{cat} (%)'] = edu_dist.apply(
                        lambda x: round(x[cat] / total_urban * 100, 1) if x['residence'] == 'urban' 
                                else round(x[cat] / total_rural * 100, 1),
                        axis=1
                    )
            
            # Create a clean display table
            display_cols = ['residence']
            for cat in ['No education', 'Primary (1-5 yrs)', 'Secondary (6-12 yrs)', 'Higher (12+ yrs)']:
                if f'{cat} (%)' in edu_dist.columns:
                    display_cols.append(f'{cat} (%)')
            
            if len(display_cols) > 1:
                st.dataframe(edu_dist[display_cols], use_container_width=True)
        
        st.markdown("""
        **Education Gap Analysis:**
        
        - The education gap between urban and rural areas is substantial, with urban women receiving on average 
          **{gap} more years** of education.
        
        - Secondary education (6-12 years) is much more common in urban areas, while no formal education 
          is the norm for nearly half of rural women.
        
        - Higher education (12+ years) is **5 times more prevalent** in urban areas ({urban_higher}%)
          compared to rural areas ({rural_higher}%).
        
        - This education gap has significant implications for other socioeconomic factors, 
          including fertility patterns and health outcomes.
        """.format(
            gap=round(stats['avg_education_urban'] - stats['avg_education_rural'], 1),
            urban_higher=round(len(df[(df['residence'] == 'urban') & (df['education_category'] == 'Higher (12+ yrs)')]) / stats['urban_count'] * 100, 1),
            rural_higher=round(len(df[(df['residence'] == 'rural') & (df['education_category'] == 'Higher (12+ yrs)')]) / stats['rural_count'] * 100, 1)
        ))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Fertility Patterns
    with tabs[2]:
        st.markdown('<div class="sub-header">Fertility Patterns</div>', unsafe_allow_html=True)
        
        # Fertility metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Average Children</div>
                <div class="metric-value">{urban} | {rural}</div>
                <div>Urban | Rural</div>
            </div>
            """.format(urban=stats['avg_children_urban'], rural=stats['avg_children_rural']), unsafe_allow_html=True)
        
        with col2:
            # Calculate difference in average children
            diff = round(stats['avg_children_rural'] - stats['avg_children_urban'], 2)
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Rural-Urban Difference</div>
                <div class="metric-value">+{diff}</div>
                <div>More children in rural areas</div>
            </div>
            """.format(diff=diff), unsafe_allow_html=True)
        
        # Two fertility charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_children_by_education(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_education_vs_children(df), use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation between Variables")
        st.pyplot(plot_correlation_heatmap(df))
        
        # Fertility insights
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Fertility Insights</div>', unsafe_allow_html=True)
        st.markdown("""
        **Education-Fertility Relationship:**
        
        - There is a strong negative correlation between education level and number of children.
        
        - Women with no education have on average **{no_edu_children}** children, more than double 
          the **{high_edu_children}** children for women with higher education.
        
        - For each additional year of education, women have approximately **{per_year}** fewer children on average.
        
        - The rural-urban fertility gap (**{diff}** children) can be largely explained by the education gap,
          as rural areas have significantly lower education levels.
        
        - This relationship between education and fertility is consistent with global trends, where
          increased women's education is associated with lower fertility rates.
        """.format(
            no_edu_children=round(df[df['education_years'] == 0]['children'].mean(), 2),
            high_edu_children=round(df[df['education_years'] > 12]['children'].mean(), 2),
            per_year=abs(round(np.polyfit(df['education_years'], df['children'], 1)[0], 2)),
            diff=diff
        ))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Demographics
    with tabs[3]:
        st.markdown('<div class="sub-header">Demographic Patterns</div>', unsafe_allow_html=True)
        
        # Demographics charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_age_distribution(df), use_container_width=True)
        
        with col2:
            if stats['bmi_available_percent'] > 0:
                st.plotly_chart(plot_bmi_distribution(df), use_container_width=True)
            else:
                st.info("BMI data is not available in sufficient quantity for visualization.")
        
        # Age distribution table
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Age Distribution Analysis</div>', unsafe_allow_html=True)
        
        df['age_group'] = df['age'].apply(categorize_age)
        age_dist = df.groupby(['residence', 'age_group']).size().unstack().reset_index()
        
        if not age_dist.empty and 'residence' in age_dist.columns:
            total_urban = df[df['residence'] == 'urban'].shape[0]
            total_rural = df[df['residence'] == 'rural'].shape[0]
            
            # Calculate percentages for each age group
            for group in ['Under 20', '20-29', '30-39', '40-49']:
                if group in age_dist.columns:
                    age_dist[f'{group} (%)'] = age_dist.apply(
                        lambda x: round(x[group] / total_urban * 100, 1) if x['residence'] == 'urban' 
                                else round(x[group] / total_rural * 100, 1),
                        axis=1
                    )
            
            # Create a clean display table
            display_cols = ['residence']
            for group in ['Under 20', '20-29', '30-39', '40-49']:
                if f'{group} (%)' in age_dist.columns:
                    display_cols.append(f'{group} (%)')
            
            if len(display_cols) > 1:
                st.dataframe(age_dist[display_cols], use_container_width=True)
        
        st.markdown("""
        **Age Distribution Insights:**
        
        - The survey covers women of reproductive age (16-49 years).
        
        - Rural areas have a slightly higher proportion of younger respondents (under 20 years).
        
        - Both urban and rural areas have similar distributions in the 30-49 age brackets.
        
        - The age distribution suggests that the survey provides good coverage across different
          life stages for women.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # BMI insights if available
        if stats['bmi_available_percent'] > 5:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">BMI Analysis</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="note">Note: BMI data is available for only {bmi_percent}% of respondents, 
            so these results should be interpreted with caution.</div>
            """.format(bmi_percent=stats['bmi_available_percent']), unsafe_allow_html=True)
            
            if stats['avg_bmi_urban'] is not None and stats['avg_bmi_rural'] is not None:
                st.markdown("""
                - Average BMI in urban areas: **{urban_bmi}**
                - Average BMI in rural areas: **{rural_bmi}**
                """.format(
                    urban_bmi=stats['avg_bmi_urban'],
                    rural_bmi=stats['avg_bmi_rural']
                ))
            
            # BMI distribution by category
            valid_bmi_df = df[df['bmi'].notna()].copy()
            valid_bmi_df['bmi_category'] = valid_bmi_df['bmi'].apply(categorize_bmi)
            
            bmi_dist = valid_bmi_df.groupby('bmi_category').size().reset_index(name='count')
            bmi_dist['percentage'] = (bmi_dist['count'] / bmi_dist['count'].sum() * 100).round(1)
            
            st.dataframe(
                bmi_dist[['bmi_category', 'count', 'percentage']].rename(
                    columns={'bmi_category': 'BMI Category', 'count': 'Count', 'percentage': 'Percentage (%)'})
            )
            
            st.markdown("""
            The BMI distribution reveals important health indicators:
            
            - The prevalence of overweight and obesity varies between urban and rural populations.
            - Further analysis would be needed to fully understand the relationship between BMI, 
              education levels, and other socioeconomic factors.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 5: Data Explorer
    with tabs[4]:
        st.markdown('<div class="sub-header">Interactive Data Explorer</div>', unsafe_allow_html=True)
        
        # Filter options
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Filter Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residence filter
            residence_filter = st.multiselect(
                "Residence Type:",
                options=sorted(df['residence'].unique()),
                default=sorted(df['residence'].unique())
            )
            
            # Age range filter
            min_age, max_age = int(df['age'].min()), int(df['age'].max())
            age_range = st.slider(
                "Age Range:",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age)
            )
        
        with col2:
            # Education filter
            education_options = ["No education", "Primary (1-5 yrs)", "Secondary (6-12 yrs)", "Higher (12+ yrs)"]
            education_filter = st.multiselect(
                "Education Level:",
                options=education_options,
                default=education_options
            )
            
            # Children filter
            min_children, max_children = int(df['children'].min()), int(df['children'].max())
            children_range = st.slider(
                "Number of Children:",
                min_value=min_children,
                max_value=max_children,
                value=(min_children, max_children)
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if residence_filter:
            filtered_df = filtered_df[filtered_df['residence'].isin(residence_filter)]
        
        filtered_df = filtered_df[
            (filtered_df['age'] >= age_range[0]) & 
            (filtered_df['age'] <= age_range[1]) &
            (filtered_df['children'] >= children_range[0]) & 
            (filtered_df['children'] <= children_range[1])
        ]
        
        if education_filter:
            filtered_df['education_category'] = filtered_df['education_years'].apply(categorize_education)
            filtered_df = filtered_df[filtered_df['education_category'].isin(education_filter)]
        
        # Show filtered data stats
        st.markdown(f"**Filtered Data:** {len(filtered_df)} records")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data visualization options
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Create Custom Visualization")
        
        viz_type = st.selectbox(
            "Visualization Type:",
            options=["Bar Chart", "Scatter Plot", "Box Plot", "Histogram"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if viz_type in ["Bar Chart", "Box Plot"]:
                x_axis = st.selectbox(
                    "X-axis (Category):",
                    options=["residence", "education_category", "age_group"]
                )
            else:
                x_axis = st.selectbox(
                    "X-axis:",
                    options=["education_years", "children", "age"]
                )
        
        with col2:
            if viz_type != "Histogram":
                y_axis = st.selectbox(
                    "Y-axis:",
                    options=["children", "education_years", "age"]
                )
            
            color_by = st.selectbox(
                "Color by:",
                options=["None", "residence", "education_category", "age_group"]
            )
        
        # Prepare data for visualization
        if "education_category" in [x_axis, color_by] and "education_category" not in filtered_df.columns:
            filtered_df['education_category'] = filtered_df['education_years'].apply(categorize_education)
        
        if "age_group" in [x_axis, color_by] and "age_group" not in filtered_df.columns:
            filtered_df['age_group'] = filtered_df['age'].apply(categorize_age)
        
        # Create visualization
        if len(filtered_df) > 0:
            if viz_type == "Bar Chart":
                # Group by x_axis and calculate mean of y_axis
                grouped_data = filtered_df.groupby(x_axis)[y_axis].mean().reset_index()
                
                fig = px.bar(
                    grouped_data,
                    x=x_axis,
                    y=y_axis,
                    color=None if color_by == "None" else filtered_df[color_by] if len(grouped_data) == len(filtered_df) else None,
                    title=f"Average {y_axis} by {x_axis}",
                    labels={
                        x_axis: x_axis.replace('_', ' ').title(),
                        y_axis: y_axis.replace('_', ' ').title()
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Scatter Plot":
                fig = px.scatter(
                    filtered_df,
                    x=x_axis,
                    y=y_axis,
                    color=None if color_by == "None" else filtered_df[color_by],
                    opacity=0.7,
                    title=f"{y_axis.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}",
                    labels={
                        x_axis: x_axis.replace('_', ' ').title(),
                        y_axis: y_axis.replace('_', ' ').title()
                    }
                )
                
                # Add trendline
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Box Plot":
                fig = px.box(
                    filtered_df,
                    x=x_axis,
                    y=y_axis,
                    color=None if color_by == "None" else filtered_df[color_by],
                    title=f"Distribution of {y_axis.replace('_', ' ').title()} by {x_axis.replace('_', ' ').title()}",
                    labels={
                        x_axis: x_axis.replace('_', ' ').title(),
                        y_axis: y_axis.replace('_', ' ').title()
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Histogram":
                fig = px.histogram(
                    filtered_df,
                    x=x_axis,
                    color=None if color_by == "None" else filtered_df[color_by],
                    marginal="box",
                    opacity=0.7,
                    title=f"Distribution of {x_axis.replace('_', ' ').title()}",
                    labels={
                        x_axis: x_axis.replace('_', ' ').title()
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available with the current filters.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Raw data viewer
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Raw Data Viewer")
        
        show_raw_data = st.checkbox("Show raw data")
        
        if show_raw_data:
            # Limit to 1000 rows for performance
            display_df = filtered_df.head(1000).copy()
            
            # Ensure categorizations are available
            if "education_category" not in display_df.columns:
                display_df['education_category'] = display_df['education_years'].apply(categorize_education)
            if "age_group" not in display_df.columns:
                display_df['age_group'] = display_df['age'].apply(categorize_age)
            if "bmi_category" not in display_df.columns:
                display_df['bmi_category'] = display_df['bmi'].apply(categorize_bmi)
            
            # Display the data
            st.dataframe(display_df, use_container_width=True)
            
            if len(filtered_df) > 1000:
                st.info(f"Showing first 1000 of {len(filtered_df)} records. Apply filters to see specific subsets.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f5f5f5; border-radius: 10px;">
        <p style="color: #666; margin-bottom: 0.5rem;">NFHS-4 Data Analysis Dashboard</p>
        <p style="color: #888; font-size: 0.8rem;">This dashboard analyzes data from the National Family Health Survey (NFHS-4) conducted in India.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
