import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.data_analyzer import DataAnalyzer
from utils.ml_trainer import MLTrainer
from utils.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Interactive Data Science Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

def main():
    st.title("Interactive Data Science Platform")
    st.markdown("Upload your CSV data and get automated analysis, visualizations, and machine learning insights!")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # File upload section
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file to begin analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.sidebar.success(f"File uploaded successfully!")
            st.sidebar.info(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
            
            # Navigation options
            analysis_option = st.sidebar.selectbox(
                "Choose Analysis Type",
                ["Data Overview", "Exploratory Data Analysis", "Machine Learning", "Download Report"]
            )
            
            # Main content area
            if analysis_option == "Data Overview":
                show_data_overview(data)
            elif analysis_option == "Exploratory Data Analysis":
                show_eda(data)
            elif analysis_option == "Machine Learning":
                show_ml_section(data)
            elif analysis_option == "Download Report":
                show_report_section(data)
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format.")
    
    else:
        # Welcome screen
        show_welcome_screen()

def show_welcome_screen():
    st.markdown("""
    ## Welcome to the Interactive Data Science Platform! 
    
    This platform provides automated data analysis and machine learning capabilities for your CSV datasets.
    
    ### Features:
    - **Automated Exploratory Data Analysis**: Get instant insights about your data
    - **Interactive Visualizations**: Dynamic charts and plots
    - **Machine Learning**: Automated model training and evaluation
    - **Downloadable Reports**: Generate comprehensive analysis reports
    
    ### Getting Started:
    1. Upload a CSV file using the sidebar
    2. Explore your data with automated analysis
    3. Train machine learning models
    4. Download your analysis report
    
    ---
    ****
    """)

def show_data_overview(data):
    st.header("Data Overview")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", data.shape[0])
    with col2:
        st.metric("Total Columns", data.shape[1])
    with col3:
        st.metric("Numeric Columns", len(data.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Text Columns", len(data.select_dtypes(include=['object']).columns))
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Data types and missing values
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Column Information")
        info_df = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes.astype(str),
            'Non-Null Count': data.count(),
            'Null Count': data.isnull().sum(),
            'Null Percentage': round((data.isnull().sum() / len(data)) * 100, 2)
        })
        st.dataframe(info_df, use_container_width=True)
    
    with col2:
        st.subheader("Summary Statistics")
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.dataframe(numeric_data.describe(), use_container_width=True)
        else:
            st.info("No numeric columns found for summary statistics.")

def show_eda(data):
    st.header("Exploratory Data Analysis")
    
    analyzer = DataAnalyzer(data)
    
    # Visualization options
    viz_type = st.selectbox(
        "Choose Visualization Type",
        ["Distribution Analysis", "Correlation Analysis", "Scatter Plot Analysis", "Missing Data Analysis"]
    )
    
    if viz_type == "Distribution Analysis":
        show_distribution_analysis(data, analyzer)
    elif viz_type == "Correlation Analysis":
        show_correlation_analysis(data, analyzer)
    elif viz_type == "Scatter Plot Analysis":
        show_scatter_analysis(data, analyzer)
    elif viz_type == "Missing Data Analysis":
        show_missing_data_analysis(data, analyzer)

def show_distribution_analysis(data, analyzer):
    st.subheader("Distribution Analysis")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_columns:
        st.markdown("### Numeric Distributions")
        selected_numeric = st.multiselect(
            "Select numeric columns to analyze",
            numeric_columns,
            default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
        )
        
        if selected_numeric:
            for col in selected_numeric:
                fig = analyzer.create_distribution_plot(col)
                st.plotly_chart(fig, use_container_width=True)
    
    if categorical_columns:
        st.markdown("### Categorical Distributions")
        selected_categorical = st.selectbox(
            "Select a categorical column to analyze",
            categorical_columns
        )
        
        if selected_categorical:
            fig = analyzer.create_categorical_plot(selected_categorical)
            st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(data, analyzer):
    st.subheader("Correlation Analysis")
    
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return
    
    # Correlation heatmap
    fig = analyzer.create_correlation_heatmap()
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations
    st.markdown("### Strongest Correlations")
    correlations = analyzer.get_top_correlations()
    if not correlations.empty:
        st.dataframe(correlations, use_container_width=True)
    else:
        st.info("No significant correlations found.")

def show_scatter_analysis(data, analyzer):
    st.subheader("Scatter Plot Analysis")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.warning("Need at least 2 numeric columns for scatter plot analysis.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("Select X-axis", numeric_columns)
    with col2:
        y_axis = st.selectbox("Select Y-axis", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
    
    # Optional color coding
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    color_by = st.selectbox("Color by (optional)", ["None"] + categorical_columns)
    
    if x_axis and y_axis:
        fig = analyzer.create_scatter_plot(x_axis, y_axis, color_by if color_by != "None" else None)
        st.plotly_chart(fig, use_container_width=True)

def show_missing_data_analysis(data, analyzer):
    st.subheader("Missing Data Analysis")
    
    # Missing data overview
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if missing_data.empty:
        st.success("No missing data found in your dataset!")
        return
    
    # Missing data visualization
    fig = analyzer.create_missing_data_plot()
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing data summary table
    st.markdown("### Missing Data Summary")
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': round((missing_data.values / len(data)) * 100, 2)
    })
    st.dataframe(missing_df, use_container_width=True)

def show_ml_section(data):
    st.header("Machine Learning")
    
    # Check if data has enough columns
    if data.shape[1] < 2:
        st.warning("Need at least 2 columns for machine learning (features + target).")
        return
    
    trainer = MLTrainer(data)
    
    # Model configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target selection
        target_column = st.selectbox(
            "Select Target Column",
            data.columns.tolist(),
            help="Choose the column you want to predict"
        )
    
    with col2:
        # Problem type detection
        if data[target_column].dtype == 'object' or data[target_column].nunique() <= 10:
            problem_type = "Classification"
        else:
            problem_type = st.selectbox(
                "Problem Type",
                ["Regression", "Classification"],
                help="Choose based on your target variable"
            )
    
    # Feature selection
    available_features = [col for col in data.columns if col != target_column]
    selected_features = st.multiselect(
        "Select Features",
        available_features,
        default=available_features,
        help="Choose which columns to use as features for prediction"
    )
    
    if not selected_features:
        st.warning("Please select at least one feature column.")
        return
    
    # Model selection
    if problem_type == "Classification":
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "Logistic Regression"]
        )
    else:
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "Linear Regression"]
        )
    
    # Train model button
    if st.button("Train Model", type="primary"):
        try:
            with st.spinner("Training model..."):
                results = trainer.train_model(
                    target_column, 
                    selected_features, 
                    problem_type, 
                    model_type
                )
                st.session_state.model_results = results
            
            st.success("Model trained successfully!")
            
            # Display results
            show_model_results(results, problem_type)
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")

def show_model_results(results, problem_type):
    st.subheader("Model Results")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Performance Metrics")
        if problem_type == "Classification":
            st.metric("Accuracy", f"{results['accuracy']:.3f}")
            st.metric("Training Accuracy", f"{results['train_accuracy']:.3f}")
        else:
            st.metric("R² Score", f"{results['r2_score']:.3f}")
            st.metric("RMSE", f"{results['rmse']:.3f}")
    
    with col2:
        st.markdown("### Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': results['feature_names'],
            'Importance': results['feature_importance']
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Feature Importance"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    if problem_type == "Classification":
        st.markdown("### Classification Report")
        st.text(results['classification_report'])
    
    # Model predictions vs actual (for regression)
    if problem_type == "Regression":
        st.markdown("###  Predictions vs Actual")
        pred_df = pd.DataFrame({
            'Actual': results['y_test'],
            'Predicted': results['predictions']
        })
        
        fig = px.scatter(
            pred_df, 
            x='Actual', 
            y='Predicted',
            title="Predictions vs Actual Values"
        )
        # Add perfect prediction line
        min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
        max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
        fig.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig, use_container_width=True)

def show_report_section(data):
    st.header("Download Report")
    
    st.markdown("""
    Generate and download a comprehensive analysis report of your dataset.
    The report will include:
    - Data overview and summary statistics
    - Exploratory data analysis insights
    - Machine learning results (if available)
    """)
    
    # Report options
    include_ml = st.checkbox(
        "Include ML Results", 
        value=bool(st.session_state.model_results),
        disabled=not bool(st.session_state.model_results)
    )
    
    if st.button("Generate Report", type="primary"):
        try:
            generator = ReportGenerator(data)
            
            # Generate report
            report_content = generator.generate_report(
                ml_results=st.session_state.model_results if include_ml else None
            )
            
            # Create download button
            st.download_button(
                label="📥 Download Report",
                data=report_content,
                file_name=f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            st.success("Report generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main()
