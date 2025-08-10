import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    def create_distribution_plot(self, column):
        """Create a distribution plot for a numeric column"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Histogram', 'Box Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=self.data[column],
                name='Distribution',
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=self.data[column],
                name='Box Plot',
                boxpoints='outliers'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Distribution Analysis: {column}',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_categorical_plot(self, column):
        """Create a bar plot for a categorical column"""
        value_counts = self.data[column].value_counts().head(20)  # Top 20 categories
        
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Distribution of {column}',
            labels={'x': column, 'y': 'Count'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        return fig
    
    def create_correlation_heatmap(self):
        """Create a correlation heatmap for numeric columns"""
        if len(self.numeric_columns) < 2:
            return None
        
        correlation_matrix = self.data[self.numeric_columns].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def get_top_correlations(self, threshold=0.5):
        """Get the strongest correlations above threshold"""
        if len(self.numeric_columns) < 2:
            return pd.DataFrame()
        
        correlation_matrix = self.data[self.numeric_columns].corr()
        
        # Get upper triangle of correlation matrix
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find correlations above threshold
        correlations = []
        for column in upper_tri.columns:
            for index in upper_tri.index:
                corr_value = upper_tri.loc[index, column]
                if not pd.isna(corr_value) and abs(corr_value) >= threshold:
                    correlations.append({
                        'Variable 1': index,
                        'Variable 2': column,
                        'Correlation': corr_value,
                        'Strength': 'Strong' if abs(corr_value) >= 0.7 else 'Moderate'
                    })
        
        if correlations:
            df = pd.DataFrame(correlations)
            return df.sort_values('Correlation', key=abs, ascending=False)
        else:
            return pd.DataFrame()
    
    def create_scatter_plot(self, x_column, y_column, color_column=None):
        """Create a scatter plot between two numeric columns"""
        if color_column:
            fig = px.scatter(
                self.data,
                x=x_column,
                y=y_column,
                color=color_column,
                title=f'{y_column} vs {x_column}',
                opacity=0.7
            )
        else:
            fig = px.scatter(
                self.data,
                x=x_column,
                y=y_column,
                title=f'{y_column} vs {x_column}',
                opacity=0.7
            )
        
        # Add trend line
        fig.add_trace(
            px.scatter(
                self.data,
                x=x_column,
                y=y_column,
                trendline="ols"
            ).data[1]
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_missing_data_plot(self):
        """Create a visualization of missing data"""
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
        
        if missing_data.empty:
            return None
        
        fig = px.bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            title='Missing Data by Column',
            labels={'x': 'Number of Missing Values', 'y': 'Columns'}
        )
        
        fig.update_layout(height=max(400, len(missing_data) * 30))
        
        return fig
    
    def get_data_summary(self):
        """Get a comprehensive summary of the dataset"""
        summary = {
            'shape': self.data.shape,
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns),
            'missing_values': self.data.isnull().sum().sum(),
            'duplicate_rows': self.data.duplicated().sum(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return summary
