import pandas as pd
import numpy as np
from datetime import datetime

class ReportGenerator:
    def __init__(self, data):
        self.data = data
        
    def generate_report(self, ml_results=None):
        """Generate a comprehensive analysis report"""
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("DATA SCIENCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset Overview
        report.extend(self._generate_dataset_overview())
        
        # Data Quality Assessment
        report.extend(self._generate_data_quality_section())
        
        # Statistical Summary
        report.extend(self._generate_statistical_summary())
        
        # Correlation Analysis
        report.extend(self._generate_correlation_analysis())
        
        # Missing Data Analysis
        report.extend(self._generate_missing_data_analysis())
        
        # Machine Learning Results
        if ml_results:
            report.extend(self._generate_ml_results_section(ml_results))
        
        # Recommendations
        report.extend(self._generate_recommendations())
        
        # Footer
        report.append("=" * 80)
        report.append("End of Report")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _generate_dataset_overview(self):
        """Generate dataset overview section"""
        section = []
        section.append("1. DATASET OVERVIEW")
        section.append("-" * 50)
        
        section.append(f"Dataset Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
        section.append(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        section.append("")
        
        # Column types
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        section.append(f"Numeric Columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}")
        if len(numeric_cols) > 10:
            section.append(f"  ... and {len(numeric_cols) - 10} more")
        
        section.append(f"Categorical Columns ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}")
        if len(categorical_cols) > 10:
            section.append(f"  ... and {len(categorical_cols) - 10} more")
        
        section.append("")
        return section
    
    def _generate_data_quality_section(self):
        """Generate data quality assessment"""
        section = []
        section.append("2. DATA QUALITY ASSESSMENT")
        section.append("-" * 50)
        
        # Missing values
        missing_count = self.data.isnull().sum().sum()
        missing_percentage = (missing_count / (self.data.shape[0] * self.data.shape[1])) * 100
        
        section.append(f"Total Missing Values: {missing_count} ({missing_percentage:.2f}%)")
        
        # Duplicate rows
        duplicates = self.data.duplicated().sum()
        section.append(f"Duplicate Rows: {duplicates}")
        
        # Columns with missing data
        missing_by_column = self.data.isnull().sum()
        missing_columns = missing_by_column[missing_by_column > 0]
        
        if not missing_columns.empty:
            section.append("\nColumns with Missing Data:")
            for col, count in missing_columns.items():
                percentage = (count / len(self.data)) * 100
                section.append(f"  {col}: {count} ({percentage:.1f}%)")
        else:
            section.append("\nNo missing data detected!")
        
        section.append("")
        return section
    
    def _generate_statistical_summary(self):
        """Generate statistical summary"""
        section = []
        section.append("3. STATISTICAL SUMMARY")
        section.append("-" * 50)
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            section.append("Numeric Variables Summary:")
            
            for col in numeric_data.columns[:10]:  # Limit to first 10 columns
                stats = numeric_data[col].describe()
                section.append(f"\n{col}:")
                section.append(f"  Mean: {stats['mean']:.3f}")
                section.append(f"  Std:  {stats['std']:.3f}")
                section.append(f"  Min:  {stats['min']:.3f}")
                section.append(f"  Max:  {stats['max']:.3f}")
                
                # Skewness and outliers
                skewness = numeric_data[col].skew()
                section.append(f"  Skewness: {skewness:.3f}")
                
                # Outliers using IQR method
                Q1 = stats['25%']
                Q3 = stats['75%']
                IQR = Q3 - Q1
                outliers = ((numeric_data[col] < (Q1 - 1.5 * IQR)) | 
                           (numeric_data[col] > (Q3 + 1.5 * IQR))).sum()
                section.append(f"  Outliers: {outliers}")
            
            if len(numeric_data.columns) > 10:
                section.append(f"\n... and {len(numeric_data.columns) - 10} more numeric variables")
        
        section.append("")
        return section
    
    def _generate_correlation_analysis(self):
        """Generate correlation analysis"""
        section = []
        section.append("4. CORRELATION ANALYSIS")
        section.append("-" * 50)
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            section.append("Insufficient numeric variables for correlation analysis.")
            section.append("")
            return section
        
        correlation_matrix = numeric_data.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if strong_correlations:
            section.append("Strong Correlations (|r| >= 0.5):")
            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            for corr in strong_correlations[:10]:  # Top 10
                section.append(f"  {corr['var1']} <-> {corr['var2']}: {corr['correlation']:.3f}")
        else:
            section.append("No strong correlations found (|r| >= 0.5)")
        
        section.append("")
        return section
    
    def _generate_missing_data_analysis(self):
        """Generate missing data analysis"""
        section = []
        section.append("5. MISSING DATA PATTERNS")
        section.append("-" * 50)
        
        missing_data = self.data.isnull().sum()
        total_missing = missing_data.sum()
        
        if total_missing == 0:
            section.append("No missing data patterns to analyze - dataset is complete!")
            section.append("")
            return section
        
        # Missing data by column
        missing_cols = missing_data[missing_data > 0].sort_values(ascending=False)
        
        section.append("Missing Data by Column:")
        for col, count in missing_cols.items():
            percentage = (count / len(self.data)) * 100
            section.append(f"  {col}: {count}/{len(self.data)} ({percentage:.1f}%)")
        
        # Missing data patterns
        missing_patterns = self.data.isnull().value_counts().head(5)
        section.append(f"\nTop Missing Data Patterns:")
        for pattern, count in missing_patterns.items():
            if isinstance(pattern, tuple):
                missing_cols_in_pattern = [col for col, is_missing in zip(self.data.columns, pattern) if is_missing]
                if missing_cols_in_pattern:
                    section.append(f"  Missing {', '.join(missing_cols_in_pattern)}: {count} rows")
                else:
                    section.append(f"  Complete rows: {count}")
        
        section.append("")
        return section
    
    def _generate_ml_results_section(self, ml_results):
        """Generate machine learning results section"""
        section = []
        section.append("6. MACHINE LEARNING RESULTS")
        section.append("-" * 50)
        
        section.append(f"Model Type: {ml_results['model_type']}")
        section.append(f"Problem Type: {ml_results['problem_type']}")
        section.append(f"Features Used: {', '.join(ml_results['feature_names'])}")
        section.append(f"Test Samples: {len(ml_results['y_test'])}")
        section.append("")
        
        # Performance metrics
        section.append("Performance Metrics:")
        if ml_results['problem_type'] == "Classification":
            section.append(f"  Test Accuracy: {ml_results['accuracy']:.3f}")
            section.append(f"  Training Accuracy: {ml_results['train_accuracy']:.3f}")
            
            # Overfitting check
            if ml_results['train_accuracy'] - ml_results['accuracy'] > 0.1:
                section.append("  ⚠️  Potential overfitting detected!")
        else:
            section.append(f"  R² Score: {ml_results['r2_score']:.3f}")
            section.append(f"  RMSE: {ml_results['rmse']:.3f}")
        
        section.append("")
        
        # Feature importance
        section.append("Feature Importance (Top 10):")
        feature_importance = list(zip(ml_results['feature_names'], ml_results['feature_importance']))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for feature, importance in feature_importance[:10]:
            section.append(f"  {feature}: {importance:.3f}")
        
        section.append("")
        return section
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis"""
        section = []
        section.append("7. RECOMMENDATIONS")
        section.append("-" * 50)
        
        recommendations = []
        
        # Data quality recommendations
        missing_count = self.data.isnull().sum().sum()
        if missing_count > 0:
            recommendations.append("• Address missing data through imputation or removal strategies")
        
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            recommendations.append(f"• Remove {duplicates} duplicate rows to improve data quality")
        
        # Feature engineering recommendations
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            for col in numeric_data.columns:
                skewness = abs(numeric_data[col].skew())
                if skewness > 2:
                    recommendations.append(f"• Consider log transformation for highly skewed variable: {col}")
        
        # Correlation recommendations
        if not numeric_data.empty and numeric_data.shape[1] > 1:
            correlation_matrix = numeric_data.corr()
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.9:
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
            
            if high_corr_pairs:
                recommendations.append("• Consider removing highly correlated features to reduce multicollinearity")
        
        # General recommendations
        if self.data.shape[0] < 1000:
            recommendations.append("• Consider collecting more data for better model performance")
        
        if len(recommendations) == 0:
            recommendations.append("• Data quality is good - ready for advanced modeling!")
        
        recommendations.append("• Consider feature engineering to create new meaningful variables")
        recommendations.append("• Validate results with domain experts")
        recommendations.append("• Monitor model performance over time")
        
        section.extend(recommendations)
        section.append("")
        
        return section
