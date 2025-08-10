import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MLTrainer:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_data(self, target_column, feature_columns):
        """Prepare data for machine learning"""
        # Create feature matrix and target vector
        X = self.data[feature_columns].copy()
        y = self.data[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        y = y.fillna(y.mode().iloc[0] if y.dtype == 'object' else y.mean())
        
        # Encode categorical features
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target if it's categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            self.label_encoders['target'] = le_target
        
        return X, y
    
    def train_model(self, target_column, feature_columns, problem_type, model_type):
        """Train a machine learning model"""
        # Prepare data
        X, y = self.prepare_data(target_column, feature_columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if problem_type == "Classification" else None
        )
        
        # Scale features for non-tree models
        if model_type in ["Logistic Regression", "Linear Regression"]:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Initialize model
        if problem_type == "Classification":
            if model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Logistic Regression
                model = LogisticRegression(random_state=42, max_iter=1000)
        else:  # Regression
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:  # Linear Regression
                model = LinearRegression()
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_predictions = model.predict(X_train_scaled)
        test_predictions = model.predict(X_test_scaled)
        
        # Calculate metrics
        results = {
            'model': model,
            'feature_names': feature_columns,
            'y_test': y_test,
            'predictions': test_predictions,
            'model_type': model_type,
            'problem_type': problem_type
        }
        
        if problem_type == "Classification":
            results['accuracy'] = accuracy_score(y_test, test_predictions)
            results['train_accuracy'] = accuracy_score(y_train, train_predictions)
            
            # Handle label encoding for classification report
            if 'target' in self.label_encoders:
                target_names = self.label_encoders['target'].classes_
                y_test_labels = self.label_encoders['target'].inverse_transform(y_test)
                pred_labels = self.label_encoders['target'].inverse_transform(test_predictions)
                results['classification_report'] = classification_report(
                    y_test_labels, pred_labels, target_names=target_names
                )
            else:
                results['classification_report'] = classification_report(y_test, test_predictions)
        
        else:  # Regression
            results['r2_score'] = r2_score(y_test, test_predictions)
            results['rmse'] = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results['feature_importance'] = np.abs(model.coef_)
        else:
            results['feature_importance'] = np.zeros(len(feature_columns))
        
        return results
    
    def get_model_summary(self, results):
        """Get a summary of model performance"""
        summary = {
            'model_type': results['model_type'],
            'problem_type': results['problem_type'],
            'num_features': len(results['feature_names']),
            'test_samples': len(results['y_test'])
        }
        
        if results['problem_type'] == "Classification":
            summary['accuracy'] = results['accuracy']
            summary['train_accuracy'] = results['train_accuracy']
        else:
            summary['r2_score'] = results['r2_score']
            summary['rmse'] = results['rmse']
        
        return summary
