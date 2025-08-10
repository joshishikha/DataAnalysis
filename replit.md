# Interactive Data Science Platform

## Overview

This is a Streamlit-based interactive data science platform that provides automated data analysis, visualization, and machine learning capabilities. Users can upload CSV files and receive comprehensive insights through an intuitive web interface. The platform is built with Python and leverages popular data science libraries to deliver end-to-end analytics workflows.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **Layout**: Wide layout with expandable sidebar for navigation
- **Components**: File upload, interactive visualizations, and report generation
- **Styling**: Light theme with customizable configuration

### Backend Architecture
- **Language**: Python 3.11
- **Structure**: Modular design with utility classes for different functionalities
- **Data Processing**: Pandas for data manipulation and NumPy for numerical operations
- **Machine Learning**: Scikit-learn for model training and evaluation
- **Visualizations**: Plotly for interactive charts and Matplotlib/Seaborn for statistical plots

### Key Components
1. **Main Application** (`app.py`): Entry point with Streamlit UI and session state management
2. **Data Analyzer** (`utils/data_analyzer.py`): Handles data exploration and visualization generation
3. **ML Trainer** (`utils/ml_trainer.py`): Manages machine learning workflows including preprocessing and model training
4. **Report Generator** (`utils/report_generator.py`): Creates comprehensive analysis reports

## Data Flow

1. **Data Upload**: Users upload CSV files through the Streamlit file uploader
2. **Data Processing**: Raw data is loaded into pandas DataFrames and validated
3. **Analysis Pipeline**: 
   - Automated data profiling and quality assessment
   - Statistical analysis and correlation discovery
   - Interactive visualization generation
4. **Machine Learning**: 
   - Feature engineering and preprocessing
   - Model training (classification/regression)
   - Performance evaluation and metrics calculation
5. **Report Generation**: Comprehensive text-based reports summarizing findings
6. **Results Display**: Interactive dashboard presenting all insights

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for data science
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive plotting and visualization
- **Scikit-learn**: Machine learning algorithms and utilities
- **Matplotlib/Seaborn**: Statistical data visualization

### System Dependencies (via Nix)
- Cairo, FFmpeg, FreeType: Graphics and media processing
- GTK3, GObject Introspection: GUI toolkit support
- Ghostscript: PostScript processing
- Qhull: Computational geometry
- TCL/TK: Additional GUI framework support

## Deployment Strategy

### Environment Management
- **Nix Configuration**: Stable channel 24_05 for reproducible environments
- **Python Version**: Fixed at Python 3.11 for consistency
- **Package Management**: UV lock file ensures deterministic dependency resolution

### Deployment Configuration
- **Target**: Autoscale deployment for automatic scaling based on demand
- **Port**: Application runs on port 5000
- **Execution**: Streamlit server with headless configuration for production

### Workflow Management
- **Run Button**: Integrated project workflow execution
- **Parallel Execution**: Support for concurrent task execution
- **Port Monitoring**: Automatic health checks on application port

## Key Architectural Decisions

### Technology Stack Selection
- **Problem**: Need for rapid development of interactive data science tools
- **Solution**: Streamlit framework with Python data science ecosystem
- **Rationale**: Streamlit enables quick prototyping while maintaining professional UI standards

### Modular Architecture
- **Problem**: Managing complex data science workflows in a single file
- **Solution**: Separated concerns into utility modules (analyzer, trainer, reporter)
- **Benefits**: Improved maintainability, testability, and code reuse

### Session State Management
- **Problem**: Maintaining data and results across user interactions
- **Solution**: Streamlit session state for persistent data storage
- **Rationale**: Prevents data loss during UI interactions and improves user experience

### Visualization Strategy
- **Problem**: Need for both interactive and static visualizations
- **Solution**: Plotly for interactive charts, Matplotlib/Seaborn for statistical plots
- **Benefits**: Comprehensive visualization capabilities covering different use cases

## Changelog

```
Changelog:
- June 24, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```