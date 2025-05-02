# DataSightPro

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-FF4B4B.svg)](https://streamlit.io/)
[![pandas](https://img.shields.io/badge/pandas-2.2.3-150458.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-6.0.1-3F4F75.svg)](https://plotly.com/)

<div align="center">
  <h3>A Comprehensive Data Analysis and ML Preparation Tool</h3>
  <p>Interactive exploration, visualization, and preparation of datasets for data science projects</p>
</div>

---

## 📋 Overview

DataSightPro is a powerful web-based application that provides an intuitive interface for data scientists, analysts, and ML engineers to explore, visualize, analyze, and preprocess datasets. Built with Streamlit, it offers a comprehensive set of tools for quick data profiling, interactive visualization, statistical analysis, and data preparation for machine learning.

## ✨ Features

### Data Exploration
- **Dataset Overview**: Quick profile with data types, missing values summary, and basic statistics
- **Interactive Data Browsing**: Easily navigate and filter your dataset
- **Automated Data Type Detection**: Intelligent identification of categorical and numerical features
- **Data Quality Assessment**: Comprehensive quality checks including duplicates, missing values, and class imbalance

### Statistical Analysis
- **Summary Statistics**: Distribution metrics, quartile information, skewness, and kurtosis
- **Correlation Analysis**: Interactive correlation matrices and significance testing
- **Distribution Analysis**: Histograms, box plots, and violin plots with automated outlier detection
- **Hypothesis Testing**: ANOVA for numerical variables, Chi-square tests for categorical relationships
- **Feature Importance**: Identifying influential features for predictive modeling

### Data Visualization
- **Distribution Visualizations**: Histograms, box plots, violin plots
- **Relationship Visualizations**: Scatter plots, correlation heatmaps, pair plots
- **Categorical Visualizations**: Bar charts, pie charts, sunburst charts
- **Time Series Visualizations**: Line charts
- **Multi-dimensional Visualizations**: 3D scatter plots
- **Smart Visualization Suggestions**: Context-aware recommendations based on data characteristics

### Data Preprocessing
- **Missing Value Handling**: Multiple imputation strategies (mean/mode, median, constant)
- **Categorical Encoding**: One-hot encoding, label encoding
- **Feature Scaling**: Min-max normalization, z-score standardization, robust scaling
- **Outlier Removal**: IQR method, z-score method
- **Distribution Transformation**: Log, square root, Box-Cox, Yeo-Johnson transformations
- **Feature Engineering**: Polynomial feature generation
- **Pattern Discovery**: Clustering and segmentation capabilities

### Export Capabilities
- **Multiple Formats**: Export processed data as CSV, Excel, or JSON
- **Processing History**: Track transformations applied to your dataset
- **Seamless ML Integration**: Prepare data for direct use in machine learning pipelines

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher

### Setup
1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/DataSightPro.git
   cd DataSightPro
   ```

2. Create and activate a virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -e .
   ```

## 🖥️ Usage

1. Run the application
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload your dataset (CSV, Excel, or JSON) using the file uploader in the sidebar

4. Navigate through different analysis sections:
   - Dataset Overview
   - Statistical Analysis
   - Visualization
   - Data Preprocessing
   - Export Options

## 📊 Example Data Flow

1. **Upload Data**: Load your dataset through the file uploader
   - The system supports CSV, Excel, and JSON formats
   - For quick testing, use the sample dataset included in the `SampleDataset` directory

2. **Explore**: Get a quick overview and understand data structure
   - View basic statistics, data types, and missing value summaries
   - Browse through the first few rows of your dataset
   - Identify potential data quality issues automatically

3. **Analyze**: Run statistical tests and visualize distributions
   - Generate correlation matrices to identify relationships
   - Create distribution plots to understand data patterns
   - Perform hypothesis tests to validate your assumptions

4. **Preprocess**: Handle missing values, encode categories, normalize features
   - Select appropriate strategies for handling missing data
   - Convert categorical variables to numerical formats
   - Scale numerical features for machine learning algorithms
   - Transform skewed distributions for better model performance

5. **Export**: Save the processed dataset for your machine learning pipeline
   - Download in CSV, Excel, or JSON format
   - Track all applied transformations for reproducibility
   - Use directly in your preferred ML framework

## 💡 Implementation Architecture

DataSightPro is built on a modular architecture designed for extensibility and maintainability:

### Frontend Layer (Streamlit)
- **User Interface**: Interactive web components built with Streamlit
- **Session Management**: Persistent state handling across interactions
- **Modern UI Design**: Custom CSS styling for enhanced user experience

### Analysis Layer
- **Data Processing Pipeline**: Modular data transformation flow
- **Statistical Engine**: Advanced statistical computations
- **Visualization Framework**: Interactive Plotly-based graphics

### Data Layer
- **Input Handling**: Support for multiple file formats
- **Data Transformation**: In-memory data processing
- **Output Generation**: Exporting processed data in various formats

## 🗂️ Project Structure

```
DataSightPro/
├── app.py                 # Main application entry point
├── analysis.py            # Statistical analysis functions
├── preprocessing.py       # Data preprocessing utilities
├── utils.py               # Helper functions
├── visualization.py       # Plotting and visualization functions
├── pyproject.toml         # Project dependencies and metadata
├── assets/                # Static assets
├── docs/                  # Documentation
│   ├── README.md          # Documentation overview
│   ├── GettingStarted.md  # Quick start guide
│   ├── UserGuide.md       # Comprehensive user instructions
│   ├── QuickReference.md  # Quick reference tables
│   └── TechnicalReport.md # Implementation details
└── SampleDataset/         # Example data for demonstration
    └── customer_data.csv  # Sample customer dataset
```

## 🔧 Modules

### app.py
The main application interface built with Streamlit. It handles the UI, file uploading, and coordinates the analysis workflow. The module implements:
- Modern UI with custom styling
- Session state management for persistent data
- Tabbed interface for logical workflow organization
- Responsive layout with multi-column design

### analysis.py
Statistical analysis module providing correlation analysis, distribution analysis, ANOVA tests, chi-square tests, feature importance detection, and clustering. Key features include:
- Relationship detection between variables
- Hypothesis testing for statistical significance
- Distribution shape analysis with normality testing
- Feature importance ranking
- K-means clustering with silhouette score evaluation

### preprocessing.py
Data preparation module with functions for handling missing values, encoding categorical variables, normalizing data, removing outliers, transforming skewed data, and feature engineering. Implementations include:
- Type-specific missing value imputation
- Multiple encoding strategies for categorical data
- Three normalization methods (min-max, z-score, robust)
- Statistical and distance-based outlier detection
- Advanced distribution transformations
- Polynomial feature generation

### visualization.py
Visualization module with interactive Plotly charts including histograms, box plots, scatter plots, correlation heatmaps, and more. Capabilities include:
- Distribution visualizations with statistical annotations
- Relationship visualizations with correlation coefficients
- Categorical data visualizations with frequency analysis
- Time series visualizations
- Multi-dimensional visualizations (3D plots)

### utils.py
Utility functions for data exploration, type detection, and common operations, featuring:
- Intelligent data type detection and classification
- Comprehensive summary statistics generation
- Missing value analysis
- Data quality assessment
- Smart visualization recommendations

## 🔬 Advanced Usage Scenarios

### Exploratory Data Analysis Workflow
1. Upload your dataset
2. Review the automatically generated data quality report
3. Examine distributions of key variables
4. Analyze relationships between variables using correlation analysis
5. Test statistical hypotheses based on data types
6. Generate visualizations based on the discovered patterns

### Machine Learning Preparation Workflow
1. Upload your dataset
2. Handle missing values using appropriate imputation strategies
3. Encode categorical variables based on your model requirements
4. Apply feature scaling for algorithm-specific needs
5. Transform skewed distributions for improved model performance
6. Remove or handle outliers based on your modeling strategy
7. Export the processed dataset for model training

### Feature Engineering Workflow
1. Upload your dataset
2. Identify important features using statistical methods
3. Create polynomial features to capture non-linear relationships
4. Remove or transform outliers
5. Apply dimensionality reduction if needed
6. Evaluate the impact of engineered features
7. Export the engineered dataset

## 💻 Technologies

- **Streamlit**: Interactive web interface
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **scikit-learn**: Machine learning tools
- **SciPy**: Scientific computing
- **statsmodels**: Statistical models

## 📝 Documentation

DataSightPro includes comprehensive documentation to help users and developers:

### User Documentation
- **README.md**: Overview, installation, and basic usage instructions
- **Sample Dataset**: Example data for testing and learning the application

### Technical Documentation
- **[Technical Report](docs/TechnicalReport.md)**: Detailed architecture, implementation details, and design decisions
- **Module Docstrings**: Comprehensive function and class documentation
- **Code Comments**: Explanatory comments for complex algorithms and processes

The Technical Report provides in-depth information about:
- System architecture and component interactions
- Data processing pipeline details
- Statistical analysis methodologies
- Visualization system design
- Preprocessing techniques implementation
- UI/UX implementation
- Performance considerations
- Extensibility options
- Testing strategy
- Deployment guidelines

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for their amazing framework
- [Plotly](https://plotly.com/) for interactive visualization capabilities
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [pandas](https://pandas.pydata.org/) for powerful data manipulation tools
- [NumPy](https://numpy.org/) for numerical computing infrastructure