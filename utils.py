"""
Data Analysis Utility Functions
==============================

This module provides a collection of utility functions for data analysis, exploration,
and quality assessment. These functions help with:

1. Data type detection and classification
2. Statistical summaries and analysis
3. Missing value assessment
4. Data quality checks
5. Outlier detection
6. Visualization recommendations

All functions accept pandas DataFrames as input and return various structured outputs
to facilitate data analysis workflows.
"""

import pandas as pd
import numpy as np

# ----------------------
# DATA TYPE ANALYSIS
# ----------------------

def detect_data_types(df):
    """
    Detect and classify data types of each column in the DataFrame.
    
    This function analyzes each column and categorizes it into one of the following types:
    - Binary: Numeric with ≤10 unique values, all in {0, 1}
    - Discrete: Numeric with ≤10 unique values
    - Continuous: Numeric with >10 unique values
    - DateTime: Datetime objects
    - Categorical: Non-numeric with ≤10 unique values
    - Text: Non-numeric with >10 unique values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame whose columns will be analyzed
        
    Returns:
    --------
    dict
        Dictionary mapping column names to their detected data types
    """
    data_types = {}
    
    for column in df.columns:
        # Numeric data type analysis
        if pd.api.types.is_numeric_dtype(df[column]):
            # Binary data (values limited to 0 and 1)
            if df[column].nunique() <= 10 and set(df[column].dropna().unique()).issubset({0, 1}):
                data_types[column] = 'Binary'
            # Discrete numeric data (limited unique values)
            elif df[column].nunique() <= 10:
                data_types[column] = 'Discrete'
            # Continuous numeric data (many unique values)
            else:
                data_types[column] = 'Continuous'
        
        # Datetime data type
        elif pd.api.types.is_datetime64_dtype(df[column]):
            data_types[column] = 'DateTime'
        
        # Categorical and text data types
        else:
            # Categorical (limited unique values)
            if df[column].nunique() <= 10:
                data_types[column] = 'Categorical'
            # Text (many unique values)
            else:
                data_types[column] = 'Text'
    
    return data_types

# ----------------------
# STATISTICAL ANALYSIS
# ----------------------

def get_summary_statistics(df):
    """
    Generate comprehensive summary statistics for all columns in the DataFrame.
    
    This function creates separate statistical summaries for numerical and categorical
    columns, then combines them if both types exist.
    
    For numerical columns: count, mean, std, min, 25%, 50%, 75%, max, skew, kurtosis, median
    For categorical columns: unique count, most frequent value, frequency, frequency percentage
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing summary statistics with column names as index
        If both numerical and categorical columns exist, includes a 'type' column
        Returns empty DataFrame if no valid columns are found
    """
    # Process numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        # Get standard descriptive statistics and transpose
        numerical_stats = df[numerical_cols].describe().T
        
        # Add distribution shape statistics
        numerical_stats['skew'] = df[numerical_cols].skew()
        numerical_stats['kurtosis'] = df[numerical_cols].kurtosis()
        numerical_stats['median'] = df[numerical_cols].median()
        numerical_stats = numerical_stats.round(4)  # Round for readability
    else:
        numerical_stats = pd.DataFrame()
    
    # Process categorical columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    if len(categorical_cols) > 0:
        cat_stats = pd.DataFrame()
        for col in categorical_cols:
            try:
                # Calculate unique value statistics
                value_counts = df[col].value_counts()
                cat_stats.loc[col, 'unique'] = df[col].nunique()
                cat_stats.loc[col, 'top'] = value_counts.index[0]  # Most frequent value
                cat_stats.loc[col, 'freq'] = value_counts.values[0]  # Count of most frequent
                cat_stats.loc[col, 'freq_pct'] = (value_counts.values[0] / len(df)) * 100  # Percentage
            except:
                # Skip empty columns or columns with all NaN values
                pass
    else:
        cat_stats = pd.DataFrame()
    
    # Combine numerical and categorical statistics if both exist
    if not numerical_stats.empty and not cat_stats.empty:
        # Reset indices to add column names as a regular column
        numerical_stats = numerical_stats.reset_index().rename(columns={'index': 'column'})
        cat_stats = cat_stats.reset_index().rename(columns={'index': 'column'})
        
        # Add column type identifier
        numerical_stats['type'] = 'numerical'
        cat_stats['type'] = 'categorical'
        
        # Combine and set column name as index
        combined_stats = pd.concat([numerical_stats, cat_stats], ignore_index=True)
        return combined_stats.set_index('column')
    
    # Return appropriate statistics if only one type exists
    elif not numerical_stats.empty:
        return numerical_stats
    elif not cat_stats.empty:
        return cat_stats
    else:
        return pd.DataFrame()

# ----------------------
# DATA QUALITY ASSESSMENT
# ----------------------

def get_missing_values_summary(df):
    """
    Calculate the percentage of missing values for each column in the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze for missing values
        
    Returns:
    --------
    dict
        Dictionary mapping column names to the percentage of missing values
        Percentages range from 0.0 (no missing values) to 100.0 (all values missing)
    """
    missing_values = {}
    for column in df.columns:
        # Count missing values
        missing_count = df[column].isna().sum()
        # Calculate percentage
        missing_percentage = (missing_count / len(df)) * 100
        missing_values[column] = missing_percentage
    
    return missing_values

def check_for_outliers(df, columns=None, method='iqr'):
    """
    Detect outliers in specified DataFrame columns using statistical methods.
    
    This function supports two outlier detection methods:
    1. IQR (Interquartile Range): Values outside Q1-1.5*IQR and Q3+1.5*IQR
    2. Z-score: Values more than 3 standard deviations from the mean
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    columns : list, optional
        List of column names to check for outliers
        If None, all numerical columns are checked
    method : str, optional
        Outlier detection method to use: 'iqr' or 'zscore'
        Default is 'iqr'
        
    Returns:
    --------
    dict
        Dictionary with column names as keys and outlier analysis results as values.
        Each value contains method-specific statistics and outlier counts/percentages.
    """
    if columns is None:
        # Default to all numerical columns if not specified
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    outlier_info = {}
    
    for col in columns:
        if method.lower() == 'iqr':
            # IQR method - detect values beyond 1.5*IQR from Q1 and Q3
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate boundaries for outlier detection
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers (values outside the boundaries)
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            # Store outlier information and statistics
            outlier_info[col] = {
                'method': 'IQR',
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100
            }
        
        elif method.lower() == 'zscore':
            # Z-score method - detect values beyond 3 standard deviations from mean
            mean = df[col].mean()
            std = df[col].std()
            
            # Calculate absolute Z-scores for each value
            z_scores = np.abs((df[col] - mean) / std)
            # Find outliers (Z-score > 3)
            outliers = df[z_scores > 3][col]
            
            # Store outlier information and statistics
            outlier_info[col] = {
                'method': 'Z-Score',
                'mean': mean,
                'std': std,
                'threshold': 3,  # Standard threshold of 3 standard deviations
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100
            }
    
    return outlier_info

def check_data_quality(df):
    """
    Perform a comprehensive data quality assessment on the DataFrame.
    
    This function analyzes:
    1. Overall statistics (rows, columns)
    2. Duplicate row detection
    3. Missing value analysis
    4. Data type classification
    5. Class imbalance in categorical variables
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
        
    Returns:
    --------
    dict
        Dictionary containing various quality metrics and detailed analyses
        including data types, missing values, and class imbalance information
    """
    # Generate basic quality metrics
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'duplicated_rows': {
            'count': df.duplicated().sum(),
            'percentage': (df.duplicated().sum() / len(df)) * 100
        },
        'missing_values': get_missing_values_summary(df),
        'data_types': detect_data_types(df)
    }
    
    # Check for class imbalance in categorical variables
    categorical_cols = [col for col, dtype in quality_report['data_types'].items() 
                       if dtype in ['Categorical', 'Binary']]
    
    if categorical_cols:
        imbalance_report = {}
        for col in categorical_cols:
            # Calculate value distribution percentages
            value_counts = df[col].value_counts(normalize=True) * 100
            
            # Detect imbalance based on thresholds
            # (most frequent class > 80% or least frequent class < 10%)
            if value_counts.max() > 80 or value_counts.min() < 10:
                imbalance_report[col] = {
                    'most_common': value_counts.index[0],
                    'most_common_pct': value_counts.iloc[0],
                    'least_common': value_counts.index[-1],
                    'least_common_pct': value_counts.iloc[-1],
                    'is_imbalanced': True
                }
            else:
                imbalance_report[col] = {
                    'is_imbalanced': False
                }
        
        quality_report['class_imbalance'] = imbalance_report
    
    return quality_report

# ----------------------
# VISUALIZATION RECOMMENDATIONS
# ----------------------

def suggest_visualizations(df):
    """
    Recommend appropriate visualizations based on DataFrame composition and data types.
    
    This function suggests visualizations in multiple categories:
    1. Single variable visualizations (for numerical, categorical, datetime columns)
    2. Two-variable visualizations (relationships between columns)
    3. Grouped visualizations (numerical data grouped by categories)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze for visualization suggestions
        
    Returns:
    --------
    dict
        Hierarchical dictionary of visualization suggestions organized by:
        - Data type category (numerical, categorical, time_series)
        - Visualization type (e.g., histogram, bar_chart, scatter_plot)
        - List of suitable columns or column pairs for each visualization
    """
    # Detect data types to inform visualization choices
    data_types = detect_data_types(df)
    suggestions = {}
    
    # Categorize columns by their detected types
    numerical_cols = [col for col, type_info in data_types.items() 
                      if type_info in ['Continuous', 'Discrete']]
    categorical_cols = [col for col, type_info in data_types.items() 
                        if type_info in ['Categorical', 'Binary']]
    datetime_cols = [col for col, type_info in data_types.items() 
                     if type_info == 'DateTime']
    
    # --- Single variable visualizations ---
    
    # Numerical data visualizations
    if numerical_cols:
        suggestions['numerical'] = {
            'histogram': numerical_cols,  # Distribution of values
            'box_plot': numerical_cols,   # Quartiles and outliers
            # Density plots more suitable for continuous variables
            'density_plot': [col for col in numerical_cols if data_types[col] == 'Continuous'],
        }
    
    # Categorical data visualizations
    if categorical_cols:
        suggestions['categorical'] = {
            'bar_chart': categorical_cols,  # Frequency of categories
            # Pie charts only practical with limited categories (<=5)
            'pie_chart': [col for col in categorical_cols if df[col].nunique() <= 5],
        }
    
    # Time-series data visualizations
    if datetime_cols:
        suggestions['time_series'] = {
            'line_chart': datetime_cols,  # Trends over time
        }
    
    # --- Multi-variable visualizations ---
    
    # Relationships between numerical variables
    if len(numerical_cols) >= 2:
        suggestions['bivariate'] = {
            # Generate all possible pairs of numerical columns
            'scatter_plot': [(x, y) for i, x in enumerate(numerical_cols) 
                            for y in numerical_cols[i+1:]],
            'heatmap': ['correlation_matrix']  # Correlation matrix visualization
        }
    
    # Numerical data grouped by categories
    if numerical_cols and categorical_cols:
        # Only include categories with reasonable number of groups (<=10)
        filtered_cat_cols = [cat for cat in categorical_cols if df[cat].nunique() <= 10]
        
        if filtered_cat_cols:
            suggestions['grouped'] = {
                # Numerical distribution by category
                'grouped_box_plot': [(num, cat) for num in numerical_cols 
                                    for cat in filtered_cat_cols],
                # Mean/sum of numerical by category                                
                'grouped_bar_chart': [(num, cat) for num in numerical_cols 
                                     for cat in filtered_cat_cols],
            }
    
    return suggestions
