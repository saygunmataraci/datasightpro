"""
Data Analysis Module for DataSightPro
-------------------------------------
This module provides statistical analysis and machine learning functions for exploring datasets.
It includes tools for correlation analysis, distribution analysis, statistical testing, 
feature importance evaluation, and clustering.

The functions in this module expect pandas DataFrames as input and return structured results
that can be easily visualized or further processed.
"""

import pandas as pd
import numpy as np
from scipy import stats

# ============================================================================
# CORRELATION AND RELATIONSHIP ANALYSIS
# ============================================================================

def generate_correlation_matrix(df):
    """
    Generate a correlation matrix for numerical columns in the dataframe.
    
    The correlation coefficient ranges from -1 to 1, where:
    * 1 indicates perfect positive correlation
    * 0 indicates no correlation
    * -1 indicates perfect negative correlation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing numerical columns to analyze
        
    Returns:
    --------
    pandas.DataFrame or None
        A correlation matrix as a pandas DataFrame if at least 2 numeric columns exist,
        None otherwise
    """
    # Extract only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=['number'])
    
    # Require at least 2 numeric columns to compute meaningful correlations
    if numeric_df.shape[1] < 2:
        return None
    
    # Calculate the Pearson correlation matrix
    correlation_matrix = numeric_df.corr()
    
    return correlation_matrix


def perform_anova(df, numeric_col, categorical_col):
    """
    Perform one-way ANOVA test to determine if means of a numeric variable 
    differ significantly between categories.
    
    ANOVA (Analysis of Variance) tests the null hypothesis that samples from
    different groups are drawn from the same population.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    numeric_col : str
        Name of the numeric column to analyze
    categorical_col : str
        Name of the categorical column defining the groups
        
    Returns:
    --------
    dict or None
        Dictionary containing:
        - 'f_statistic': F-statistic of the ANOVA test
        - 'p_value': p-value of the test
        - 'significant': Boolean indicating if the difference is statistically 
                         significant (p < 0.05)
        Returns None if there are fewer than 2 valid groups
    """
    # Create separate groups of data for each category
    groups = []
    for category in df[categorical_col].unique():
        # Extract numeric data for this category, removing NaN values
        category_data = df[df[categorical_col] == category][numeric_col].dropna()
        if len(category_data) > 0:
            groups.append(category_data)
    
    # ANOVA requires at least 2 groups to compare
    if len(groups) >= 2:
        # Perform one-way ANOVA using scipy's f_oneway function
        f_stat, p_value = stats.f_oneway(*groups)
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05  # Conventional significance threshold
        }
    else:
        return None


def perform_chi_square(df, cat_col1, cat_col2):
    """
    Perform Chi-Square test of independence between two categorical variables.
    
    The Chi-Square test evaluates whether there is a significant relationship
    between two categorical variables based on their observed frequencies.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    cat_col1 : str
        Name of the first categorical column
    cat_col2 : str
        Name of the second categorical column
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'chi2': Chi-square statistic
        - 'p_value': p-value of the test
        - 'dof': Degrees of freedom
        - 'significant': Boolean indicating if the relationship is statistically
                         significant (p < 0.05)
    """
    # Create a contingency table (cross-tabulation) of the two categorical variables
    contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
    
    # Perform Chi-Square test of independence
    # This tests whether the frequency distribution of one categorical variable
    # depends on the other categorical variable
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    return {
        'chi2': chi2,
        'p_value': p,
        'dof': dof,
        'significant': p < 0.05  # Conventional significance threshold
    }


# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_distributions(df):
    """
    Analyze the statistical distributions of numerical columns in the dataframe.
    
    For each numerical column, this function calculates:
    - Basic statistics (mean, median, standard deviation, etc.)
    - Distribution shape metrics (skewness, kurtosis)
    - Normality test results using the Shapiro-Wilk test
    - Quantile information for box plots
    - Outlier detection using the IQR method
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing numerical columns to analyze
        
    Returns:
    --------
    dict
        A nested dictionary with column names as keys and distribution metrics as values
    """
    distribution_info = {}
    
    # Only analyze numeric columns since distribution analysis requires quantitative data
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        # Skip columns with zero variance (constant values)
        if df[col].std() == 0:
            continue
        
        # Remove missing values for analysis
        data = df[col].dropna()
        
        # Require sufficient data points for meaningful statistical analysis
        if len(data) < 10:
            continue
        
        # Calculate basic descriptive statistics
        distribution_info[col] = {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'skewness': data.skew(),  # Measures asymmetry of distribution
            'kurtosis': data.kurtosis(),  # Measures "tailedness" of distribution
            'is_normal': False  # Default value, will be updated after testing
        }
        
        # Test for normality using Shapiro-Wilk test
        # Note: Shapiro-Wilk has limitations for large sample sizes (>5000)
        sample_size = min(5000, len(data))
        # Random sampling if needed, with fixed random state for reproducibility
        sample = data.sample(sample_size, random_state=42) if len(data) > sample_size else data
        
        try:
            # Perform Shapiro-Wilk test for normality
            # Null hypothesis: data comes from a normal distribution
            shapiro_test = stats.shapiro(sample)
            distribution_info[col]['shapiro_test_p_value'] = shapiro_test.pvalue
            # If p-value > 0.05, we don't reject the null hypothesis (data is normal)
            distribution_info[col]['is_normal'] = shapiro_test.pvalue > 0.05
        except:
            # Normality test might fail in some edge cases
            distribution_info[col]['shapiro_test_p_value'] = None
        
        # Calculate quantiles for box plot visualization
        distribution_info[col]['quantiles'] = {
            '25%': data.quantile(0.25),  # First quartile (Q1)
            '50%': data.quantile(0.5),   # Median (Q2)
            '75%': data.quantile(0.75)   # Third quartile (Q3)
        }
        
        # Identify outliers using the Interquartile Range (IQR) method
        # Outliers are defined as data points below Q1-1.5*IQR or above Q3+1.5*IQR
        Q1 = distribution_info[col]['quantiles']['25%']
        Q3 = distribution_info[col]['quantiles']['75%']
        IQR = Q3 - Q1
        
        # Calculate outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify data points outside the boundaries
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Store outlier information
        distribution_info[col]['outliers'] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100 if len(data) > 0 else 0,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return distribution_info


# ============================================================================
# FEATURE IMPORTANCE AND PREDICTIVE ANALYSIS
# ============================================================================

def detect_feature_importance(df, target_col):
    """
    Detect feature importance relative to a target column using appropriate
    statistical methods based on data types.
    
    This function handles both:
    - Numeric target: uses correlation and ANOVA
    - Categorical target: uses chi-square and correlation with dummy variables
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    target_col : str
        Name of the target column to measure importance against
        
    Returns:
    --------
    dict
        Dictionary mapping feature names to their importance scores,
        sorted in descending order of importance
    """
    importance = {}
    
    # Decide analysis approach based on target column data type
    if pd.api.types.is_numeric_dtype(df[target_col]):
        # === NUMERIC TARGET ANALYSIS ===
        
        # Method 1: Correlation analysis for numeric features
        # Higher absolute correlation indicates stronger relationship
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != target_col:
                # Calculate Pearson correlation coefficient
                correlation = df[[target_col, col]].corr().iloc[0, 1]
                # Use absolute correlation as importance (direction doesn't matter for importance)
                importance[col] = abs(correlation)
        
        # Method 2: ANOVA for categorical features
        # Higher F-statistic indicates the categorical feature has a stronger effect
        # on the distribution of the target variable
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        for col in categorical_cols:
            anova_result = perform_anova(df, target_col, col)
            if anova_result and not pd.isna(anova_result['f_statistic']):
                importance[col] = anova_result['f_statistic']
    else:
        # === CATEGORICAL TARGET ANALYSIS ===
        
        # Method 1: Chi-square test for categorical features
        # Higher chi-square value indicates stronger association
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        for col in categorical_cols:
            if col != target_col:
                chi2_result = perform_chi_square(df, target_col, col)
                if chi2_result:
                    importance[col] = chi2_result['chi2']
        
        # Method 2: Correlation with dummy variables for numeric features
        # Limit to reasonable number of categories to avoid excessive computation
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(df[target_col].unique()) <= 10:
            # Convert categorical target to dummy variables (one-hot encoding)
            target_dummies = pd.get_dummies(df[target_col], prefix=target_col)
            
            for col in numeric_cols:
                # For each numeric feature, find its maximum correlation with any target category
                max_correlation = 0
                for dummy_col in target_dummies.columns:
                    correlation = abs(df[col].corr(target_dummies[dummy_col]))
                    max_correlation = max(max_correlation, correlation)
                
                importance[col] = max_correlation
    
    # Sort features by importance score in descending order
    importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
    
    return importance


# ============================================================================
# CLUSTERING AND PATTERN DISCOVERY
# ============================================================================

def identify_clusters(df, numeric_cols, n_clusters=3):
    """
    Identify clusters in the data using KMeans clustering algorithm.
    
    This function:
    1. Preprocesses the data (handles missing values, standardizes features)
    2. Applies KMeans clustering
    3. Transforms cluster centers back to original scale
    4. Evaluates clustering quality using silhouette score when possible
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    numeric_cols : list
        List of numeric column names to use for clustering
    n_clusters : int, default=3
        Number of clusters to identify
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'labels': Array of cluster labels for each data point
        - 'centers': DataFrame of cluster centers in original feature scale
        - 'silhouette_score': Measure of clustering quality (higher is better)
    """
    # Import clustering-specific libraries here to avoid unnecessary imports
    # if this function is not used
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Step 1: Data preparation
    # Select only the specified numerical columns for clustering
    X = df[numeric_cols].copy()
    
    # Handle missing values by replacing with column means
    # This is a simple imputation strategy that preserves the feature distribution
    X = X.fillna(X.mean())
    
    # Step 2: Feature standardization
    # Standardize features to have mean=0 and variance=1
    # This is crucial for KMeans which is sensitive to feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 3: Apply KMeans clustering
    # Using fixed random_state for reproducibility
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Step 4: Transform cluster centers back to original scale for interpretability
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create a DataFrame with cluster centers for easy access and visualization
    centers_df = pd.DataFrame(cluster_centers, columns=numeric_cols)
    centers_df.index.name = 'Cluster'
    
    # Step 5: Evaluate clustering quality using silhouette score if possible
    # Silhouette score ranges from -1 to 1:
    # - Values near 1 indicate well-separated clusters
    # - Values near 0 indicate overlapping clusters
    # - Values near -1 indicate misassigned data points
    silhouette_avg = None
    if n_clusters > 1:  # Silhouette score requires at least 2 clusters
        from sklearn.metrics import silhouette_score
        try:
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        except:
            # Silhouette score may fail in edge cases (e.g., single-sample clusters)
            pass
    
    # Return comprehensive clustering results
    return {
        'labels': cluster_labels,
        'centers': centers_df,
        'silhouette_score': silhouette_avg
    }
