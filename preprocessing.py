"""
Data Preprocessing Module
-------------------------
This module provides functions for preparing datasets for machine learning tasks.
It includes tools for handling missing values, encoding categorical variables,
normalizing numerical features, removing outliers, transforming skewed data,
and creating polynomial features.

Each function returns a new DataFrame without modifying the original input.
"""

# Standard library imports
import pandas as pd
import numpy as np

# Machine learning imports
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Statistical functions
import scipy.stats as stats


def handle_missing_values(df, strategy="mean/mode", fill_value=None, columns=None):
    """
    Handle missing values in the dataframe using the specified strategy.
    
    This function processes numerical and categorical columns differently:
    - Numerical columns: Uses SimpleImputer with appropriate strategy
    - Categorical columns: Uses pandas' mode or constant fill methods
    
    Args:
        df: The pandas DataFrame containing missing values
        strategy: Strategy to use for imputation:
                 "mean/mode" - Use mean for numerical, mode for categorical
                 "median" - Use median for numerical, mode for categorical
                 "constant" - Use specified fill_value for all columns
        fill_value: Value to use with "constant" strategy (defaults to 0 for numerical
                   columns if a non-numeric value is provided)
        columns: Specific columns to handle, or None to process all columns
    
    Returns:
        DataFrame: A new DataFrame with missing values handled
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Determine which columns to process (default: all columns)
    if columns is None:
        columns = df.columns.tolist()
    
    # Separate columns by data type for appropriate processing
    num_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_cols = [col for col in columns if col not in num_cols]
    
    # ==== NUMERICAL COLUMNS PROCESSING ====
    if num_cols:
        if strategy == "mean/mode":
            # Replace missing values with column mean
            imputer = SimpleImputer(strategy="mean")
            result_df[num_cols] = pd.DataFrame(
                imputer.fit_transform(result_df[num_cols]), 
                columns=num_cols,
                index=result_df.index
            )
        elif strategy == "median":
            # Replace missing values with column median
            imputer = SimpleImputer(strategy="median")
            result_df[num_cols] = pd.DataFrame(
                imputer.fit_transform(result_df[num_cols]), 
                columns=num_cols,
                index=result_df.index
            )
        elif strategy == "constant":
            # Try to convert fill_value to float for numeric columns
            # If conversion fails, default to 0
            try:
                num_fill_value = float(fill_value)
            except (ValueError, TypeError):
                num_fill_value = 0
                
            # Replace missing values with the constant value
            imputer = SimpleImputer(strategy="constant", fill_value=num_fill_value)
            result_df[num_cols] = pd.DataFrame(
                imputer.fit_transform(result_df[num_cols]), 
                columns=num_cols,
                index=result_df.index
            )
    
    # ==== CATEGORICAL COLUMNS PROCESSING ====
    if cat_cols:
        if strategy == "mean/mode":
            # For categorical data, use the most frequent value (mode)
            for col in cat_cols:
                mode_value = result_df[col].mode().iloc[0]  # Take first mode if multiple exist
                result_df[col] = result_df[col].fillna(mode_value)
        elif strategy == "constant":
            # Replace with the provided constant value
            for col in cat_cols:
                result_df[col] = result_df[col].fillna(fill_value)
        else:
            # For any other strategy (like "median"), default to mode for categorical
            for col in cat_cols:
                mode_value = result_df[col].mode().iloc[0]
                result_df[col] = result_df[col].fillna(mode_value)
    
    return result_df


def encode_categorical_variables(df, columns=None, method="one_hot_encoding"):
    """
    Convert categorical variables to numerical format using specified encoding method.
    
    Args:
        df: The pandas DataFrame containing categorical variables
        columns: List of categorical columns to encode, or None to process all non-numeric columns
        method: Encoding method to use:
                "one_hot_encoding" - Creates binary columns for each category (dummy variables)
                "label_encoding" - Assigns a unique integer to each category
    
    Returns:
        DataFrame: A new DataFrame with categorical variables encoded
        
    Notes:
        - One-hot encoding creates new columns and removes original columns
        - Label encoding preserves the original columns but changes their values
        - Missing values are filled with the mode before label encoding
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If no columns specified, detect all non-numeric columns
    if columns is None:
        columns = df.select_dtypes(exclude=['number']).columns.tolist()
    else:
        # Filter to include only columns that exist in the dataframe
        columns = [col for col in columns if col in df.columns]
    
    # Return immediately if no columns to process
    if not columns:
        return result_df
    
    # Apply the selected encoding method
    if method == "one_hot_encoding":
        # Convert categories to binary indicator columns (0/1)
        for col in columns:
            # Create dummy variables with column name prefix to avoid name collisions
            dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=False)
            
            # Add the new dummy columns to the dataframe
            result_df = pd.concat([result_df, dummies], axis=1)
            
            # Remove the original categorical column
            result_df = result_df.drop(col, axis=1)
    
    elif method == "label_encoding":
        # Convert categories to integer values (0, 1, 2, ...)
        label_encoder = LabelEncoder()
        
        for col in columns:
            # Handle missing values before encoding (LabelEncoder can't process NaN)
            if result_df[col].isna().any():
                # Fill missing values with the most common category
                result_df[col] = result_df[col].fillna(result_df[col].mode().iloc[0])
            
            # Transform categorical values to integers
            result_df[col] = label_encoder.fit_transform(result_df[col])
    
    return result_df


def normalize_data(df, columns=None, method="minmax"):
    """
    Scale numerical features to a standard range using the specified method.
    
    Args:
        df: The pandas DataFrame containing numerical features to normalize
        columns: List of numerical columns to normalize, or None for all numerical columns
        method: Normalization method to use:
                "minmax" - Scales features to [0,1] range
                "standard" - Transforms to zero mean and unit variance (z-score)
                "robust" - Uses median and IQR, less sensitive to outliers
    
    Returns:
        DataFrame: A new DataFrame with normalized numerical features
        
    Notes:
        - MinMax scaling is sensitive to outliers but preserves the shape of distribution
        - Standardization centers data around 0 with std of 1, good for many ML algorithms
        - Robust scaling is preferred when data contains significant outliers
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If no columns specified, detect all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter to include only numerical columns
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Return immediately if no columns to process
    if not columns:
        return result_df
    
    # Apply the selected normalization method
    if method == "minmax":
        # ==== Min-Max Scaling [0,1] ====
        # Formula: x_scaled = (x - min) / (max - min)
        scaler = MinMaxScaler()
        result_df[columns] = pd.DataFrame(
            scaler.fit_transform(result_df[columns]),
            columns=columns,
            index=result_df.index
        )
    
    elif method == "standard":
        # ==== Standardization (Z-score normalization) ====
        # Formula: z = (x - mean) / std
        # Result: Mean = 0, Standard deviation = 1
        scaler = StandardScaler()
        result_df[columns] = pd.DataFrame(
            scaler.fit_transform(result_df[columns]),
            columns=columns,
            index=result_df.index
        )
    
    elif method == "robust":
        # ==== Robust Scaling ====
        # Formula: z = (x - median) / IQR
        # Result: Less influenced by outliers than standard scaling
        # Import RobustScaler only when needed
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        result_df[columns] = pd.DataFrame(
            scaler.fit_transform(result_df[columns]),
            columns=columns,
            index=result_df.index
        )
    
    return result_df


def remove_outliers(df, columns=None, method="iqr", threshold=1.5):
    """
    Identify and remove rows containing outlier values in specified columns.
    
    Args:
        df: The pandas DataFrame to process
        columns: List of numerical columns to check for outliers, or None for all numerical columns
        method: Method to identify outliers:
                "iqr" - Interquartile Range method (default threshold=1.5)
                "zscore" - Standard deviation distance from mean (default threshold=3)
        threshold: Threshold multiplier for outlier detection:
                  - For IQR method: points > Q3 + threshold*IQR or < Q1 - threshold*IQR
                  - For Z-score method: points with z-scores > threshold
    
    Returns:
        DataFrame: A new DataFrame with outlier rows removed
        
    Notes:
        - The function removes entire rows if they contain outliers in any specified column
        - IQR method works well for skewed distributions
        - Z-score method assumes normally distributed data
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If no columns specified, detect all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter to include only numerical columns
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Return immediately if no columns to process
    if not columns:
        return result_df
    
    # Initialize mask to track which rows should be kept (True = keep, False = remove)
    mask = pd.Series(True, index=df.index)
    
    # Apply the selected outlier detection method
    if method == "iqr":
        # ==== IQR (Interquartile Range) Method ====
        # Commonly used in box plots to identify outliers
        for col in columns:
            # Calculate quartiles and IQR
            Q1 = result_df[col].quantile(0.25)  # 25th percentile
            Q3 = result_df[col].quantile(0.75)  # 75th percentile
            IQR = Q3 - Q1  # Interquartile range
            
            # Define outlier boundaries
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Update the mask to exclude outliers in this column
            # (The tilde ~ operator inverts the boolean condition)
            mask = mask & ~((result_df[col] < lower_bound) | (result_df[col] > upper_bound))
    
    elif method == "zscore":
        # ==== Z-score Method ====
        # Assumes data follows a normal distribution
        for col in columns:
            # Calculate absolute z-scores for the column
            # Formula: z = |x - mean| / std
            z_scores = np.abs((result_df[col] - result_df[col].mean()) / result_df[col].std())
            
            # Update the mask to exclude points with z-scores beyond the threshold
            mask = mask & (z_scores < threshold)
    
    # Apply the mask to keep only non-outlier rows
    return result_df[mask]


def transform_skewed_data(df, columns=None, method="log", threshold=0.5):
    """
    Apply transformations to reduce skewness in numerical data distributions.
    
    Args:
        df: The pandas DataFrame containing skewed numerical data
        columns: List of numerical columns to transform, or None for all numerical columns
        method: Transformation method to apply:
                "log" - Natural logarithm transformation
                "sqrt" - Square root transformation
                "boxcox" - Box-Cox power transformation (requires positive values)
                "yeojohnson" - Yeo-Johnson transformation (works with negative values)
        threshold: Absolute skewness value above which to apply transformation
                  (typical values range from 0.5 to 1.0)
    
    Returns:
        DataFrame: A new DataFrame with transformed data
        
    Notes:
        - Transformations are only applied to columns with skewness > threshold
        - Log, sqrt, and boxcox transformations require positive values (shifting is applied if needed)
        - Yeo-Johnson works with both positive and negative values
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If no columns specified, detect all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter to include only numerical columns
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Process each column individually
    for col in columns:
        # Measure the asymmetry of the data distribution (skewness)
        skewness = result_df[col].skew()
        
        # Only apply transformation if the distribution is sufficiently skewed
        if abs(skewness) > threshold:
            # Some transformations require strictly positive values
            # If data contains zero or negative values, shift the entire distribution
            if method in ["log", "sqrt", "boxcox"]:
                min_val = result_df[col].min()
                # If minimum value is non-positive, shift the entire distribution
                shift = 0 if min_val > 0 else abs(min_val) + 1
                shifted_data = result_df[col] + shift
            
            # Apply the selected transformation method
            if method == "log":
                # Natural logarithm transformation
                # Effective for strong positive skew (right-skewed data)
                result_df[col] = np.log(shifted_data)
            elif method == "sqrt":
                # Square root transformation
                # Milder effect than log, good for moderate positive skew
                result_df[col] = np.sqrt(shifted_data)
            elif method == "boxcox":
                # Box-Cox power transformation
                # Automatically finds optimal lambda parameter for normality
                # More flexible than log or sqrt transformations
                result_df[col], _ = stats.boxcox(shifted_data)
            elif method == "yeojohnson":
                # Yeo-Johnson transformation
                # Extension of Box-Cox that handles negative values
                # Most flexible transformation, works on any distribution
                result_df[col], _ = stats.yeojohnson(result_df[col])
    
    return result_df


def create_polynomial_features(df, columns, degree=2, interaction_only=False):
    """
    Generate polynomial and interaction features from numerical columns.
    
    Args:
        df: The pandas DataFrame containing features
        columns: List of numerical columns to use for creating polynomial features
        degree: The maximum polynomial degree (default: 2)
        interaction_only: If True, only generates interaction terms without powers
                         (e.g., x1*x2 but not x1², x2²)
    
    Returns:
        DataFrame: A new DataFrame with polynomial features added as new columns
    
    Notes:
        - For degree=2 with columns ['a', 'b'], creates features like ['a²', 'a*b', 'b²']
        - With interaction_only=True, only creates ['a*b']
        - Original columns are preserved, and new polynomial features are added
        - Feature names follow the pattern: 'a^2', 'a b', 'b^2' etc.
    """
    # Import here to avoid overhead if function isn't used
    from sklearn.preprocessing import PolynomialFeatures
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Filter to include only numerical columns
    columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Return immediately if no columns to process
    if not columns:
        return result_df
    
    # ==== Generate polynomial features ====
    # include_bias=False excludes the constant term (1) from the output
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    
    # Transform the selected columns
    poly_features = poly.fit_transform(result_df[columns])
    
    # Get feature names generated by the transformation
    feature_names = poly.get_feature_names_out(columns)
    
    # Convert the polynomial features to DataFrame format
    if degree > 1:
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=result_df.index)
        
        # Remove the original linear features to avoid duplication
        # (they already exist in the original DataFrame)
        poly_df = poly_df.drop(columns=columns, errors='ignore')
        
        # Add the new polynomial features to the result
        result_df = pd.concat([result_df, poly_df], axis=1)
    else:
        # If degree is 1, no transformation is needed
        return result_df
    
    return result_df
