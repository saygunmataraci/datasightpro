# DataSightPro User Guide

## Table of Contents

- [DataSightPro User Guide](#datasightpro-user-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Launching the Application](#launching-the-application)
    - [Interface Overview](#interface-overview)
  - [Working with Data](#working-with-data)
    - [Loading Data](#loading-data)
    - [Supported File Formats](#supported-file-formats)
    - [Using Sample Data](#using-sample-data)
  - [Dataset Overview](#dataset-overview)
    - [Data Preview](#data-preview)
    - [Data Type Information](#data-type-information)
    - [Missing Values Summary](#missing-values-summary)
    - [Basic Statistics](#basic-statistics)
  - [Statistical Analysis](#statistical-analysis)
    - [Summary Statistics](#summary-statistics)
    - [Correlation Analysis](#correlation-analysis)
    - [Distribution Analysis](#distribution-analysis)
  - [Data Visualization](#data-visualization)
    - [Distribution Visualizations](#distribution-visualizations)
    - [Relationship Visualizations](#relationship-visualizations)
    - [Categorical Visualizations](#categorical-visualizations)
  - [Data Preprocessing](#data-preprocessing)
    - [Handling Missing Values](#handling-missing-values)
    - [Encoding Categorical Variables](#encoding-categorical-variables)
    - [Feature Scaling](#feature-scaling)
    - [Outlier Removal](#outlier-removal)
    - [Feature Engineering](#feature-engineering)
  - [Exporting Data](#exporting-data)
    - [Export Formats](#export-formats)
    - [Export Options](#export-options)
  - [Common Workflows](#common-workflows)
    - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Preparing Data for Machine Learning](#preparing-data-for-machine-learning)
    - [Feature Engineering Workflow](#feature-engineering-workflow)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues and Solutions](#common-issues-and-solutions)
  - [Tips and Best Practices](#tips-and-best-practices)

## Introduction

DataSightPro is a comprehensive data analysis and machine learning preparation tool designed to streamline the process of exploring, analyzing, visualizing, and preparing data for machine learning tasks. This user guide provides detailed instructions on how to use DataSightPro effectively.

## Getting Started

### Installation

Before using DataSightPro, you need to install it on your machine:

1. Ensure you have Python 3.11 or higher installed
2. Clone the repository:
   ```bash
   git clone https://github.com/saygunmataraci/datasightpro.git
   cd DataSightPro
   ```

3. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. Install the dependencies:
   ```bash
   pip install -e .
   ```

### Launching the Application

To start DataSightPro:

1. Ensure you're in the DataSightPro directory
2. Run the application:
   ```bash
   streamlit run app.py
   ```

3. Your default web browser should automatically open to `http://localhost:8501`. If not, open your browser and navigate to this address.

### Interface Overview

The DataSightPro interface consists of:

- **Sidebar**: Contains the file uploader, navigation menu, and dataset information
- **Main Content Area**: Displays the currently selected section's content
- **Tabs**: Separate different functionalities within each section

The application is divided into the following main sections:

1. **Dataset Overview**: Provides a summary of your dataset
2. **Statistical Analysis**: Offers various statistical analyses
3. **Visualization**: Creates interactive visualizations
4. **Preprocessing**: Provides tools for preparing data for machine learning
5. **Export Options**: Allows you to export your processed data

## Working with Data

### Loading Data

To load a dataset into DataSightPro:

1. Click on the "Upload Dataset" button in the sidebar
2. Select your file from your computer's file system
3. Wait for the file to upload and process
4. Once processed, you'll see a confirmation message and basic dataset information in the sidebar

### Supported File Formats

DataSightPro supports the following file formats:

- **CSV** (.csv): Comma-separated values files
- **Excel** (.xlsx): Microsoft Excel spreadsheets
- **JSON** (.json): JavaScript Object Notation files

### Using Sample Data

If you want to try the application without uploading your own data:

1. In the sidebar, click the "Use Sample Dataset" button
2. The sample customer data will be loaded
3. You can now explore all features using this dataset

## Dataset Overview

The Dataset Overview section provides a quick summary of your dataset.

### Data Preview

The data preview shows the first few rows of your dataset:

1. Navigate to the Dataset Overview section
2. View the data table at the top of the page
3. Use the pagination controls to navigate through the data
4. Sort columns by clicking on column headers

### Data Type Information

DataSightPro automatically detects and displays the data types of each column:

1. In the Dataset Overview section, find the "Data Types" card
2. Review the automatically detected data types
3. For each column, you'll see one of the following types:
   - Numerical (Continuous)
   - Numerical (Discrete)
   - Categorical
   - DateTime
   - Boolean
   - Text

### Missing Values Summary

To understand missing data in your dataset:

1. Locate the "Missing Values" card in the Dataset Overview
2. Review the count and percentage of missing values for each column
3. Use this information to determine which columns need preprocessing

### Basic Statistics

The basic statistics section provides a quick overview of your data's characteristics:

1. Find the "Basic Statistics" card
2. For numerical columns, you'll see min, max, mean, median, and standard deviation
3. For categorical columns, you'll see the count of unique values and the most frequent category

## Statistical Analysis

The Statistical Analysis section provides more in-depth analysis of your data.

### Summary Statistics

To view comprehensive summary statistics:

1. Navigate to the Statistical Analysis section
2. Click on the "Summary Statistics" tab
3. Select columns of interest from the dropdown
4. View detailed statistics including:
   - Central tendency measures (mean, median, mode)
   - Dispersion measures (standard deviation, variance, range)
   - Distribution shape metrics (skewness, kurtosis)
   - Percentiles (quartiles, 5th and 95th percentiles)

### Correlation Analysis

To analyze relationships between variables:

1. In the Statistical Analysis section, click on the "Correlation Analysis" tab
2. The correlation matrix will be displayed as an interactive heatmap
3. Hover over cells to see the exact correlation coefficient
4. Strong positive correlations are shown in dark blue, while strong negative correlations are shown in dark red
5. Use the controls to filter by correlation strength or specific variables

### Distribution Analysis

To analyze the distributions of numerical variables:

1. Click on the "Distribution Analysis" tab
2. Select a numerical column from the dropdown
3. View the distribution metrics:
   - Shape parameters (skewness, kurtosis)
   - Normality test results
   - Outlier information
4. The analysis will indicate whether the distribution is normal, skewed, or has outliers

## Data Visualization

The Visualization section provides interactive charts to visualize your data.

### Distribution Visualizations

To create distribution visualizations:

1. Navigate to the Visualization section
2. Click on the "Distribution Visualizations" tab
3. Select the visualization type (Histogram, Box Plot, or Violin Plot)
4. For Histograms:
   - Select a column to visualize
   - Optionally, select a column to group by
   - Adjust the number of bins if needed
5. For Box Plots:
   - Select one or more columns to visualize
   - Optionally, select a column to group by
   - Choose the orientation (vertical or horizontal)
6. For Violin Plots:
   - Select a categorical column for the x-axis
   - Select a numerical column for the y-axis
   - Optionally, select a column to group by

### Relationship Visualizations

To create visualizations showing relationships between variables:

1. Click on the "Relationship Visualizations" tab
2. Select the visualization type (Scatter Plot, Correlation Heatmap, or Pair Plot)
3. For Scatter Plots:
   - Select columns for x and y axes
   - Optionally, select columns for color and size encoding
4. For Correlation Heatmaps:
   - The heatmap will automatically display correlations between all numerical variables
   - Hover over cells to see exact correlation values
5. For Pair Plots:
   - Select multiple columns to include
   - Optionally, select a column to color by

### Categorical Visualizations

To visualize categorical data:

1. Click on the "Categorical Visualizations" tab
2. Select the visualization type (Bar Chart, Pie Chart, or Sunburst Chart)
3. For Bar Charts:
   - Select a categorical column for categories
   - Select a numerical column for values
   - Choose the orientation (vertical or horizontal)
4. For Pie Charts:
   - Select a categorical column for segments
   - Select a numerical column for values
5. For Sunburst Charts:
   - Select columns to define the hierarchy levels
   - Optionally, select a column for segment sizes

## Data Preprocessing

The Preprocessing section provides tools to prepare your data for machine learning.

### Handling Missing Values

To handle missing values in your dataset:

1. Navigate to the Preprocessing section
2. Click on the "Missing Values" tab
3. Select columns to process (or process all columns)
4. Choose a strategy:
   - Mean/Mode: Replaces missing values with mean for numerical and mode for categorical
   - Median: Uses median for numerical and mode for categorical
   - Constant: Replaces with a specified value
5. If using Constant strategy, enter the fill value
6. Click "Apply" to process the data
7. Review the results in the preview table

### Encoding Categorical Variables

To convert categorical variables to numerical format:

1. Click on the "Categorical Encoding" tab
2. Select categorical columns to encode
3. Choose an encoding method:
   - One-Hot Encoding: Creates binary columns for each category
   - Label Encoding: Assigns a unique integer to each category
4. Click "Apply" to encode the data
5. Review the results in the preview table

### Feature Scaling

To scale numerical features:

1. Click on the "Feature Scaling" tab
2. Select numerical columns to scale
3. Choose a scaling method:
   - MinMax: Scales features to [0,1] range
   - Standard: Transforms to zero mean and unit variance
   - Robust: Uses median and IQR, less sensitive to outliers
4. Click "Apply" to scale the data
5. Review the results in the preview table

### Outlier Removal

To identify and remove outliers:

1. Click on the "Feature Selection/Transformation" tab
2. In the Outlier Removal section, select columns to check for outliers
3. Choose a detection method:
   - IQR Method: Uses interquartile range
   - Z-Score Method: Uses standard deviation distance from mean
4. Adjust the threshold if needed
5. Click "Remove Outliers" to process the data
6. Review the results, including the number of rows removed

### Feature Engineering

To create new features:

1. In the "Feature Selection/Transformation" tab
2. For polynomial features:
   - Select columns to use
   - Set the polynomial degree
   - Choose whether to include interaction terms only
   - Click "Create Polynomial Features"
3. For distribution transformation:
   - Select columns with skewed distributions
   - Choose a transformation method (Log, Square Root, Box-Cox, or Yeo-Johnson)
   - Set the skewness threshold
   - Click "Transform Distributions"
4. Review the results in the preview table

## Exporting Data

Once you've finished preprocessing your data, you can export it for use in machine learning or other applications.

### Export Formats

DataSightPro supports exporting to:

1. CSV (.csv)
2. Excel (.xlsx)
3. JSON (.json)

### Export Options

To export your processed data:

1. Navigate to the Export Options section
2. Select the export format
3. Optionally, provide a filename (default is "processed_data")
4. Choose whether to include an export summary with processing details
5. Click "Generate Export File" to create the export
6. Click the download link that appears to save the file to your computer

## Common Workflows

### Exploratory Data Analysis

A typical exploratory data analysis workflow:

1. Upload your dataset
2. In Dataset Overview, review the data types and missing values
3. In Statistical Analysis, examine summary statistics and correlation matrix
4. In Visualization:
   - Create histograms for important numerical variables
   - Use box plots to identify outliers
   - Create scatter plots to visualize relationships between key variables
   - Generate a correlation heatmap to identify strong relationships
5. Return to Statistical Analysis to perform hypothesis tests on identified relationships

### Preparing Data for Machine Learning

A typical workflow for preparing data for machine learning:

1. Upload your dataset
2. In Dataset Overview, identify data quality issues
3. In Preprocessing:
   - Handle missing values using appropriate strategies
   - Encode categorical variables (one-hot encoding for most algorithms)
   - Scale numerical features (standardization for many ML algorithms)
   - Remove or transform outliers if needed
   - Transform skewed distributions for better model performance
4. In Export Options, export the processed dataset for use in your ML pipeline

### Feature Engineering Workflow

A typical feature engineering workflow:

1. Upload your dataset
2. In Statistical Analysis, identify important features and relationships
3. In Preprocessing:
   - Transform features with skewed distributions
   - Create polynomial features to capture non-linear relationships
   - Remove irrelevant features
4. In Visualization, verify the impact of engineered features using scatter plots or correlation analysis
5. In Export Options, export the engineered dataset

## Troubleshooting

### Common Issues and Solutions

1. **File Upload Issues**
   - Ensure your file is in a supported format (CSV, Excel, JSON)
   - Check that your file isn't too large (max 200MB)
   - Verify your file isn't corrupted by opening it in another application

2. **Processing Errors**
   - If you encounter an error during processing, check the error message for details
   - For errors during encoding, ensure categorical columns don't have too many unique values
   - For errors during scaling, check for infinite or extremely large values

3. **Visualization Problems**
   - If visualizations fail to render, try selecting different columns
   - For large datasets, try limiting the number of rows or columns visualized
   - Ensure the selected columns match the required data types for the visualization

4. **Performance Issues**
   - For large datasets, consider sampling the data
   - Close other applications to free up system resources
   - If the application becomes unresponsive, refresh the page and try again with a smaller dataset

## Tips and Best Practices

1. **Data Preparation**
   - Always examine your data before preprocessing
   - Handle missing values before encoding categorical variables
   - Scale features after handling missing values and encoding

2. **Visualization**
   - Use appropriate chart types for your data:
     - Histograms and box plots for numerical distributions
     - Scatter plots for relationships between numerical variables
     - Bar charts for categorical comparisons
   - Limit the number of categories in categorical visualizations for clarity

3. **Preprocessing**
   - Choose imputation strategies based on your data and domain knowledge
   - Consider the impact of preprocessing decisions on your models
   - Document your preprocessing steps for reproducibility

4. **Workflow Efficiency**
   - Use the sidebar navigation to quickly move between sections
   - Save intermediate results by exporting processed data at key stages
   - For repeated analyses, create a script to automate the preprocessing steps

By following this guide, you'll be able to effectively utilize DataSightPro to explore, analyze, visualize, and prepare your data for machine learning tasks.