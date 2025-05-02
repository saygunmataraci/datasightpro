# DataSightPro: Getting Started

This guide will help you quickly get up and running with DataSightPro, enabling you to start analyzing your data in minutes.

## Quick Installation

### Prerequisites
- Python 3.11 or higher
- Git (for cloning the repository)

### Step-by-Step Setup

1. **Clone the repository and navigate to the project folder**
   ```bash
   git clone https://github.com/saygunmataraci/datasightpro.git
   cd DataSightPro
   ```

2. **Set up a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it (Windows)
   venv\Scripts\activate
   
   # Activate it (macOS/Linux)
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the web interface**
   Open your browser and go to: http://localhost:8501

## Your First Analysis

### Using the Sample Dataset

1. When the application launches, click on "Use Sample Dataset" in the sidebar
2. The sample customer dataset will load automatically
3. You can immediately start exploring and analyzing this data

### Using Your Own Data

1. Prepare your data file (CSV, Excel, or JSON format)
2. Click "Upload Dataset" in the sidebar
3. Select your file from your computer
4. Once uploaded, the application will automatically display an overview

## Quick Tour

### 1. Dataset Overview
- Examine the first few rows of your data
- Review automatically detected data types
- Check for missing values and their distribution
- View basic statistics for each column

### 2. Statistical Analysis
- Generate detailed summary statistics
- Explore correlations between variables
- Analyze distributions and identify outliers

### 3. Visualization
- Create histograms, box plots, and scatter plots
- Generate correlation heatmaps
- Visualize categorical data with bar charts and pie charts

### 4. Preprocessing
- Handle missing values with various imputation strategies
- Encode categorical variables for machine learning
- Scale numerical features
- Transform skewed distributions
- Remove outliers

### 5. Export
- Save your processed data in CSV, Excel, or JSON format
- Include a processing summary with your export

## 5-Minute Example Workflow

Here's a quick example workflow to try with the sample dataset or your own data:

1. **Load Data**: Use the sample dataset or upload your own
2. **Explore**: In Dataset Overview, note the data types and check for missing values
3. **Visualize**: Go to Visualization → Distribution Visualizations → select "Histogram" → choose a numerical column
4. **Analyze**: Go to Statistical Analysis → Correlation Analysis to see relationships between variables
5. **Process**: Go to Preprocessing → Missing Values → select columns with missing values → choose "Mean/Mode" strategy → Apply
6. **Export**: Go to Export Options → select CSV format → click "Generate Export File"

## Next Steps

Now that you're up and running with DataSightPro, you can:

- Explore the [User Guide](UserGuide.md) for detailed instructions on all features
- Reference the [Quick Reference](QuickReference.md) for common tasks and options
- Review the [Technical Report](TechnicalReport.md) to understand the implementation details

## Troubleshooting

### Common Issues

- **Application doesn't start**: Ensure you're in the correct directory and that all dependencies are installed
- **File upload fails**: Check that your file is in a supported format and under 200MB
- **Visualizations don't display**: Verify you've selected compatible columns for the chart type
- **Processing errors**: Look for extreme values or inconsistent data types in your dataset

### Getting Help

If you encounter any issues not covered in this guide:

1. Check the [User Guide](UserGuide.md) for more detailed information
2. Look for similar issues in the project's issue tracker
3. Contact the project maintainers for support

Happy data analyzing!