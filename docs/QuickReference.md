# DataSightPro Quick Reference Guide

## Installation & Startup

```bash
# Clone repository
git clone https://github.com/saygunmataraci/datasightpro.git
cd DataSightPro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Launch application
streamlit run app.py
```

## UI Navigation

| Section | Description | Key Features |
|---------|-------------|-------------|
| Dataset Overview | Initial data assessment | Data preview, type detection, missing values summary |
| Statistical Analysis | Statistical insights | Summary statistics, correlation analysis, distribution metrics |
| Visualization | Interactive charts | Distribution, relationship, and categorical visualizations |
| Preprocessing | Data preparation | Missing value handling, encoding, scaling, transformation |
| Export | Data export | Export to CSV, Excel, or JSON with processing summary |

## Common Tasks

### Loading Data

1. Click "Upload Dataset" in the sidebar
2. Select a CSV, Excel, or JSON file
3. Wait for processing confirmation

### Exploring Dataset

1. Review data types and missing values in Dataset Overview
2. Check basic statistics for each column
3. Examine correlations in Statistical Analysis section
4. Create visualizations to understand distributions and relationships

### Preprocessing Workflow

1. Handle missing values (Preprocessing → Missing Values tab)
2. Encode categorical variables (Preprocessing → Categorical Encoding)
3. Scale numerical features (Preprocessing → Feature Scaling)
4. Transform skewed distributions (Preprocessing → Feature Selection/Transformation)
5. Remove outliers if needed (Preprocessing → Feature Selection/Transformation)
6. Export processed data (Export Options section)

## Visualization Reference

| Visualization Type | Best Used For | Required Data |
|-------------------|---------------|---------------|
| Histogram | Distribution of numerical variables | 1 numerical column |
| Box Plot | Distributions and outliers | 1+ numerical columns |
| Violin Plot | Distribution density across categories | 1 categorical, 1 numerical column |
| Scatter Plot | Relationships between variables | 2 numerical columns |
| Correlation Heatmap | Correlation strengths between all variables | Multiple numerical columns |
| Pair Plot | Relationships between multiple variables | 2+ numerical columns |
| Bar Chart | Comparing values across categories | 1 categorical, 1 numerical column |
| Pie Chart | Part-to-whole relationships | 1 categorical, 1 numerical column |
| Sunburst Chart | Hierarchical data | 2+ categorical columns |

## Preprocessing Options

### Missing Value Handling

| Strategy | Description | Best For |
|----------|-------------|----------|
| Mean/Mode | Replace with mean (numerical) or mode (categorical) | Data with random missing values |
| Median | Replace with median (numerical) or mode (categorical) | Data with outliers |
| Constant | Replace with specified value | Domain-specific requirements |

### Categorical Encoding

| Method | Description | Best For |
|--------|-------------|----------|
| One-Hot | Creates binary columns for each category | Most ML algorithms |
| Label | Assigns integers to categories | Tree-based algorithms |

### Feature Scaling

| Method | Description | Best For |
|--------|-------------|----------|
| MinMax | Scales to [0,1] range | Neural networks, algorithms requiring bounded values |
| Standard | Transforms to zero mean, unit variance | SVM, linear/logistic regression, PCA |
| Robust | Uses median and IQR | Data with outliers |

### Outlier Handling

| Method | Description | Threshold |
|--------|-------------|-----------|
| IQR | Uses interquartile range | Default: 1.5 IQR |
| Z-Score | Uses standard deviation | Default: 3 standard deviations |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+F / Cmd+F | Search within the application |
| Ctrl+R / Cmd+R | Refresh the application |
| Escape | Close current modal dialog |
| Tab | Navigate between interactive elements |

## Troubleshooting Tips

- **File upload fails**: Ensure file is under 200MB and in a supported format
- **Charts not displaying**: Verify selected columns match required data types
- **Processing errors**: Check for extreme values or incompatible data types
- **Performance issues**: Use a smaller dataset or sample your data
- **Missing visualization options**: Ensure you've selected appropriate columns for the visualization type

## Export Options

| Format | Best For | Limitations |
|--------|----------|-------------|
| CSV | Universal compatibility | Limited formatting options |
| Excel | Rich formatting | Larger file size |
| JSON | Web applications | Less efficient for tabular data |

## Tips for Best Results

- Start with Dataset Overview to understand your data structure
- Always check for missing values and outliers before analysis
- Use appropriate visualizations for your data types
- Document your preprocessing steps for reproducibility
- Export interim results at key stages of your analysis
- For large datasets, consider using sampling to improve performance

Visit the [User Guide](UserGuide.md) for detailed instructions and the [Technical Report](TechnicalReport.md) for in-depth implementation details.