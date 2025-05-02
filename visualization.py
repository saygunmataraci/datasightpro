"""
Data Visualization Module
=========================

This module provides a comprehensive set of functions for creating interactive visualizations
using Plotly. Each function is designed to handle common data visualization tasks with
appropriate defaults while offering customization options.

The module is organized into the following categories:
- Distribution visualizations (histograms, box plots, violin plots)
- Relationship visualizations (scatter plots, correlation matrices, pair plots)
- Categorical visualizations (bar charts, pie charts, sunburst charts)
- Time series visualizations (line charts)
- Multidimensional visualizations (3D scatter plots)

All functions return Plotly figure objects that can be displayed in web applications,
notebooks, or exported as static images.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# =====================================================================
# DISTRIBUTION VISUALIZATIONS
# =====================================================================

def create_histogram(df, column, bins=None, color=None):
    """
    Create a histogram to visualize the distribution of a numerical variable.
    
    This function creates a histogram with additional statistical information.
    When color is provided, it creates a grouped histogram with a marginal rug plot.
    When no color is provided, it adds mean and median lines and a marginal box plot.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        column (str): The column name to plot (should be numerical).
        bins (int, optional): Number of bins to use. If None, calculated using Sturges' formula.
        color (str, optional): Column name to group and color by. Creates a grouped histogram.
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Auto-calculate bins if not provided using Sturges' formula
    # Formula: k = ceiling(log2(n) + 1) where n is the number of data points
    if bins is None:
        bins = int(np.ceil(np.log2(len(df))) + 1)
        bins = min(bins, 50)  # Cap at 50 bins to prevent overplotting with large datasets
    
    if color:
        # Create a grouped histogram with a marginal rug plot when color is specified
        # Rug plots add small ticks along the axis to show individual data points
        fig = px.histogram(
            df, 
            x=column,
            color=color,
            marginal="rug",  # Add a rug plot to show distribution of individual points
            nbins=bins,
            opacity=0.7,     # Semi-transparent bars to see overlap
            barmode="overlay", # Overlay bars from different groups
            histnorm="probability density", # Normalize y-axis as density
            title=f"Distribution of {column} by {color}"
        )
    else:
        # Calculate distribution statistics for annotations
        mean_val = df[column].mean()
        median_val = df[column].median()
        
        # Create a single histogram with a marginal box plot
        fig = px.histogram(
            df, 
            x=column,
            marginal="box",  # Add a box plot to show quartiles and outliers
            nbins=bins,
            opacity=0.7,
            histnorm="probability density",
            title=f"Distribution of {column}"
        )
        
        # Add reference lines for mean and median with annotations
        fig.add_vline(
            x=mean_val, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Mean: {mean_val:.2f}", 
            annotation_position="top right"
        )
        fig.add_vline(
            x=median_val, 
            line_dash="dash", 
            line_color="green", 
            annotation_text=f"Median: {median_val:.2f}", 
            annotation_position="bottom right"
        )
    
    # Configure layout for better readability
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Density",
        legend_title=color if color else None,
        template="plotly_white"  # Clean white background with minimal gridlines
    )
    
    return fig


def create_box_plot(df, columns, color=None, orientation="v"):
    """
    Create box plots for one or more numerical columns to visualize distributions.
    
    Box plots show the quartiles, median, and potential outliers of numerical data.
    This function can handle both single and multiple columns, automatically restructuring
    the data as needed.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        columns (list): List of column names to plot. Can be a single column in a list.
        color (str, optional): Column name to group and color by.
        orientation (str, optional): "v" for vertical or "h" for horizontal box plots.
            Vertical is better for comparing distributions across categories.
            Horizontal is better for long category names and limited width.
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Handle multiple columns by melting the dataframe into long format
    if len(columns) > 1:
        # Melt converts from wide format to long format
        # e.g., from columns [A, B, C] to rows with Column/Value pairs
        melted_df = df[columns].melt(var_name="Column", value_name="Value")
        
        # Create oriented box plot based on specified orientation
        if orientation == "v":
            fig = px.box(
                melted_df, 
                x="Column",  # Categories on x-axis
                y="Value",   # Values on y-axis
                color=color if color in df.columns else None,
                points="outliers",  # Only show outlier points to reduce clutter
                title=f"Box Plots of Selected Variables"
            )
        else:  # horizontal orientation
            fig = px.box(
                melted_df, 
                y="Column",  # Categories on y-axis
                x="Value",   # Values on x-axis
                color=color if color in df.columns else None,
                points="outliers",
                title=f"Box Plots of Selected Variables"
            )
    else:
        # Handle single column case - no need to restructure data
        column = columns[0]
        
        if orientation == "v":
            fig = px.box(
                df, 
                y=column,  # Values on y-axis for vertical orientation
                color=color,
                points="outliers",
                title=f"Box Plot of {column}"
            )
        else:  # horizontal orientation
            fig = px.box(
                df, 
                x=column,  # Values on x-axis for horizontal orientation
                color=color,
                points="outliers",
                title=f"Box Plot of {column}"
            )
    
    # Configure layout with appropriate box display mode
    fig.update_layout(
        template="plotly_white",
        boxmode="group" if color else "overlay"  # Group boxes by color categories if provided
    )
    
    return fig


def create_violin_plot(df, x, y, color=None, orientation="v"):
    """
    Create a violin plot to visualize distribution density across categories.
    
    Violin plots combine box plots with kernel density estimates, showing the
    full distribution shape. This provides more information than a standard box plot.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        x (str): Column name for categorical variable (for vertical orientation).
        y (str): Column name for numerical variable (for vertical orientation).
        color (str, optional): Column name to group and color by.
        orientation (str, optional): "v" for vertical or "h" for horizontal.
            In horizontal orientation, x and y are swapped.
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Create oriented violin plot based on specified orientation
    if orientation == "v":
        fig = px.violin(
            df,
            x=x,  # Categorical variable on x-axis
            y=y,  # Numerical variable on y-axis
            color=color,
            box=True,    # Include box plot inside the violin for quartile reference
            points="all", # Show all individual data points for complete transparency
            title=f"Violin Plot of {y} by {x}"
        )
    else:  # horizontal orientation
        fig = px.violin(
            df,
            y=x,  # Categorical variable on y-axis
            x=y,  # Numerical variable on x-axis
            color=color,
            box=True,
            points="all",
            title=f"Violin Plot of {y} by {x}"
        )
    
    # Configure layout with appropriate violin display mode
    fig.update_layout(
        template="plotly_white",
        violinmode="group" if color else "overlay"  # Group or overlay violins based on color
    )
    
    return fig


# =====================================================================
# RELATIONSHIP VISUALIZATIONS
# =====================================================================

def create_scatter_plot(df, x_column, y_column, color_column=None, size_column=None, trendline=False):
    """
    Create a scatter plot to visualize relationships between two variables.
    
    Scatter plots are ideal for examining correlations and patterns between
    two numerical variables. This function supports additional dimensions through
    color and size parameters.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        x_column (str): Column name for the x-axis.
        y_column (str): Column name for the y-axis.
        color_column (str, optional): Column name to color points by (adds a third dimension).
        size_column (str, optional): Column name to scale point sizes by (adds a fourth dimension).
        trendline (bool, optional): Whether to add a trendline. Note: This parameter is currently
            not implemented to avoid statsmodels dependency, but kept for future compatibility.
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Note: trendline parameter is kept in the signature for API compatibility
    # but not implemented to avoid the statsmodels dependency
    
    # Create the scatter plot with optional color and size dimensions
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color_column,  # Color encodes an additional categorical or numerical dimension
        size=size_column,    # Size encodes an additional numerical dimension
        opacity=0.7,         # Semi-transparent points to handle overplotting
        title=f"Scatter Plot: {x_column} vs {y_column}"
    )
    
    # Configure axis labels and theme
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=color_column if color_column else None,
        template="plotly_white"
    )
    
    # Add correlation coefficient annotation if both variables are numerical
    if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
        correlation = df[[x_column, y_column]].corr().iloc[0, 1]
        fig.add_annotation(
            x=0.05, y=0.95,         # Position in the top-left corner of the plot
            xref="paper", yref="paper",  # Use paper coordinates (0-1) instead of data coordinates
            text=f"Correlation: {correlation:.4f}",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
            bordercolor="black",
            borderwidth=1
        )
    
    return fig


def create_correlation_heatmap(correlation_matrix):
    """
    Create a heatmap visualization of a correlation matrix.
    
    This function displays correlations between pairs of variables in a color-coded
    matrix. It shows only the lower triangle to avoid redundancy, as correlation
    matrices are symmetrical.
    
    Args:
        correlation_matrix (pandas.DataFrame): A pandas DataFrame containing 
            the correlation matrix, typically created with df.corr()
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Create a mask for the upper triangle of the correlation matrix
    # This avoids showing redundant information (correlation of A with B is same as B with A)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Apply the mask by replacing upper triangle values with NaN
    # so they won't be shown in the heatmap
    heatmap_data = correlation_matrix.copy()
    heatmap_data.values[mask] = np.nan
    
    # Create the heatmap using imshow which works well for displaying matrices
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale="RdBu_r",  # Red-Blue diverging colorscale (reversed)
        zmin=-1, zmax=1,                  # Fix scale to correlation range -1 to 1
        title="Correlation Heatmap"
    )
    
    # Configure layout and axes
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Features",
        template="plotly_white"
    )
    
    # Add correlation values as text annotations in the lower triangle
    for i, row in enumerate(correlation_matrix.index):
        for j, col in enumerate(correlation_matrix.columns):
            if i > j:  # Only show lower triangle
                # Add text with correlation value
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{correlation_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    # Use white text for dark cells (high positive/negative correlation)
                    # and black text for light cells (low correlation)
                    font=dict(color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black")
                )
    
    return fig


def create_pair_plot(df, columns, hue=None, kind="scatter"):
    """
    Create a matrix of plots showing relationships between multiple variables.
    
    A pair plot (or scatter matrix) creates a grid of plots with each variable
    plotted against every other variable, providing a comprehensive view of
    pairwise relationships in the dataset.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        columns (list): List of column names to include in the pair plot.
        hue (str, optional): Column name to color points by.
        kind (str, optional): Kind of plot to show. Currently, only "scatter" is 
            fully implemented in plotly.express.scatter_matrix.
            The parameter is kept for API compatibility with future enhancements.
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Create the scatter matrix using plotly express
    # This creates a grid of scatter plots for all variable combinations
    fig = px.scatter_matrix(
        df,
        dimensions=columns,  # Variables to include in the matrix
        color=hue,          # Color points by this variable
        opacity=0.7,        # Semi-transparent points
        title="Pair Plot"
    )
    
    # Configure the layout with clean styling
    fig.update_layout(
        template="plotly_white",
    )
    
    # Update all traces (individual scatter plots)
    fig.update_traces(
        diagonal_visible=True,   # Show plots on the diagonal (can be histograms)
        showupperhalf=False      # Only show lower triangle to avoid redundancy
    )
    
    return fig


# =====================================================================
# CATEGORICAL VISUALIZATIONS
# =====================================================================

def create_bar_chart(df, x, y, color=None, orientation="v"):
    """
    Create a bar chart to compare values across categories.
    
    Bar charts are ideal for comparing discrete values across categories.
    This function supports both vertical and horizontal orientations.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        x (str): Column name for categories (for vertical orientation).
        y (str): Column name for values (for vertical orientation).
        color (str, optional): Column name to group and color by.
        orientation (str, optional): "v" for vertical or "h" for horizontal.
            In horizontal orientation, x and y are swapped.
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Create oriented bar chart based on specified orientation
    if orientation == "v":
        fig = px.bar(
            df,
            x=x,  # Categories on x-axis
            y=y,  # Values on y-axis
            color=color,
            title=f"Bar Chart of {y} by {x}"
        )
    else:  # horizontal orientation
        fig = px.bar(
            df,
            y=x,  # Categories on y-axis
            x=y,  # Values on x-axis
            color=color,
            title=f"Bar Chart of {y} by {x}"
        )
    
    # Configure layout with appropriate bar mode
    # - group: bars side by side (used with color grouping)
    # - relative: standard stacked bars (default without color)
    fig.update_layout(
        template="plotly_white",
        barmode="group" if color else "relative"
    )
    
    return fig


def create_pie_chart(df, names, values):
    """
    Create a pie chart to visualize part-to-whole relationships.
    
    Pie charts show how individual parts make up a whole, with each slice
    representing a percentage of the total. Best used when there are relatively
    few categories (<7).
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        names (str): Column name for slice labels (categories).
        values (str): Column name for slice sizes (values).
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Create a pie chart with auto-calculated percentages
    fig = px.pie(
        df,
        names=names,    # Categories for each slice
        values=values,  # Values to determine slice sizes
        title=f"Pie Chart of {values} by {names}"
    )
    
    # Configure layout with clean styling
    fig.update_layout(
        template="plotly_white"
    )
    
    return fig


def create_sunburst_chart(df, path, values=None):
    """
    Create a sunburst chart for visualizing hierarchical data.
    
    Sunburst charts display hierarchical data through a series of rings,
    where each ring represents a level in the hierarchy. This is useful
    for showing nested categorical relationships.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        path (list): List of column names defining the hierarchical levels
            from center outwards.
        values (str, optional): Column name for segment sizes. If None,
            counts will be used.
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Create a sunburst chart with the specified hierarchy path
    fig = px.sunburst(
        df,
        path=path,     # List of columns defining the hierarchy levels
        values=values, # Values determining segment sizes
        title="Sunburst Chart"
    )
    
    # Configure layout with clean styling
    fig.update_layout(
        template="plotly_white"
    )
    
    return fig


# =====================================================================
# TIME SERIES VISUALIZATIONS
# =====================================================================

def create_line_chart(df, x, y, color=None):
    """
    Create a line chart for visualizing trends over time or ordered categories.
    
    Line charts connect data points with lines, making them ideal for showing
    trends, changes over time, or continuous data series.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        x (str): Column name for x-axis (typically time or ordered categories).
        y (str): Column name for y-axis values.
        color (str, optional): Column name to group and color by, creating
            multiple lines.
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Create a line chart with markers at data points
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,   # Split into multiple lines by this variable
        markers=True,  # Add markers at actual data points
        title=f"Line Chart of {y} by {x}"
    )
    
    # Configure layout with clean styling
    fig.update_layout(
        template="plotly_white"
    )
    
    return fig


# =====================================================================
# MULTIDIMENSIONAL VISUALIZATIONS
# =====================================================================

def create_3d_scatter(df, x, y, z, color=None, size=None):
    """
    Create a 3D scatter plot to visualize relationships between three variables.
    
    3D scatter plots allow visualization of relationships between three numerical
    variables, with optional additional dimensions through color and size.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        x (str): Column name for x-axis.
        y (str): Column name for y-axis.
        z (str): Column name for z-axis.
        color (str, optional): Column name to color points by (adds a fourth dimension).
        size (str, optional): Column name to scale point sizes by (adds a fifth dimension).
        
    Returns:
        plotly.graph_objects.Figure: An interactive Plotly figure object.
    """
    # Create a 3D scatter plot
    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color=color,  # Color encodes an additional categorical or numerical dimension
        size=size,    # Size encodes an additional numerical dimension
        opacity=0.7,  # Semi-transparent points
        title=f"3D Scatter Plot: {x} vs {y} vs {z}"
    )
    
    # Configure layout with properly labeled 3D axes
    fig.update_layout(
        template="plotly_white",
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z
        )
    )
    
    return fig
