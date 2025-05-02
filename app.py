"""
DataSightPro - Data Analysis and ML Preparation Tool

This application provides a comprehensive interface for:
1. Data exploration and visualization
2. Statistical analysis
3. Data preprocessing for machine learning
4. Dataset export and reporting

The app uses Streamlit for the UI and integrates custom modules for data
processing, analysis, visualization, and preprocessing functionality.
"""

# Standard library imports
import io
import base64
import time
from collections import Counter

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image

# Custom module imports for modular functionality
from utils import detect_data_types, get_summary_statistics, get_missing_values_summary
from analysis import generate_correlation_matrix, analyze_distributions
from visualization import create_histogram, create_scatter_plot, create_box_plot, create_correlation_heatmap
from preprocessing import handle_missing_values, encode_categorical_variables, normalize_data

#------------------------------------------------------------------------------
# APPLICATION CONFIGURATION
#------------------------------------------------------------------------------

# Configure Streamlit page settings
st.set_page_config(
    page_title="DataSightPro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application color scheme (modernized for better UI design)
# Define color variables for consistent use throughout the application
COLORS = {
    "primary": "#4F46E5",     # Indigo 600 - Modern primary blue
    "secondary": "#EC4899",   # Pink 500 - Vibrant secondary
    "accent": "#0EA5E9",      # Sky 500 - Fresh accent
    "success": "#10B981",     # Emerald 500 - Modern green
    "warning": "#F59E0B",     # Amber 500 - Warm warning
    "danger": "#EF4444",      # Red 500 - Clear danger
    "background": "#F9FAFB",  # Gray 50 - Ultra light background
    "text": "#111827",        # Gray 900 - Rich dark text
    "light_text": "#6B7280",  # Gray 500 - Balanced medium gray
    "card_bg": "#FFFFFF",     # White for card backgrounds
    "sidebar_bg": "#F3F4F6",  # Gray 100 - Light sidebar
    "gradient_start": "#4F46E5", # Indigo 600 - Gradient start
    "gradient_end": "#8B5CF6",   # Violet 500 - Gradient end (purple)
}

# Add custom CSS for modern styling
st.markdown("""
<style>
    /* Modern Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        color: """ + COLORS["text"] + """;
        margin-bottom: 1rem;
    }
    
    h1 {
        font-size: 2rem;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: """ + COLORS["sidebar_bg"] + """;
    }
    
    .sidebar .sidebar-content {
        background-color: """ + COLORS["sidebar_bg"] + """;
    }
    
    /* Cards and containers */
    .card {
        background-color: """ + COLORS["card_bg"] + """;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    }
    
    .card:hover {
        box-shadow: 0 10px 20px rgba(0,0,0,0.1), 0 6px 6px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }
    
    /* Modern gradient headers */
    .gradient-header {
        background: linear-gradient(90deg, """ + COLORS["gradient_start"] + """ 0%, """ + COLORS["gradient_end"] + """ 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, """ + COLORS["primary"] + """ 0%, """ + COLORS["gradient_end"] + """ 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
    }
    
    /* Danger button styling for "Remove Dataset and Reset" */
    .stButton > button:contains("Remove Dataset and Reset") {
        background: linear-gradient(90deg, """ + COLORS["danger"] + """ 0%, #F87171 100%);
        border: 1px solid """ + COLORS["danger"] + """;
        box-shadow: 0 2px 5px rgba(239, 68, 68, 0.2);
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transform: translateY(-2px);
        color: white; /* Keep text white on hover for better readability */
    }
    
    /* Danger button hover state */
    .stButton > button:contains("Remove Dataset and Reset"):hover {
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 3px rgba(0,0,0,0.1);
    }
    
    /* Metrics and key stats */
    .metric-container {
        background: """ + COLORS["card_bg"] + """;
        border-left: 4px solid """ + COLORS["primary"] + """;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: """ + COLORS["light_text"] + """;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: """ + COLORS["text"] + """;
        margin: 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 0 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F5F7FA;
        border-radius: 6px 6px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #EDF2F7;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, """ + COLORS["primary"] + """ 0%, """ + COLORS["gradient_end"] + """ 100%);
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Improve dataframe styling */
    .dataframe {
        border-collapse: collapse;
        font-size: 0.9rem;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        width: 100%;
    }
    
    .dataframe th {
        background: linear-gradient(90deg, """ + COLORS["primary"] + """ 0%, """ + COLORS["gradient_end"] + """ 100%);
        color: white;
        font-weight: 500;
        text-align: left;
        padding: 0.9rem 1.2rem;
        border: none;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    
    .dataframe td {
        padding: 0.8rem 1.2rem;
        border-bottom: 1px solid #EDF2F7;
        transition: all 0.2s ease;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    
    .dataframe tr:hover {
        background-color: #EDF2F7;
        transform: scale(1.005);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 2px dashed #E2E8F0;
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: """ + COLORS["primary"] + """;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
        background-color: #F1F5F9;
    }
    
    .dataframe tr:last-child td {
        border-bottom: none;
    }
    
    /* Sidebar header styling */
    .sidebar-header {
        color: """ + COLORS["primary"] + """;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E2E8F0;
        font-size: 1rem;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 1px dashed """ + COLORS["accent"] + """;
        border-radius: 8px;
        padding: 0.75rem;
        background-color: rgba(76, 201, 240, 0.05);
        transition: all 0.2s ease;
    }
    
    .uploadedFile:hover {
        border-color: """ + COLORS["primary"] + """;
        background-color: rgba(76, 201, 240, 0.1);
    }
    
    /* Widget label styling */
    .css-2trqyj, .css-an2lwi, .css-145kmo2, .css-16huue1, .css-10y5sf6 {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: """ + COLORS["light_text"] + """ !important;
        margin-bottom: 0.25rem !important;
    }
    
    /* Alerts and notifications */
    div.stAlert {
        border-radius: 8px !important;
        border: none !important;
        padding: 0.75rem 1rem !important;
        margin: 1rem 0 !important;
    }
    
    div.stAlert > div {
        padding: 0 !important;
    }
    
    div.stAlert[data-baseweb="notification"][data-testid="stNotification"] {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* Success message */
    div.element-container div[data-testid="stNotification"] div[role="alert"][data-baseweb="notification"][kind="success"] {
        background-color: #DCFCE7 !important;
        color: #166534 !important;
        border-left: 4px solid #4ADE80 !important;
    }
    
    /* Error message */
    div.element-container div[data-testid="stNotification"] div[role="alert"][data-baseweb="notification"][kind="error"] {
        background-color: #FEE2E2 !important;
        color: #991B1B !important;
        border-left: 4px solid #EF476F !important;
    }
    
    /* Info message */
    div.element-container div[data-testid="stNotification"] div[role="alert"][data-baseweb="notification"][kind="info"] {
        background-color: #E0F2FE !important;
        color: #0C4A6E !important;
        border-left: 4px solid #4CC9F0 !important;
    }
    
    /* Form elements styling */
    input[type="text"], input[type="number"], input[type="password"], textarea {
        border-radius: 6px !important;
        border-color: #E2E8F0 !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.2s ease !important;
    }
    
    input[type="text"]:focus, input[type="number"]:focus, input[type="password"]:focus, textarea:focus {
        border-color: """ + COLORS["primary"] + """ !important;
        box-shadow: 0 0 0 2px rgba(67, 97, 238, 0.15) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 6px !important;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        border-radius: 6px !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > div {
        margin-bottom: 0.75rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: """ + COLORS["background"] + """ !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: """ + COLORS["sidebar_bg"] + """ !important;
    }
    
    details[open] .streamlit-expanderHeader {
        border-bottom-left-radius: 0 !important;
        border-bottom-right-radius: 0 !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }
    
    details[open] .streamlit-expanderContent {
        border: 1px solid #E2E8F0 !important;
        border-top: none !important;
        border-bottom-left-radius: 6px !important;
        border-bottom-right-radius: 6px !important;
        padding: 1rem !important;
    }
    
    /* Modern scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F1F5F9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

#------------------------------------------------------------------------------
# STATE MANAGEMENT
#------------------------------------------------------------------------------

# Initialize session state variables for persistent data storage across reruns
def initialize_session_state():
    """Initialize all session state variables if they don't already exist"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'data_types' not in st.session_state:
        st.session_state.data_types = None
    if 'summary_stats' not in st.session_state:
        st.session_state.summary_stats = None
    if 'missing_values' not in st.session_state:
        st.session_state.missing_values = None
    if 'correlation_matrix' not in st.session_state:
        st.session_state.correlation_matrix = None
    if 'distribution_analysis' not in st.session_state:
        st.session_state.distribution_analysis = None

# Initialize the session state
initialize_session_state()

#------------------------------------------------------------------------------
# HELPER FUNCTIONS
#------------------------------------------------------------------------------

def load_dataset(uploaded_file):
    """
    Load a dataset from an uploaded file based on its extension
    
    Parameters:
        uploaded_file: The file uploaded through Streamlit's file_uploader
        
    Returns:
        DataFrame: The loaded pandas DataFrame
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        return pd.read_csv(uploaded_file)
    elif file_extension == 'xlsx':
        return pd.read_excel(uploaded_file)
    elif file_extension == 'json':
        return pd.read_json(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def reset_analysis_state():
    """Reset all analysis-related session state variables when a new file is loaded"""
    st.session_state.data_types = None
    st.session_state.summary_stats = None
    st.session_state.missing_values = None
    st.session_state.correlation_matrix = None
    st.session_state.distribution_analysis = None

def custom_alert(message, alert_type="success"):
    """Generate a custom styled alert message
    
    Parameters:
        message (str): The message to display
        alert_type (str): One of "success", "info", "warning", "error"
    """
    icons = {
        "success": "✅",
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌"
    }
    
    colors = {
        "success": {"bg": "#E6F6F1", "border": "#0C9D71", "text": "#0B815D"},
        "info": {"bg": "#E4EEFB", "border": "#3267D6", "text": "#3267D6"},
        "warning": {"bg": "#FFF7E5", "border": "#F9B831", "text": "#B07B12"},
        "error": {"bg": "#FEECEB", "border": "#F5634F", "text": "#D93129"}
    }
    
    style = colors.get(alert_type, colors["info"])
    icon = icons.get(alert_type, "ℹ️")
    
    html = f"""
    <div style="
        background-color: {style['bg']};
        border-left: 4px solid {style['border']};
        color: {style['text']};
        padding: 1rem 1rem 1rem 1rem;
        margin: 1rem 0px;
        border-radius: 6px;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
        animation: fadeIn 0.5s ease;
    ">
        <div style="font-size: 1.2rem; margin-top: -0.1rem;">{icon}</div>
        <div style="font-size: 0.95rem; line-height: 1.5;">{message}</div>
    </div>
    <style>
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """
    
    return st.markdown(html, unsafe_allow_html=True)

def perform_initial_analysis():
    """Perform initial analysis on the loaded dataset to prepare for visualization"""
    # Auto-detect data types
    st.session_state.data_types = detect_data_types(st.session_state.df)
    
    # Generate summary statistics
    st.session_state.summary_stats = get_summary_statistics(st.session_state.df)
    
    # Analyze missing values
    st.session_state.missing_values = get_missing_values_summary(st.session_state.df)
    
    # Generate correlation matrix for numerical columns
    st.session_state.correlation_matrix = generate_correlation_matrix(st.session_state.df)
    
    # Analyze distributions of numerical columns
    st.session_state.distribution_analysis = analyze_distributions(st.session_state.df)

def update_dataframe_and_analysis(new_df):
    """
    Update the session state dataframe and recalculate all analysis data
    
    Parameters:
    -----------
    new_df : pandas.DataFrame
        The new DataFrame to set as the current dataset
    """
    st.session_state.df = new_df
    perform_initial_analysis()  # Recalculate all analysis

def get_download_link(df, file_format, file_name):
    """
    Generate an HTML download link for the DataFrame in the specified format
    
    Parameters:
        df: The pandas DataFrame to be downloaded
        file_format: The export format (CSV, Excel, JSON)
        file_name: The prefix for the downloaded file name
        
    Returns:
        str: HTML anchor tag with the download link
    """
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    if file_format == "CSV":
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        content = buffer.getvalue()
        b64 = base64.b64encode(content.encode()).decode()
        extension = "csv"
        mime_type = "text/csv"
    elif file_format == "Excel":
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        content = buffer.getvalue()
        b64 = base64.b64encode(content).decode()
        extension = "xlsx"
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif file_format == "JSON":
        buffer = io.StringIO()
        df.to_json(buffer, orient="records")
        content = buffer.getvalue()
        b64 = base64.b64encode(content.encode()).decode()
        extension = "json"
        mime_type = "application/json"
    
    filename = f"{file_name}_{timestamp}.{extension}"
    
    # Style the download link as a button to match the "Generate Report" button
    href = f'''
    <a href="data:{mime_type};base64,{b64}"
       download="{filename}"
       style="display: inline-block;
              background: linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["gradient_end"]} 100%);
              color: white;
              border: none;
              border-radius: 6px;
              padding: 0.5rem 1.2rem;
              font-weight: 500;
              text-decoration: none;
              transition: all 0.3s ease;
              box-shadow: 0 2px 5px rgba(0,0,0,0.15);
              margin: 0.25rem 0;">
        Download <span style="color: #FF7F7F; font-weight: 600;">{file_name}</span> as {file_format}
    </a>
    <style>
    a[download]:hover {{
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }}
    
    a[download]:active {{
        transform: translateY(1px);
        box-shadow: 0 2px 3px rgba(0,0,0,0.1);
    }}
    </style>
    '''
    return href

#------------------------------------------------------------------------------
# UI COMPONENTS
#------------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with dataset upload and navigation controls"""
    with st.sidebar:
        # Modern styled app header
        st.markdown("""
            <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 1rem; border-radius: 12px;
                      background: linear-gradient(135deg, """ + COLORS["gradient_start"] + """ 0%, """ + COLORS["gradient_end"] + """ 100%);">
                <h1 style="color: white; margin-bottom: 0.3rem; font-size: 2rem; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">
                    DataSightPro
                </h1>
                <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-top: 0.2rem; font-weight: 500;">
                    Data Analysis & Insights
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Control Panel header - made larger and more prominent
        st.markdown("""
            <div style="margin: 1rem 0;">
                <h2 style="color: """ + COLORS["primary"] + """;
                        font-size: 1.4rem;
                        font-weight: 700;
                        margin: 0.5rem 0;
                        padding-bottom: 0.7rem;
                        border-bottom: 2px solid """ + COLORS["primary"] + """;">
                    📊 Control Panel
                </h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Conditional display based on whether a dataset is loaded
        if st.session_state.df is None:
            # Upload section with styled header - smaller than Control Panel
            st.markdown('<p class="sidebar-header">📤 Upload Dataset</p>', unsafe_allow_html=True)
            
            # File upload with description
            st.markdown("""
                <p style="font-size: 0.85rem; color: """ + COLORS["light_text"] + """;">
                    Upload a CSV, Excel, or JSON file to begin analysis
                </p>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "json"], label_visibility="collapsed")
            
            # Process uploaded file
            if uploaded_file is not None:
                try:
                    # Check if a new file was uploaded and reset state if so
                    if st.session_state.uploaded_file_name != uploaded_file.name:
                        st.session_state.uploaded_file_name = uploaded_file.name
                        reset_analysis_state()
                        
                        # Load the dataset based on file type
                        df = load_dataset(uploaded_file)
                        
                        # Success message with custom styling
                        custom_alert(f"Successfully loaded: {uploaded_file.name}", "success")
                        
                        # Perform initial analysis
                        with st.spinner("Analyzing dataset..."):
                            update_dataframe_and_analysis(df)
                        
                except Exception as e:
                    # Error message with custom styling
                    custom_alert(f"Error: {str(e)}", "error")
        else:
            # Show loaded dataset info
            st.markdown('<p class="sidebar-header">📤 Dataset Loaded</p>', unsafe_allow_html=True)
            
            # Display loaded file info in a card
            st.markdown(f"""
            <div style="
                background-color: {COLORS['card_bg']};
                border-left: 4px solid {COLORS['success']};
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                box-shadow: 0 2px 5px rgba(0,0,0,0.08);">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 0.5rem;">📄</span>
                    <div>
                        <p style="font-weight: 500; margin: 0 0 0.2rem 0;">{st.session_state.uploaded_file_name}</p>
                        <p style="color: {COLORS['light_text']}; font-size: 0.8rem; margin: 0;">
                            {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a reset button
            if st.button("Remove Dataset and Reset", use_container_width=True):
                # Reset all session state variables
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                initialize_session_state()
                st.rerun()
        
        # Display navigation options when data is loaded
        if st.session_state.df is not None:
            # Navigation section with icons and modern styling
            st.markdown('<p class="sidebar-header">🧭 Navigation</p>', unsafe_allow_html=True)
            
            analysis_options = [
                "📊 Dataset Overview",
                "📈 Statistical Analysis",
                "🔍 Visualization",
                "⚙️ Data Preprocessing",
                "💾 Export Options"
            ]
            
            option_mapping = {
                "📊 Dataset Overview": "Dataset Overview",
                "📈 Statistical Analysis": "Statistical Analysis",
                "🔍 Visualization": "Visualization",
                "⚙️ Data Preprocessing": "Data Preprocessing",
                "💾 Export Options": "Export Options"
            }
            
            selected_option = st.radio("Navigation Options", analysis_options, label_visibility="collapsed")
            
            # Display basic dataset metrics with styled containers
            st.markdown('<p class="sidebar-header">📋 Dataset Info</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-label">Rows</p>
                        <p class="metric-value">{st.session_state.df.shape[0]:,}</p>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-container">
                        <p class="metric-label">Columns</p>
                        <p class="metric-value">{st.session_state.df.shape[1]}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Data types summary
            if st.session_state.data_types:
                # Use the same classification logic as in the Dataset Overview
                type_counts = {"numerical": 0, "categorical": 0, "datetime": 0, "boolean": 0}
                
                for dtype in st.session_state.data_types.values():
                    if dtype == 'Binary':
                        type_counts["boolean"] += 1
                    elif dtype in ['Discrete', 'Continuous']:
                        type_counts["numerical"] += 1
                    elif dtype in ['Categorical', 'Text']:
                        type_counts["categorical"] += 1
                    elif dtype == 'DateTime':
                        type_counts["datetime"] += 1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div class="metric-container" style="border-left-color: #0066CC">
                            <p class="metric-label">Numerical</p>
                            <p class="metric-value">{type_counts["numerical"]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="metric-container" style="border-left-color: #FF9900">
                            <p class="metric-label">Categorical</p>
                            <p class="metric-value">{type_counts["categorical"]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Add boolean and datetime if they exist
                if type_counts["boolean"] > 0 or type_counts["datetime"] > 0:
                    col1, col2 = st.columns(2)
                    
                    if type_counts["boolean"] > 0:
                        with col1:
                            st.markdown(f"""
                                <div class="metric-container" style="border-left-color: #33CC99">
                                    <p class="metric-label">Boolean</p>
                                    <p class="metric-value">{type_counts["boolean"]}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    if type_counts["datetime"] > 0:
                        with col2 if type_counts["boolean"] > 0 else col1:
                            st.markdown(f"""
                                <div class="metric-container" style="border-left-color: #6633CC">
                                    <p class="metric-label">Datetime</p>
                                    <p class="metric-value">{type_counts["datetime"]}</p>
                                </div>
                            """, unsafe_allow_html=True)
            
            return option_mapping[selected_option]
    
    return None

def render_dataset_overview():
    """Render the dataset overview section with data preview and summary info"""
    # Modern gradient header with icon and description
    st.markdown(f"""
        <div class="gradient-header">
            <h1 style="margin: 0; color: white; font-size: 2.2rem; font-weight: 600;">
                <span style="font-size: 2.2rem; margin-right: 0.5rem;">📊</span> Dataset Overview
            </h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem; margin-bottom: 0; font-size: 1.1rem;">
                Exploring key characteristics and structure of your dataset
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Dataset shape card
    st.markdown(f"""
        <div style="background-color: white; border-radius: 8px; padding: 1.2rem;
                  box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                <div style="flex: 1; min-width: 150px;">
                    <p style="color: {COLORS['light_text']}; font-size: 0.9rem; margin: 0;">DATASET NAME</p>
                    <p style="font-size: 1.1rem; font-weight: 600; margin: 0.3rem 0 0 0;">{st.session_state.uploaded_file_name}</p>
                </div>
                <div style="flex: 1; min-width: 150px;">
                    <p style="color: {COLORS['light_text']}; font-size: 0.9rem; margin: 0;">ROWS</p>
                    <p style="font-size: 1.1rem; font-weight: 600; margin: 0.3rem 0 0 0;">{st.session_state.df.shape[0]:,}</p>
                </div>
                <div style="flex: 1; min-width: 150px;">
                    <p style="color: {COLORS['light_text']}; font-size: 0.9rem; margin: 0;">COLUMNS</p>
                    <p style="font-size: 1.1rem; font-weight: 600; margin: 0.3rem 0 0 0;">{st.session_state.df.shape[1]}</p>
                </div>
                <div style="flex: 1; min-width: 150px;">
                    <p style="color: {COLORS['light_text']}; font-size: 0.9rem; margin: 0;">MEMORY USAGE</p>
                    <p style="font-size: 1.1rem; font-weight: 600; margin: 0.3rem 0 0 0;">{st.session_state.df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Create a two-column layout for main content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Data preview with modern styling
        st.markdown(f"""
            <h2 style="font-size: 1.4rem; margin-bottom: 0.7rem; color: {COLORS['text']};">
                <span style="color: {COLORS['primary']}; margin-right: 0.5rem;">▶</span> Data Preview
            </h2>
        """, unsafe_allow_html=True)
        
        # Add a toggle to show more rows
        show_more = st.checkbox("Show more rows", value=False)
        preview_rows = 15 if show_more else 6
        
        # Use st.dataframe with styling
        st.dataframe(
            st.session_state.df.head(preview_rows),
            height=min(35 * preview_rows + 38, 500),
            use_container_width=True
        )
        
        # Data types section with improved visual design
        st.markdown(f"""
            <h2 style="font-size: 1.4rem; margin: 1.5rem 0 0.7rem 0; color: {COLORS['text']};">
                <span style="color: {COLORS['primary']}; margin-right: 0.5rem;">🔍</span> Data Types
            </h2>
        """, unsafe_allow_html=True)
        
        # Create a more visually appealing data types display
        data_types_df = pd.DataFrame({
            'Column': st.session_state.data_types.keys(),
            'Type': st.session_state.data_types.values()
        })
        
        # Count data types for summary display
        type_counts = {}
        for dtype in data_types_df['Type']:
            base_type = dtype
            if dtype == 'Binary':
                base_type = 'boolean'
            elif dtype in ['Discrete', 'Continuous']:
                base_type = 'numerical'
            elif dtype in ['Categorical', 'Text']:
                base_type = 'categorical'
            elif dtype == 'DateTime':
                base_type = 'datetime'
                
            type_counts[base_type] = type_counts.get(base_type, 0) + 1
            
        # Display type summary before the detailed table
        st.markdown("""
            <div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 0.7rem;">
        """, unsafe_allow_html=True)
        
        for dtype, count in type_counts.items():
            color = {
                'numerical': "#0066CC",
                'categorical': "#FF9900",
                'datetime': "#6633CC",
                'boolean': "#33CC99"
            }.get(dtype, "#999999")
            
            st.markdown(f"""
                <div style="flex: 1; min-width: 120px; padding: 0.6rem; border-radius: 6px;
                          background-color: white; border-left: 3px solid {color};">
                    <p style="margin: 0; color: {COLORS['light_text']}; font-size: 0.8rem;">{dtype.upper()}</p>
                    <p style="margin: 0; font-weight: 600; font-size: 1.1rem;">{count}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display the detailed data types table
        st.dataframe(data_types_df, use_container_width=True)
        
    with col2:
        # Missing values section with improved visualization
        st.markdown(f"""
            <h2 style="font-size: 1.4rem; margin-bottom: 0.7rem; color: {COLORS['text']};">
                <span style="color: {COLORS['primary']}; margin-right: 0.5rem;">📉</span> Missing Values
            </h2>
        """, unsafe_allow_html=True)
        
        if st.session_state.missing_values is not None:
            # Calculate overall missing values percentage
            total_cells = st.session_state.df.shape[0] * st.session_state.df.shape[1]
            missing_cells = st.session_state.df.isna().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            
            # Create a visual indicator for overall completeness
            st.markdown(f"""
                <div style="background-color: white; border-radius: 8px; padding: 1rem;
                          box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                    <p style="color: {COLORS['light_text']}; font-size: 0.9rem; margin: 0;">DATA COMPLETENESS</p>
                    <div style="display: flex; align-items: center; gap: 1rem; margin-top: 0.5rem;">
                        <div style="flex-grow: 1; background-color: #EAECEF; height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background-color: {COLORS['primary']}; width: {100 - missing_percentage}%; height: 100%;"></div>
                        </div>
                        <p style="margin: 0; font-weight: 600; white-space: nowrap;">{100 - missing_percentage:.1f}% Complete</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display missing values by column
            columns_with_missing = [col for col, pct in st.session_state.missing_values.items() if pct > 0]
            
            if columns_with_missing:
                # Prepare missing values data for more appealing visualization
                missing_data = pd.DataFrame({
                    'Column': columns_with_missing,
                    'Percentage': [st.session_state.missing_values[col] for col in columns_with_missing]
                }).sort_values('Percentage', ascending=False)
                
                # Use Plotly for a better visualization
                if not missing_data.empty:
                    fig = px.bar(
                        missing_data,
                        x='Percentage',
                        y='Column',
                        orientation='h',
                        title="Columns with Missing Values",
                        color='Percentage',
                        color_continuous_scale=['#8dd3c7', '#fb8072', '#e41a1c'],
                        range_color=[0, 100]
                    )
                    
                    fig.update_layout(
                        height=max(250, min(40 * len(columns_with_missing), 500)),
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Missing Values (%)",
                        yaxis_title="",
                        coloraxis_showscale=False,
                        font=dict(family="Arial, sans-serif"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed missing values table
                    missing_vals_df = pd.DataFrame({
                        'Column': missing_data['Column'].values,
                        'Missing Count': [int(st.session_state.df[col].isna().sum()) for col in missing_data['Column'].values],
                        'Missing (%)': [f"{val:.2f}%" for val in missing_data['Percentage'].values]
                    })
                    st.dataframe(missing_vals_df, use_container_width=True, height=min(35 * len(missing_vals_df) + 38, 300))
            else:
                st.success("No missing values detected in the dataset! 🎉")
        
        # Add a quick summary section
        st.markdown(f"""
            <h2 style="font-size: 1.4rem; margin: 1.5rem 0 0.7rem 0; color: {COLORS['text']};">
                <span style="color: {COLORS['primary']}; margin-right: 0.5rem;">📝</span> Quick Summary
            </h2>
            
            <div style="background-color: white; border-radius: 8px; padding: 1rem;
                      box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <ul style="margin: 0; padding-left: 1.5rem;">
                    <li style="margin-bottom: 0.5rem;">Dataset contains <b>{st.session_state.df.shape[0]:,}</b> rows and <b>{st.session_state.df.shape[1]}</b> columns</li>
                    <li style="margin-bottom: 0.5rem;"><b>{len([col for col, pct in st.session_state.missing_values.items() if pct > 0])}</b> columns have missing values</li>
                    <li style="margin-bottom: 0.5rem;"><b>{sum(1 for dtype in st.session_state.data_types.values() if dtype in ['Discrete', 'Continuous'])}</b> numerical features available</li>
                    <li>Data can be explored further in the Statistical Analysis and Visualization sections</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

def render_statistical_analysis():
    """Render the statistical analysis section with summary stats, correlation, and distribution analysis"""
    # Modern gradient header with icon
    st.markdown(f"""
        <div class="gradient-header">
            <h1 style="margin: 0; color: white; font-size: 2.2rem; font-weight: 600;">
                <span style="font-size: 2.2rem; margin-right: 0.5rem;">📈</span> Statistical Analysis
            </h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem; margin-bottom: 0; font-size: 1.1rem;">
                Exploring numerical patterns and relationships in your data
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different statistical analyses with improved styling
    tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Correlation Analysis", "Distribution Analysis"])
    
    # Tab 1: Summary Statistics
    with tab1:
        if st.session_state.summary_stats is not None:
            st.subheader("Summary Statistics")
            st.dataframe(st.session_state.summary_stats)
            
            # Detailed statistics for a selected column
            numerical_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            if numerical_cols:
                selected_column = st.selectbox("Select column for detailed statistics", numerical_cols)
                st.subheader(f"Detailed Statistics for {selected_column}")
                
                # Display key metrics in a three-column layout
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{st.session_state.df[selected_column].mean():.2f}")
                    st.metric("Min", f"{st.session_state.df[selected_column].min():.2f}")
                with col2:
                    st.metric("Median", f"{st.session_state.df[selected_column].median():.2f}")
                    st.metric("Max", f"{st.session_state.df[selected_column].max():.2f}")
                with col3:
                    st.metric("Std Dev", f"{st.session_state.df[selected_column].std():.2f}")
                    st.metric("Count", f"{st.session_state.df[selected_column].count()}")
                    
                # Display histogram for the selected column
                st.subheader(f"Distribution of {selected_column}")
                fig = create_histogram(st.session_state.df, selected_column)
                st.plotly_chart(fig, use_container_width=True, key=f"summary_hist_{selected_column}")
    
    # Tab 2: Correlation Analysis
    with tab2:
        st.subheader("Correlation Analysis")
        if st.session_state.correlation_matrix is not None:
            
            # Display correlation matrix as a table
            st.write("Correlation Matrix")
            st.dataframe(st.session_state.correlation_matrix)
            
            # Display interactive correlation heatmap
            st.subheader("Correlation Heatmap")
            fig = create_correlation_heatmap(st.session_state.correlation_matrix)
            st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")
            
            # Display strongest correlations in the dataset
            st.subheader("Strongest Correlations")
            corr_unstack = st.session_state.correlation_matrix.unstack()
            corr_unstack = corr_unstack.sort_values(ascending=False)
            # Remove self-correlations (always 1.0)
            corr_unstack = corr_unstack[corr_unstack < 1.0]
            highest_corrs = corr_unstack.head(10)
            st.write(highest_corrs)
    
    # Tab 3: Distribution Analysis
    with tab3:
        st.subheader("Distribution Analysis")
        
        # Get numerical and categorical columns
        numerical_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = st.session_state.df.select_dtypes(exclude=['number']).columns.tolist()
        
        # Analysis for numerical variables
        if numerical_cols:
            st.write("Distribution of Numerical Variables")
            
            # Multi-select for columns to analyze
            selected_columns = st.multiselect(
                "Select columns to analyze",
                options=numerical_cols,
                default=numerical_cols[:min(3, len(numerical_cols))]
            )
            
            if selected_columns:
                chart_type = st.radio("Select chart type", ["Histogram", "Box Plot"])
                
                if chart_type == "Histogram":
                    # Generate individual histograms for each selected column
                    for i, col in enumerate(selected_columns):
                        st.subheader(f"Histogram: {col}")
                        fig = create_histogram(st.session_state.df, col)
                        st.plotly_chart(fig, use_container_width=True, key=f"dist_hist_{i}")
                
                elif chart_type == "Box Plot":
                    # Generate combined box plot for all selected columns
                    st.subheader("Box Plots")
                    fig = create_box_plot(st.session_state.df, selected_columns)
                    st.plotly_chart(fig, use_container_width=True, key="dist_box_plot")
        
        # Analysis for categorical variables
        if categorical_cols:
            st.write("Distribution of Categorical Variables")
            selected_cat_col = st.selectbox(
                "Select categorical column",
                options=categorical_cols
            )
            
            if selected_cat_col:
                # Display value counts as a bar chart
                value_counts = st.session_state.df[selected_cat_col].value_counts()
                st.subheader(f"Value counts for {selected_cat_col}")
                st.bar_chart(value_counts)

def render_visualization():
    """Render the visualization section with interactive data visualization options"""
    # Modern gradient header with icon
    st.markdown(f"""
        <div class="gradient-header">
            <h1 style="margin: 0; color: white; font-size: 2.2rem; font-weight: 600;">
                <span style="font-size: 2.2rem; margin-right: 0.5rem;">🔍</span> Interactive Visualizations
            </h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem; margin-bottom: 0; font-size: 1.1rem;">
                Creating insightful charts and visual representations
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["Scatter Plots", "Distribution Plots", "Custom Plots"])
    
    # Tab 1: Scatter Plots
    with tab1:
        st.subheader("Scatter Plot Analysis")
        
        numerical_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numerical_cols) >= 2:
            # Select X and Y axes in a two-column layout
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Select X-axis", options=numerical_cols, key="scatter_x")
            
            with col2:
                y_axis = st.selectbox("Select Y-axis", 
                                     options=[c for c in numerical_cols if c != x_axis], 
                                     key="scatter_y")
            
            # Optional color grouping for points
            all_cols = st.session_state.df.columns.tolist()
            color_by = st.selectbox("Color points by (optional)", 
                                  options=["None"] + all_cols,
                                  key="scatter_color")
            
            color_col = None if color_by == "None" else color_by
            
            # Generate and display the scatter plot
            st.subheader(f"Scatter Plot: {x_axis} vs {y_axis}")
            fig = create_scatter_plot(st.session_state.df, x_axis, y_axis, color_column=color_col)
            st.plotly_chart(fig, use_container_width=True, key="viz_scatter_plot")
            
            # Display correlation between the selected variables
            if x_axis in numerical_cols and y_axis in numerical_cols:
                correlation = st.session_state.df[[x_axis, y_axis]].corr().iloc[0, 1]
                st.info(f"Correlation between {x_axis} and {y_axis}: {correlation:.4f}")
        else:
            st.warning("Need at least two numerical columns for scatter plot visualization.")
    
    # Tab 2: Distribution Plots
    with tab2:
        st.subheader("Distribution Plots")
        
        plot_type = st.radio(
            "Select Plot Type",
            ["Histogram", "Box Plot", "Violin Plot"]
        )
        
        numerical_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if numerical_cols:
            selected_cols = st.multiselect(
                "Select columns to visualize",
                options=numerical_cols,
                default=numerical_cols[0]
            )
            
            if selected_cols:
                if plot_type == "Histogram":
                    # Generate individual histograms
                    for i, col in enumerate(selected_cols):
                        st.subheader(f"Histogram: {col}")
                        fig = create_histogram(st.session_state.df, col)
                        st.plotly_chart(fig, use_container_width=True, key=f"viz_hist_{i}")
                
                elif plot_type == "Box Plot":
                    # Generate combined box plot
                    st.subheader("Box Plot")
                    fig = create_box_plot(st.session_state.df, selected_cols)
                    st.plotly_chart(fig, use_container_width=True, key="viz_box_plot")
                
                elif plot_type == "Violin Plot":
                    st.subheader("Violin Plot")
                    
                    # Create a melted dataframe for the violin plot
                    plot_data = st.session_state.df[selected_cols].melt(var_name='Variable', value_name='Value')
                    fig = px.violin(plot_data, x='Variable', y='Value', box=True)
                    st.plotly_chart(fig, use_container_width=True, key="viz_violin_plot")
    
    # Tab 3: Custom Plots
    with tab3:
        st.subheader("Custom Visualizations")
        
        custom_plot_type = st.selectbox(
            "Select Plot Type",
            ["Correlation Heatmap", "Pair Plot", "3D Scatter Plot"]
        )
        
        if custom_plot_type == "Correlation Heatmap":
            st.write("Correlation Heatmap")
            
            if st.session_state.correlation_matrix is not None:
                # Filter correlation matrix based on threshold
                min_corr = st.slider("Minimum correlation strength", 0.0, 1.0, 0.1, 0.05)
                
                filtered_corr = st.session_state.correlation_matrix.copy()
                filtered_corr = filtered_corr[abs(filtered_corr) >= min_corr]
                
                fig = create_correlation_heatmap(filtered_corr)
                st.plotly_chart(fig, use_container_width=True, key="custom_corr_heatmap")
        
        elif custom_plot_type == "Pair Plot":
            st.write("Pair Plot (Scatter Matrix)")
            
            numerical_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numerical_cols) >= 2:
                selected_cols = st.multiselect(
                    "Select columns for pair plot (2-4 recommended)",
                    options=numerical_cols,
                    default=numerical_cols[:min(3, len(numerical_cols))]
                )
                
                if len(selected_cols) >= 2:
                    # Generate pair plot (scatter matrix)
                    fig = px.scatter_matrix(
                        st.session_state.df[selected_cols],
                        dimensions=selected_cols,
                        title="Pair Plot"
                    )
                    fig.update_layout(height=800)
                    st.plotly_chart(fig, use_container_width=True, key="pair_plot")
                else:
                    st.warning("Please select at least 2 columns")
            else:
                st.warning("Need at least two numerical columns for pair plot.")
        
        elif custom_plot_type == "3D Scatter Plot":
            st.write("3D Scatter Plot")
            
            numerical_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numerical_cols) >= 3:
                # Select X, Y, and Z axes
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_axis = st.selectbox("X-axis", options=numerical_cols, key="3d_x")
                
                with col2:
                    y_axis = st.selectbox("Y-axis", 
                                        options=[c for c in numerical_cols if c != x_axis], 
                                        key="3d_y")
                
                with col3:
                    z_axis = st.selectbox("Z-axis", 
                                       options=[c for c in numerical_cols if c != x_axis and c != y_axis], 
                                       key="3d_z")
                
                # Optional color dimension
                color_by = st.selectbox("Color points by (optional)", 
                                      options=["None"] + st.session_state.df.columns.tolist(),
                                      key="3d_color")
                
                # Generate 3D scatter plot
                if color_by == "None":
                    fig = px.scatter_3d(
                        st.session_state.df, 
                        x=x_axis, 
                        y=y_axis, 
                        z=z_axis,
                        title=f"3D Scatter Plot: {x_axis} vs {y_axis} vs {z_axis}"
                    )
                else:
                    fig = px.scatter_3d(
                        st.session_state.df, 
                        x=x_axis, 
                        y=y_axis, 
                        z=z_axis,
                        color=color_by,
                        title=f"3D Scatter Plot: {x_axis} vs {y_axis} vs {z_axis}, colored by {color_by}"
                    )
                
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True, key="scatter_3d_plot")
            else:
                st.warning("Need at least three numerical columns for 3D scatter plot.")

def render_data_preprocessing():
    """Render the data preprocessing section with options for preparing data for ML"""
    # Modern gradient header with icon
    st.markdown(f"""
        <div class="gradient-header">
            <h1 style="margin: 0; color: white; font-size: 2.2rem; font-weight: 600;">
                <span style="font-size: 2.2rem; margin-right: 0.5rem;">⚙️</span> Data Preprocessing
            </h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem; margin-bottom: 0; font-size: 1.1rem;">
                Prepare your dataset for machine learning and further analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different preprocessing steps
    tab1, tab2, tab3, tab4 = st.tabs([
        "Missing Value Handling", 
        "Categorical Encoding", 
        "Feature Scaling", 
        "Feature Selection"
    ])
    
    # Tab 1: Missing Value Handling
    with tab1:
        st.subheader("Missing Value Handling")
        
        # Show columns with missing values
        missing_cols = [col for col, pct in st.session_state.missing_values.items() if pct > 0]
        
        if missing_cols:
            st.write("Columns with missing values:")
            for col in missing_cols:
                st.write(f"- {col}: {st.session_state.missing_values[col]:.2f}% missing")
            
            # Select handling strategy
            handling_strategy = st.selectbox(
                "Select handling strategy",
                ["Remove rows with missing values", 
                 "Fill missing values with mean/mode", 
                 "Fill missing values with median",
                 "Fill missing values with constant"]
            )
            
            # Apply selected strategy
            if handling_strategy == "Remove rows with missing values":
                if st.button("Apply - Remove Rows"):
                    with st.spinner("Processing..."):
                        cleaned_df = st.session_state.df.dropna()
                        before_shape = st.session_state.df.shape[0]
                        after_shape = cleaned_df.shape[0]
                        
                        update_dataframe_and_analysis(cleaned_df)
                        st.success(f"Removed {before_shape - after_shape} rows with missing values")
                        st.rerun()
            
            elif handling_strategy == "Fill missing values with mean/mode":
                if st.button("Apply - Fill with Mean/Mode"):
                    with st.spinner("Processing..."):
                        processed_df = handle_missing_values(
                            st.session_state.df, 
                            strategy="mean/mode"
                        )
                        update_dataframe_and_analysis(processed_df)
                        st.success("Missing values filled with mean/mode")
                        st.rerun()
            
            elif handling_strategy == "Fill missing values with median":
                if st.button("Apply - Fill with Median"):
                    with st.spinner("Processing..."):
                        processed_df = handle_missing_values(
                            st.session_state.df, 
                            strategy="median"
                        )
                        update_dataframe_and_analysis(processed_df)
                        st.success("Missing values filled with median")
                        st.rerun()
            
            elif handling_strategy == "Fill missing values with constant":
                fill_value = st.text_input("Fill value", "0")
                
                if st.button("Apply - Fill with Constant"):
                    with st.spinner("Processing..."):
                        try:
                            # Try to convert to numeric if possible
                            fill_val = float(fill_value)
                        except ValueError:
                            fill_val = fill_value
                        
                        processed_df = handle_missing_values(
                            st.session_state.df, 
                            strategy="constant",
                            fill_value=fill_val
                        )
                        update_dataframe_and_analysis(processed_df)
                        st.success(f"Missing values filled with '{fill_value}'")
                        st.rerun()
        else:
            st.info("No missing values in the dataset.")
    
    # Tab 2: Categorical Encoding
    with tab2:
        st.subheader("Categorical Encoding")
        
        categorical_cols = st.session_state.df.select_dtypes(exclude=['number']).columns.tolist()
        
        if categorical_cols:
            st.write("Categorical columns detected:")
            for col in categorical_cols:
                unique_values = st.session_state.df[col].nunique()
                st.write(f"- {col}: {unique_values} unique values")
            
            # Select columns to encode
            cols_to_encode = st.multiselect(
                "Select columns to encode",
                options=categorical_cols,
                default=categorical_cols
            )
            
            if cols_to_encode:
                encoding_method = st.selectbox(
                    "Select encoding method",
                    ["One-Hot Encoding", "Label Encoding"]
                )
                
                if st.button("Apply Encoding"):
                    with st.spinner("Encoding categorical variables..."):
                        processed_df = encode_categorical_variables(
                            st.session_state.df,
                            columns=cols_to_encode,
                            method=encoding_method.lower().replace("-", "_").replace(" ", "_")
                        )
                        
                        update_dataframe_and_analysis(processed_df)
                        st.success(f"Applied {encoding_method} to selected columns")
                        st.rerun()
        else:
            st.info("No categorical columns detected in the dataset.")
    
    # Tab 3: Feature Scaling
    with tab3:
        st.subheader("Feature Scaling")
        
        numerical_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if numerical_cols:
            # Select columns to scale
            cols_to_scale = st.multiselect(
                "Select columns to scale",
                options=numerical_cols,
                default=numerical_cols
            )
            
            if cols_to_scale:
                scaling_method = st.selectbox(
                    "Select scaling method",
                    ["Min-Max Scaling (0-1)", "Standardization (z-score)"]
                )
                
                if st.button("Apply Scaling"):
                    with st.spinner("Scaling features..."):
                        method = "minmax" if scaling_method == "Min-Max Scaling (0-1)" else "standard"
                        
                        processed_df = normalize_data(
                            st.session_state.df,
                            columns=cols_to_scale,
                            method=method
                        )
                        
                        update_dataframe_and_analysis(processed_df)
                        st.success(f"Applied {scaling_method} to selected columns")
                        st.rerun()
        else:
            st.info("No numerical columns detected for scaling.")
    
    # Tab 4: Feature Selection
    with tab4:
        st.subheader("Feature Selection")
        
        numerical_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numerical_cols) > 1:
            st.write("Select feature selection method:")
            
            selection_method = st.radio(
                "Method",
                ["Correlation-based Selection", "Manual Selection"]
            )
            
            if selection_method == "Correlation-based Selection":
                if st.session_state.correlation_matrix is not None:
                    # Select correlation threshold
                    correlation_threshold = st.slider(
                        "Correlation threshold",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.8,
                        step=0.05
                    )
                    
                    # Find highly correlated feature pairs
                    corr_matrix = st.session_state.correlation_matrix.abs()
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr_pairs = [(col1, col2) for col1 in upper_tri.columns for col2 in upper_tri.columns 
                                    if upper_tri.loc[col1, col2] > correlation_threshold]
                    
                    if high_corr_pairs:
                        # Display highly correlated pairs
                        st.write("Highly correlated feature pairs:")
                        for col1, col2 in high_corr_pairs:
                            corr_value = corr_matrix.loc[col1, col2]
                            st.write(f"- {col1} & {col2}: {corr_value:.4f}")
                        
                        # Suggest features to remove based on correlation frequency
                        features_to_consider = list(set([col for pair in high_corr_pairs for col in pair]))
                        
                        feature_counts = Counter([col for pair in high_corr_pairs for col in pair])
                        
                        st.write("Suggested features to remove (highest correlation counts):")
                        for feature, count in feature_counts.most_common():
                            st.write(f"- {feature}: appears in {count} high correlation pairs")
                        
                        features_to_remove = st.multiselect(
                            "Select features to remove",
                            options=features_to_consider
                        )
                        
                        if features_to_remove and st.button("Remove Selected Features"):
                            with st.spinner("Removing features..."):
                                updated_df = st.session_state.df.drop(columns=features_to_remove)
                                update_dataframe_and_analysis(updated_df)
                                st.success(f"Removed {len(features_to_remove)} features")
                                st.rerun()
                    else:
                        st.info(f"No feature pairs with correlation above {correlation_threshold} found.")
            
            elif selection_method == "Manual Selection":
                # Manual column selection
                all_columns = st.session_state.df.columns.tolist()
                
                columns_to_keep = st.multiselect(
                    "Select columns to keep",
                    options=all_columns,
                    default=all_columns
                )
                
                if len(columns_to_keep) < len(all_columns) and st.button("Keep Selected Columns"):
                    with st.spinner("Updating dataset..."):
                        updated_df = st.session_state.df[columns_to_keep]
                        update_dataframe_and_analysis(updated_df)
                        st.success(f"Dataset updated to include only {len(columns_to_keep)} selected columns")
                        st.rerun()
        else:
            st.info("Need multiple numerical columns for feature selection.")

def render_export_options():
    """Render the export options section with dataset and report export functionality"""
    # Modern gradient header with icon
    st.markdown(f"""
        <div class="gradient-header">
            <h1 style="margin: 0; color: white; font-size: 2.2rem; font-weight: 600;">
                <span style="font-size: 2.2rem; margin-right: 0.5rem;">💾</span> Export Options
            </h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem; margin-bottom: 0; font-size: 1.1rem;">
                Download your processed data and analysis reports
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        # Select export format
        export_format = st.radio(
            "Export Format",
            ["CSV", "Excel", "JSON"]
        )
        
        # Download processed dataset
        st.subheader("Download Processed Dataset")
        download_name = st.text_input("File name prefix", "processed_data")
        st.markdown(get_download_link(st.session_state.df, export_format, download_name), unsafe_allow_html=True)
        
        # Generate analysis report
        st.subheader("Generate Analysis Report")
        include_overview = st.checkbox("Include Dataset Overview", value=True)
        include_stats = st.checkbox("Include Statistical Analysis", value=True)
        include_viz_data = st.checkbox("Include Visualization Data", value=True)
        
        # Custom styling for the Generate Report button to change color when clicked
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #4F46E5 0%, #8B5CF6 100%);
            color: white;
        }
        div.stButton > button:hover {
            color: #FFD700; /* Gold color on hover */
        }
        div.stButton > button:active {
            color: #FFD700 !important; /* Gold color when clicked */
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Standard Streamlit button for Generate Report - always keep the same name
        if st.button("Generate Report"):
            # Initialize report sections
            report_sections = []
            
            if include_overview:
                # Add dataset info
                report_sections.append(pd.DataFrame({
                    'Report Section': ['Dataset Overview'],
                    'Details': [f"Rows: {st.session_state.df.shape[0]}, Columns: {st.session_state.df.shape[1]}"]
                }))
                
                # Add data types
                if st.session_state.data_types:
                    dt_df = pd.DataFrame({
                        'Column': list(st.session_state.data_types.keys()),
                        'Data Type': list(st.session_state.data_types.values())
                    })
                    dt_df['Report Section'] = 'Data Types'
                    report_sections.append(dt_df)
                
                # Add missing values
                if st.session_state.missing_values:
                    mv_df = pd.DataFrame({
                        'Column': list(st.session_state.missing_values.keys()),
                        'Missing (%)': list(st.session_state.missing_values.values())
                    })
                    mv_df['Report Section'] = 'Missing Values'
                    report_sections.append(mv_df)
            
            if include_stats and st.session_state.summary_stats is not None:
                # Add summary statistics
                stats_df = st.session_state.summary_stats.reset_index()
                stats_df.columns = ['Statistic' if col == 'index' else col for col in stats_df.columns]
                stats_df['Report Section'] = 'Summary Statistics'
                report_sections.append(stats_df)
            
            if include_viz_data and st.session_state.correlation_matrix is not None:
                # Add correlation data
                corr_df = st.session_state.correlation_matrix.unstack().reset_index()
                corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
                corr_df = corr_df[corr_df['Feature 1'] != corr_df['Feature 2']]  # Remove self-correlations
                corr_df = corr_df.sort_values('Correlation', ascending=False)
                corr_df['Report Section'] = 'Correlations'
                report_sections.append(corr_df)
            
            # Combine all report sections
            if report_sections:
                report_df = pd.concat(report_sections, ignore_index=True)
                
                # Generate the CSV data
                buffer = io.StringIO()
                report_df.to_csv(buffer, index=False)
                content = buffer.getvalue()
                b64 = base64.b64encode(content.encode()).decode()
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"analysis_report_{timestamp}.csv"
                
                # Display success message
                st.success("Report generated successfully!")
                
                # Create a visible styled download link for the report (similar to get_download_link function)
                href = f'''
                <a href="data:text/csv;base64,{b64}"
                   download="{filename}"
                   style="display: inline-block;
                          background: linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["gradient_end"]} 100%);
                          color: white;
                          border: none;
                          border-radius: 6px;
                          padding: 0.5rem 1.2rem;
                          font-weight: 500;
                          text-decoration: none;
                          transition: all 0.3s ease;
                          box-shadow: 0 2px 5px rgba(0,0,0,0.15);
                          margin: 0.25rem 0;">
                    Download Analysis Report
                </a>
                <style>
                a[download]:hover {{
                    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                    transform: translateY(-2px);
                }}
                
                a[download]:active {{
                    transform: translateY(1px);
                    box-shadow: 0 2px 3px rgba(0,0,0,0.1);
                }}
                </style>
                '''
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No report sections selected.")

def render_welcome_screen():
    """Render the welcome screen when no dataset is loaded"""
    # Hero section with modern styling
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; color: {COLORS['primary']}; margin-bottom: 0.5rem;">
            Welcome to DataSightPro
        </h1>
        <p style="font-size: 1.2rem; color: {COLORS['text']}; max-width: 800px; margin: 0 auto 1.5rem auto;">
            A comprehensive platform for data analysis and machine learning preparation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards in a modern grid layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">📊</span> Data Exploration
            </h3>
            <p>
                Quickly understand your dataset with automatic profiling, data type detection,
                and missing value analysis. Gain insights into your data's structure and quality.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">🔍</span> Advanced Analytics
            </h3>
            <p>
                Perform statistical analysis, correlation studies, and distribution examinations
                with beautiful visualizations and detailed metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">📈</span> Interactive Visualization
            </h3>
            <p>
                Create beautiful, interactive charts with Plotly to uncover patterns and relationships
                in your data. Customize visualizations for your specific needs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">⚙️</span> ML Preparation
            </h3>
            <p>
                Preprocess your data with encoding, scaling, and feature engineering tools
                designed for machine learning workflows. Export processed data in multiple formats.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Custom CSS for better section alignment
    st.markdown("""
    <style>
        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            height: 42px;
            line-height: 42px;
            display: flex;
            align-items: center;
        }
        
        .equal-height-card {
            min-height: 400px;
            display: flex;
            flex-direction: column;
        }
        
        .format-card {
            min-height: 230px;
            margin-bottom: 1.5rem;
        }
        
        .sample-card {
            min-height: 120px;
        }
        
        .button-container {
            display: flex;
            justify-content: center;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Getting started section - all steps in one card with consistent height
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Getting Started</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card equal-height-card">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">🚀</span> Follow These Steps
            </h3>
            <ol style="margin-top: 1rem; padding-left: 1.5rem;">
                <li style="margin-bottom: 0.8rem; font-weight: 500;">
                    <span style="color: {COLORS['primary']};">Upload a dataset</span>
                    <p style="font-weight: 400; margin-top: 0.2rem;">Use the file uploader in the sidebar to load your CSV, Excel, or JSON file.</p>
                </li>
                <li style="margin-bottom: 0.8rem; font-weight: 500;">
                    <span style="color: {COLORS['primary']};">Explore dataset statistics</span>
                    <p style="font-weight: 400; margin-top: 0.2rem;">View data summary, distributions, and correlations in the Dataset Overview.</p>
                </li>
                <li style="margin-bottom: 0.8rem; font-weight: 500;">
                    <span style="color: {COLORS['primary']};">Preprocess your data</span>
                    <p style="font-weight: 400; margin-top: 0.2rem;">Clean, transform, and prepare your data for machine learning models.</p>
                </li>
                <li style="font-weight: 500;">
                    <span style="color: {COLORS['primary']};">Export your results</span>
                    <p style="font-weight: 400; margin-top: 0.2rem;">Download your processed data in various formats for further use.</p>
                </li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # File format section - all formats in one card with fixed height
        st.markdown('<div class="section-title">Supported File Formats</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card format-card">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">📁</span> Compatible Formats
            </h3>
            <ul style="margin-top: 1rem; padding-left: 1.5rem;">
                <li style="margin-bottom: 0.6rem;">
                    <span style="font-weight: 500; color: {COLORS['primary']};">CSV (.csv)</span>
                    <p style="margin-top: 0.2rem;">Import and export comma-separated values files</p>
                </li>
                <li style="margin-bottom: 0.6rem;">
                    <span style="font-weight: 500; color: {COLORS['primary']};">Excel (.xlsx)</span>
                    <p style="margin-top: 0.2rem;">Import and export Microsoft Excel spreadsheets</p>
                </li>
                <li>
                    <span style="font-weight: 500; color: {COLORS['primary']};">JSON (.json)</span>
                    <p style="margin-top: 0.2rem;">Import and export JavaScript Object Notation files</p>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample dataset section with card styling and fixed height
        st.markdown('<div class="section-title">Try a Sample Dataset</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card sample-card">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">📊</span> Sample Data
            </h3>
            <p>
                Try out DataSightPro's features with a demo dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Centered button below card
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        if st.button("Load Sample Dataset", key="load_sample", use_container_width=True):
            st.markdown('</div>', unsafe_allow_html=True)
            try:
                # Load the customer data sample from the SampleDataset directory
                sample_path = "SampleDataset/customer_data.csv"
                df = pd.read_csv(sample_path)
                st.session_state.uploaded_file_name = "customer_data.csv"
                update_dataframe_and_analysis(df)
                
                # Perform initial analysis
                with st.spinner("Analyzing sample dataset..."):
                    perform_initial_analysis()
                    
                st.success("Sample customer dataset loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample dataset: {str(e)}")

#------------------------------------------------------------------------------
# MAIN APPLICATION LOGIC
#------------------------------------------------------------------------------

# Render sidebar and get selected analysis option
analysis_option = render_sidebar()

# Main content area
if st.session_state.df is not None:
    # Render appropriate content based on selected option
    if analysis_option == "Dataset Overview":
        render_dataset_overview()
    
    elif analysis_option == "Statistical Analysis":
        render_statistical_analysis()
    
    elif analysis_option == "Visualization":
        render_visualization()
    
    elif analysis_option == "Data Preprocessing":
        render_data_preprocessing()
    
    elif analysis_option == "Export Options":
        render_export_options()
else:
    # Display welcome screen when no data is loaded
    render_welcome_screen()
