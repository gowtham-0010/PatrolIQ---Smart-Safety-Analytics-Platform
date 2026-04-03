"""
PatrolIQ - Smart Urban Safety Analytics Platform
Main Streamlit Application

Author: Machine Learning Engineer
Date: February 2026
"""

import streamlit as st
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="PatrolIQ - Smart Safety Analytics",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .main-header h3 {
        font-size: 1.5rem;
        color: #a0a0a0;
        font-weight: 400;
    }
    .mission-card {
        background: rgba(31, 119, 180, 0.08);
        border: 1px solid rgba(31, 119, 180, 0.25);
        border-left: 4px solid #1f77b4;
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        margin: 1.5rem 0 2rem 0;
        box-shadow: none;
    }
    .mission-card h3 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.85rem;
    }
    .mission-card p {
        color: #d6d6d6;
        font-size: 1.05rem;
        line-height: 1.8;
        margin: 0;
    }
    .metric-card {
        background-color: #1c1f26;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .feature-card {
        background-color: #1c1f26;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #888;
        font-size: 0.9rem;
    }
    hr {
        border: none;
        border-top: 1px solid #2d3748;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚨 PatrolIQ</h1>
    <h3>Smart Urban Safety Analytics Platform</h3>
</div>
""", unsafe_allow_html=True)

# Mission section
st.markdown("""
<div class="mission-card">
    <h3>🎯 Mission</h3>
    <p>
        Leverage unsupervised machine learning to analyze crime patterns and optimize
        police resource allocation in Chicago. This platform provides actionable insights
        to make cities safer through data-driven decisions.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Platform Overview
st.markdown("## 📊 Platform Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🗺️ Geographic Analysis
    - Crime Hotspot Identification
    - Cluster-based Zone Mapping
    - Interactive Geographic Visualization
    - Police District Analysis
    """)

with col2:
    st.markdown("""
    ### ⏰ Temporal Analysis
    - Time-Based Pattern Discovery
    - Peak Crime Hour Detection
    - Seasonal Trend Analysis
    - Weekend vs Weekday Patterns
    """)

with col3:
    st.markdown("""
    ### 🔍 Dimensionality Reduction
    - PCA for Feature Compression
    - t-SNE for Pattern Visualization
    - High-Dimensional Data Exploration
    - Cluster Separation Analysis
    """)

st.markdown("---")

# Dataset Statistics
st.markdown("## 📈 Dataset Statistics")

# Load processed data if available
PROCESSED_DATA_PATH = 'data/processed/chicago_crimes_processed.csv'

if os.path.exists(PROCESSED_DATA_PATH):
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            if 'Primary Type' in df.columns:
                st.metric("Crime Types", df['Primary Type'].nunique())
            else:
                st.metric("Crime Types", "N/A")
        
        with col3:
            if 'District' in df.columns:
                st.metric("Districts", df['District'].nunique())
            else:
                st.metric("Districts", "N/A")
        
        with col4:
            if 'Arrest' in df.columns:
                arrest_rate = (df['Arrest'].sum() / len(df) * 100) if len(df) > 0 else 0
                st.metric("Arrest Rate", f"{arrest_rate:.1f}%")
            else:
                st.metric("Arrest Rate", "N/A")
    
    except Exception as e:
        st.warning(f"Could not load processed data: {str(e)}")
        st.info("Please run the preprocessing notebook first.")
else:
    st.info("Processed dataset not found. Please run `01_data_preprocessing.ipynb` first.")

st.markdown("---")

# Technology Stack
st.markdown("## 🛠️ Technology Stack")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Machine Learning & Analytics
    - **Python** - Core programming language
    - **scikit-learn** - Clustering and dimensionality reduction
    - **pandas** - Data manipulation
    - **NumPy** - Numerical computations
    - **MLflow** - Experiment tracking
    """)

with col2:
    st.markdown("""
    ### Visualization & Deployment
    - **Streamlit** - Interactive web dashboard
    - **Plotly** - Interactive charts
    - **Folium** - Geographic maps
    - **Matplotlib / Seaborn** - Statistical visualization
    - **Jupyter Notebook** - Analysis workflow
    """)

st.markdown("---")

# Navigation Guide
st.markdown("## 🧭 Navigation Guide")

st.markdown("""
Use the sidebar to explore the main analytical modules:

- **Geographic Hotspots** — Analyze spatial crime clusters and hotspot zones.
- **Temporal Patterns** — Explore hourly, daily, and seasonal crime behavior.
- **Dimensionality Reduction** — Visualize high-dimensional patterns using PCA and t-SNE.

Each page provides interactive filtering, visual analysis, and downloadable outputs.
""")

st.markdown("---")

# Project Workflow
st.markdown("## 🔄 Analysis Workflow")

st.markdown("""
1. **Data Preprocessing** — Clean and prepare raw Chicago crime data.
2. **Exploratory Data Analysis** — Understand patterns and distributions.
3. **Feature Engineering** — Create ML-ready spatial and temporal features.
4. **Geographic Clustering** — Identify crime hotspots using clustering algorithms.
5. **Temporal Clustering** — Discover recurring time-based crime patterns.
6. **Dimensionality Reduction** — Reduce complexity and visualize hidden structure.
7. **Dashboard Deployment** — Interact with results using Streamlit.
""")

# Footer
st.markdown("""
<div class="footer">
    <p><strong>PatrolIQ</strong> | Smart Urban Safety Analytics Platform</p>
    <p>Built with ❤️ using Python, Streamlit, and Machine Learning</p>
    <p>Data Source: Chicago Data Portal - Crimes 2001 to Present</p>
</div>
""", unsafe_allow_html=True)