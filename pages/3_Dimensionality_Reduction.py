"""
PatrolIQ - Dimensionality Reduction Visualization
Page 3: PCA and t-SNE visualization of high-dimensional crime patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Dimensionality Reduction - PatrolIQ",
    page_icon="🔍",
    layout="wide"
)

# Header
st.title("🔍 Dimensionality Reduction")
st.markdown("### Visualize high-dimensional crime patterns in 2D space")

# Load dimensionality reduction results
PCA_PATH = 'data/artifacts/pca_components.csv'
TSNE_PATH = 'data/artifacts/tsne_components.csv'

pca_available = os.path.exists(PCA_PATH)
tsne_available = os.path.exists(TSNE_PATH)

if not pca_available and not tsne_available:
    st.error("⚠️ Dimensionality reduction results not found!")
    st.info("Please run notebooks `06_pca_analysis.ipynb` and `07_tsne_analysis.ipynb` first.")
    st.stop()

# Load data with caching
@st.cache_data
def load_pca_data():
    if pca_available:
        df = pd.read_csv(PCA_PATH)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

@st.cache_data
def load_tsne_data():
    if tsne_available:
        df = pd.read_csv(TSNE_PATH)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

# Load data
pca_df = load_pca_data()
tsne_df = load_tsne_data()

# Display status
col1, col2 = st.columns(2)
with col1:
    if pca_available and pca_df is not None:
        st.success(f"✅ PCA: {len(pca_df):,} records loaded")
    else:
        st.warning("⚠️ PCA data not available")

with col2:
    if tsne_available and tsne_df is not None:
        st.success(f"✅ t-SNE: {len(tsne_df):,} records loaded")
    else:
        st.warning("⚠️ t-SNE data not available")

# Sidebar
st.sidebar.header("🎨 Visualization Settings")

# Select visualization method
viz_method = st.sidebar.radio(
    "Select Method",
    ["PCA", "t-SNE", "Compare Both"],
    disabled=(not pca_available and not tsne_available)
)

# Color coding selection
color_by = st.sidebar.selectbox(
    "Color Points By",
    ["Primary_Type", "Crime_Severity", "Hour", "Arrest", "District", "Season"]
)

# Sample size for visualization
if viz_method in ["PCA", "Compare Both"] and pca_df is not None:
    max_sample = min(50000, len(pca_df))
elif tsne_df is not None:
    max_sample = min(50000, len(tsne_df))
else:
    max_sample = 10000

sample_size = st.sidebar.slider(
    "Sample Size (for performance)",
    1000, max_sample, min(10000, max_sample), 1000
)

# Point size
point_size = st.sidebar.slider("Point Size", 1, 10, 3)

# Opacity
opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.6, 0.1)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Visualizations", "📈 Analysis", "💾 Download"])

# Tab 1: Overview
with tab1:
    st.header("Dimensionality Reduction Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Principal Component Analysis (PCA)
        
        **Purpose:** Linear dimensionality reduction
        
        **Key Features:**
        - Reduces 20+ features to 2-3 components
        - Captures 70%+ of variance
        - Fast and interpretable
        - Shows linear relationships
        
        **Best For:**
        - Understanding feature importance
        - Variance analysis
        - Quick initial exploration
        """)
        
        if pca_available and pca_df is not None:
            st.info(f"✓ PCA Components Available: {[col for col in pca_df.columns if col.startswith('PC')]}")
    
    with col2:
        st.markdown("""
        ### 🎨 t-SNE (t-Distributed Stochastic Neighbor Embedding)
        
        **Purpose:** Non-linear dimensionality reduction
        
        **Key Features:**
        - Preserves local structure
        - Reveals complex patterns
        - Better cluster separation
        - More intuitive visualization
        
        **Best For:**
        - Discovering hidden patterns
        - Cluster visualization
        - Non-linear relationships
        """)
        
        if tsne_available and tsne_df is not None:
            st.info(f"✓ t-SNE Components Available: {[col for col in tsne_df.columns if col.startswith('tSNE')]}")
    
    st.markdown("---")
    
    # Comparison table
    st.subheader("Method Comparison")
    
    comparison_data = {
        "Aspect": ["Type", "Speed", "Scalability", "Interpretability", "Cluster Quality", "Variance Captured"],
        "PCA": ["Linear", "Fast ⚡", "Excellent", "High", "Good", "Quantifiable"],
        "t-SNE": ["Non-linear", "Slower 🐢", "Moderate", "Low", "Excellent", "Not quantifiable"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

# Tab 2: Visualizations
with tab2:
    st.header("Interactive Visualizations")
    
    if viz_method == "PCA" and pca_df is not None:
        st.subheader("PCA Visualization")
        
        # Sample data
        pca_sample = pca_df.sample(n=min(sample_size, len(pca_df)), random_state=42)
        
        # Handle crime type grouping
        if color_by == "Primary_Type":
            top_crimes = pca_sample['Primary_Type'].value_counts().head(10).index
            pca_sample['Color_Category'] = pca_sample['Primary_Type'].apply(
                lambda x: x if x in top_crimes else 'Other'
            )
        else:
            pca_sample['Color_Category'] = pca_sample[color_by]
        
        # Create scatter plot
        fig = px.scatter(
            pca_sample,
            x='PC1',
            y='PC2',
            color='Color_Category',
            hover_data=['Primary_Type', 'Crime_Severity', 'Hour'] if 'Primary_Type' in pca_sample.columns else None,
            title=f'PCA Visualization (colored by {color_by})',
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            opacity=opacity
        )
        
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D visualization if PC3 exists
        if 'PC3' in pca_df.columns:
            st.subheader("3D PCA Visualization")
            
            fig_3d = px.scatter_3d(
                pca_sample,
                x='PC1',
                y='PC2',
                z='PC3',
                color='Color_Category',
                title=f'3D PCA Visualization (colored by {color_by})',
                opacity=opacity
            )
            
            fig_3d.update_traces(marker=dict(size=point_size))
            fig_3d.update_layout(height=700)
            st.plotly_chart(fig_3d, use_container_width=True)
    
    elif viz_method == "t-SNE" and tsne_df is not None:
        st.subheader("t-SNE Visualization")
        
        # Sample data
        tsne_sample = tsne_df.sample(n=min(sample_size, len(tsne_df)), random_state=42)
        
        # Handle crime type grouping
        if color_by == "Primary_Type":
            top_crimes = tsne_sample['Primary_Type'].value_counts().head(10).index
            tsne_sample['Color_Category'] = tsne_sample['Primary_Type'].apply(
                lambda x: x if x in top_crimes else 'Other'
            )
        else:
            tsne_sample['Color_Category'] = tsne_sample[color_by]
        
        # Create scatter plot
        fig = px.scatter(
            tsne_sample,
            x='tSNE1',
            y='tSNE2',
            color='Color_Category',
            hover_data=['Primary_Type', 'Crime_Severity', 'Hour'] if 'Primary_Type' in tsne_sample.columns else None,
            title=f't-SNE Visualization (colored by {color_by})',
            labels={'tSNE1': 't-SNE Component 1', 'tSNE2': 't-SNE Component 2'},
            opacity=opacity
        )
        
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_method == "Compare Both" and pca_df is not None and tsne_df is not None:
        st.subheader("Side-by-Side Comparison")
        
        # Find common indices
        common_indices = pca_df.index.intersection(tsne_df.index)
        
        if len(common_indices) > 100:
            # Sample from common indices
            sample_indices = np.random.choice(common_indices, min(sample_size, len(common_indices)), replace=False)
            
            pca_sample = pca_df.loc[sample_indices]
            tsne_sample = tsne_df.loc[sample_indices]
            
            # Handle crime type grouping for PCA
            if color_by == "Primary_Type":
                top_crimes = pca_sample['Primary_Type'].value_counts().head(5).index
                pca_sample['Color_Category'] = pca_sample['Primary_Type'].apply(
                    lambda x: x if x in top_crimes else 'Other'
                )
                tsne_sample['Color_Category'] = tsne_sample['Primary_Type'].apply(
                    lambda x: x if x in top_crimes else 'Other'
                )
            else:
                pca_sample['Color_Category'] = pca_sample[color_by]
                tsne_sample['Color_Category'] = tsne_sample[color_by]
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('PCA', 't-SNE'),
                horizontal_spacing=0.1
            )
            
            # Get unique categories and create color map
            unique_categories = sorted(pca_sample['Color_Category'].unique())
            colors = px.colors.qualitative.Plotly
            color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}
            
            # Add PCA traces
            for category in unique_categories:
                mask = pca_sample['Color_Category'] == category
                fig.add_trace(
                    go.Scatter(
                        x=pca_sample[mask]['PC1'],
                        y=pca_sample[mask]['PC2'],
                        mode='markers',
                        name=str(category),
                        marker=dict(size=point_size, color=color_map[category], opacity=opacity),
                        legendgroup=str(category),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Add t-SNE traces
            for category in unique_categories:
                mask = tsne_sample['Color_Category'] == category
                fig.add_trace(
                    go.Scatter(
                        x=tsne_sample[mask]['tSNE1'],
                        y=tsne_sample[mask]['tSNE2'],
                        mode='markers',
                        name=str(category),
                        marker=dict(size=point_size, color=color_map[category], opacity=opacity),
                        legendgroup=str(category),
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            fig.update_xaxes(title_text="PC1", row=1, col=1)
            fig.update_yaxes(title_text="PC2", row=1, col=1)
            fig.update_xaxes(title_text="t-SNE1", row=1, col=2)
            fig.update_yaxes(title_text="t-SNE2", row=1, col=2)
            
            fig.update_layout(height=600, title_text=f"PCA vs t-SNE Comparison (colored by {color_by})")
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison insights
            st.info("""
            **Comparison Insights:**
            - **PCA** shows the linear relationships and maintains global structure
            - **t-SNE** reveals local patterns and creates more distinct clusters
            - Both methods complement each other in understanding crime patterns
            """)
        else:
            st.warning("Not enough common data points for comparison")

# Tab 3: Analysis
with tab3:
    st.header("Detailed Analysis")
    
    if pca_df is not None:
        st.subheader("📊 PCA Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PCA statistics
            st.markdown("**PCA Statistics:**")
            pca_components = [col for col in pca_df.columns if col.startswith('PC')]
            st.write(f"- Components available: {len(pca_components)}")
            st.write(f"- Total records: {len(pca_df):,}")
            
            if 'Crime_Severity' in pca_df.columns:
                st.write(f"- Avg crime severity: {pca_df['Crime_Severity'].mean():.2f}/5")
        
        with col2:
            # Component distribution
            if len(pca_components) >= 2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=pca_df['PC1'], name='PC1', opacity=0.7))
                fig.add_trace(go.Histogram(x=pca_df['PC2'], name='PC2', opacity=0.7))
                fig.update_layout(
                    title='Distribution of Principal Components',
                    xaxis_title='Component Value',
                    yaxis_title='Frequency',
                    barmode='overlay',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Crime type distribution in PC space
        if 'Primary_Type' in pca_df.columns:
            st.markdown("**Crime Type Distribution in PCA Space:**")
            
            top_crimes = pca_df['Primary_Type'].value_counts().head(10)
            
            fig = px.bar(
                x=top_crimes.index,
                y=top_crimes.values,
                labels={'x': 'Crime Type', 'y': 'Count'},
                title='Top 10 Crime Types in PCA Dataset'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    if tsne_df is not None:
        st.subheader("🎨 t-SNE Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # t-SNE statistics
            st.markdown("**t-SNE Statistics:**")
            st.write(f"- Total records: {len(tsne_df):,}")
            
            if 'Crime_Severity' in tsne_df.columns:
                st.write(f"- Avg crime severity: {tsne_df['Crime_Severity'].mean():.2f}/5")
            
            if 'Hour' in tsne_df.columns:
                st.write(f"- Peak hour: {tsne_df['Hour'].mode()[0]}:00")
        
        with col2:
            # Component distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=tsne_df['tSNE1'], name='tSNE1', opacity=0.7))
            fig.add_trace(go.Histogram(x=tsne_df['tSNE2'], name='tSNE2', opacity=0.7))
            fig.update_layout(
                title='Distribution of t-SNE Components',
                xaxis_title='Component Value',
                yaxis_title='Frequency',
                barmode='overlay',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Density plot
        st.markdown("**t-SNE Density Distribution:**")
        
        sample_for_density = tsne_df.sample(n=min(20000, len(tsne_df)), random_state=42)
        
        fig = px.density_contour(
            sample_for_density,
            x='tSNE1',
            y='tSNE2',
            title='Crime Density in t-SNE Space'
        )
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Download
with tab4:
    st.header("💾 Download Dimensionality Reduction Data")
    
    st.markdown("""
    Download the dimensionality reduction results for further analysis or visualization in other tools.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pca_df is not None:
            st.subheader("📊 PCA Data")
            
            csv = pca_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download PCA Components (CSV)",
                data=csv,
                file_name="pca_components.csv",
                mime="text/csv"
            )
            
            st.metric("Records", f"{len(pca_df):,}")
            st.metric("Components", len([col for col in pca_df.columns if col.startswith('PC')]))
        else:
            st.warning("PCA data not available")
    
    with col2:
        if tsne_df is not None:
            st.subheader("🎨 t-SNE Data")
            
            csv = tsne_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download t-SNE Components (CSV)",
                data=csv,
                file_name="tsne_components.csv",
                mime="text/csv"
            )
            
            st.metric("Records", f"{len(tsne_df):,}")
            st.metric("Components", 2)
        else:
            st.warning("t-SNE data not available")
    
    # Data preview
    st.markdown("---")
    st.subheader("Data Preview")
    
    preview_method = st.radio("Select data to preview", ["PCA", "t-SNE"])
    
    if preview_method == "PCA" and pca_df is not None:
        st.dataframe(pca_df.head(100), use_container_width=True)
    elif preview_method == "t-SNE" and tsne_df is not None:
        st.dataframe(tsne_df.head(100), use_container_width=True)
    else:
        st.warning(f"{preview_method} data not available")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Dimensionality Reduction Analysis | PatrolIQ Platform</p>
</div>
""", unsafe_allow_html=True)
