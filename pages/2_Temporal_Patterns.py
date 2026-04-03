"""
PatrolIQ - Temporal Crime Pattern Analysis
Page 2: Time-based crime pattern visualization and analysis
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
    page_title="Temporal Patterns - PatrolIQ",
    page_icon="⏰",
    layout="wide"
)

# Header
st.title("⏰ Temporal Crime Patterns")
st.markdown("### Discover when crimes occur most frequently")

# Load temporal clustering results
TEMPORAL_CLUSTERS_PATH = 'data/artifacts/temporal_clusters.csv'

if not os.path.exists(TEMPORAL_CLUSTERS_PATH):
    st.error("⚠️ Temporal clustering results not found!")
    st.info("Please run notebook `05_temporal_clustering.ipynb` first to generate the clustering artifacts.")
    st.stop()

# Load data with caching
@st.cache_data
def load_temporal_data():
    df = pd.read_csv(TEMPORAL_CLUSTERS_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_temporal_data()
    st.success(f"✅ Loaded {len(df):,} crime records with temporal clusters")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Temporal cluster filter
if 'Temporal_Cluster' in df.columns:
    unique_clusters = sorted(df['Temporal_Cluster'].unique())
    selected_clusters = st.sidebar.multiselect(
        "Select Temporal Clusters",
        unique_clusters,
        default=unique_clusters
    )
else:
    selected_clusters = None

# Crime type filter
crime_types = sorted(df['Primary Type'].unique())
selected_crimes = st.sidebar.multiselect(
    "Filter by Crime Type",
    crime_types,
    default=[]
)

# Time filters
hour_range = st.sidebar.slider(
    "Hour Range",
    0, 23, (0, 23)
)

day_filter = st.sidebar.multiselect(
    "Day of Week",
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    default=[]
)

# Apply filters
filtered_df = df.copy()

if selected_clusters and 'Temporal_Cluster' in df.columns:
    filtered_df = filtered_df[filtered_df['Temporal_Cluster'].isin(selected_clusters)]

if selected_crimes:
    filtered_df = filtered_df[filtered_df['Primary Type'].isin(selected_crimes)]

filtered_df = filtered_df[
    (filtered_df['Hour'] >= hour_range[0]) & 
    (filtered_df['Hour'] <= hour_range[1])
]

if day_filter:
    filtered_df = filtered_df[filtered_df['Day_Name'].isin(day_filter)]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⏰ Hourly Patterns", "📅 Daily & Seasonal", "💾 Download"])

# Tab 1: Overview
with tab1:
    st.header("Temporal Analysis Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Temporal_Cluster' in df.columns:
            n_patterns = df['Temporal_Cluster'].nunique()
            st.metric("Temporal Patterns", n_patterns)
        else:
            st.metric("Records Analyzed", f"{len(df):,}")
    
    with col2:
        peak_hour = filtered_df['Hour'].mode()[0] if len(filtered_df) > 0 else 0
        st.metric("Peak Crime Hour", f"{peak_hour}:00")
    
    with col3:
        weekend_pct = (filtered_df['Is_Weekend'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("Weekend Crimes", f"{weekend_pct:.1f}%")
    
    with col4:
        late_night_pct = (filtered_df['Is_Late_Night'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("Late Night (10PM-4AM)", f"{late_night_pct:.1f}%")
    
    st.markdown("---")
    
    # Temporal cluster analysis
    if 'Temporal_Cluster' in filtered_df.columns and 'Temporal_Pattern' in filtered_df.columns:
        st.subheader("Temporal Cluster Characteristics")
        
        cluster_stats = filtered_df.groupby(['Temporal_Cluster', 'Temporal_Pattern']).agg({
            'Hour': ['mean', 'count'],
            'Is_Weekend': 'mean',
            'Is_Late_Night': 'mean',
            'Crime_Severity': 'mean',
            'Arrest': 'mean'
        }).reset_index()
        
        cluster_stats.columns = ['Cluster', 'Pattern', 'Avg Hour', 'Crime Count', 
                                'Weekend %', 'Late Night %', 'Avg Severity', 'Arrest Rate']
        cluster_stats['Weekend %'] = (cluster_stats['Weekend %'] * 100).round(1)
        cluster_stats['Late Night %'] = (cluster_stats['Late Night %'] * 100).round(1)
        cluster_stats['Avg Hour'] = cluster_stats['Avg Hour'].round(1)
        cluster_stats['Avg Severity'] = cluster_stats['Avg Severity'].round(2)
        cluster_stats['Arrest Rate'] = (cluster_stats['Arrest Rate'] * 100).round(1)
        cluster_stats = cluster_stats.sort_values('Crime Count', ascending=False)
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Cluster size visualization
        fig = px.pie(
            cluster_stats,
            values='Crime Count',
            names='Pattern',
            title='Crime Distribution by Temporal Pattern',
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Hourly Patterns
with tab2:
    st.header("Hourly Crime Patterns")
    
    if len(filtered_df) > 0:
        # Overall hourly distribution
        st.subheader("Crime Distribution by Hour of Day")
        
        hourly_crimes = filtered_df['Hour'].value_counts().sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_crimes.index,
            y=hourly_crimes.values,
            mode='lines+markers',
            name='Crimes',
            line=dict(color='crimson', width=3),
            marker=dict(size=8),
            fill='tonexty',
            fillcolor='rgba(220, 20, 60, 0.2)'
        ))
        
        fig.update_layout(
            title='Hourly Crime Distribution',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Crimes',
            height=500,
            hovermode='x unified'
        )
        
        # Add annotation for peak hour
        peak_idx = hourly_crimes.idxmax()
        peak_val = hourly_crimes.max()
        fig.add_annotation(
            x=peak_idx,
            y=peak_val,
            text=f"Peak: {peak_idx}:00",
            showarrow=True,
            arrowhead=2
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly by cluster
        if 'Temporal_Cluster' in filtered_df.columns:
            st.subheader("Hourly Patterns by Temporal Cluster")
            
            hourly_cluster = filtered_df.groupby(['Hour', 'Temporal_Cluster']).size().reset_index(name='Count')
            
            fig = px.line(
                hourly_cluster,
                x='Hour',
                y='Count',
                color='Temporal_Cluster',
                title='Crime Count by Hour for Each Cluster',
                labels={'Count': 'Number of Crimes', 'Hour': 'Hour of Day'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap: Hour vs Crime Type
        st.subheader("Hour vs Crime Type Heatmap")
        
        top_crimes = filtered_df['Primary Type'].value_counts().head(10).index
        heatmap_df = filtered_df[filtered_df['Primary Type'].isin(top_crimes)]
        heatmap_data = heatmap_df.groupby(['Hour', 'Primary Type']).size().unstack(fill_value=0)
        
        fig = px.imshow(
            heatmap_data.T,
            labels=dict(x="Hour of Day", y="Crime Type", color="Crime Count"),
            x=heatmap_data.index,
            y=heatmap_data.columns,
            color_continuous_scale='YlOrRd',
            aspect='auto'
        )
        fig.update_layout(height=500, title='Crime Type Distribution Across Hours')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No data available with current filters")

# Tab 3: Daily & Seasonal
with tab3:
    st.header("Daily and Seasonal Patterns")
    
    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week distribution
            st.subheader("Day of Week Distribution")
            
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_crimes = filtered_df['Day_Name'].value_counts().reindex(day_order)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=daily_crimes.index,
                    y=daily_crimes.values,
                    marker_color='steelblue',
                    text=daily_crimes.values,
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                xaxis_title='Day of Week',
                yaxis_title='Number of Crimes',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weekend vs Weekday
            st.subheader("Weekend vs Weekday")
            
            weekend_counts = filtered_df['Is_Weekend'].value_counts()
            labels = ['Weekday', 'Weekend']
            values = [weekend_counts.get(0, 0), weekend_counts.get(1, 0)]
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker_colors=['lightcoral', 'lightskyblue']
                )
            ])
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly distribution
        st.subheader("Monthly Crime Distribution")
        
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_crimes = filtered_df['Month_Name'].value_counts().reindex(month_order)
        
        fig = px.bar(
            x=monthly_crimes.index,
            y=monthly_crimes.values,
            labels={'x': 'Month', 'y': 'Number of Crimes'},
            title='Crime Count by Month',
            color=monthly_crimes.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal distribution
        st.subheader("Seasonal Crime Distribution")
        
        seasonal_crimes = filtered_df['Season'].value_counts()
        
        fig = px.bar(
            x=seasonal_crimes.index,
            y=seasonal_crimes.values,
            labels={'x': 'Season', 'y': 'Number of Crimes'},
            title='Crime Count by Season',
            color=seasonal_crimes.index,
            color_discrete_map={
                'Winter': 'lightblue',
                'Spring': 'lightgreen',
                'Summer': 'orange',
                'Fall': 'brown'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Day-Hour heatmap
        st.subheader("Day of Week vs Hour Heatmap")
        
        day_hour_data = filtered_df.groupby(['Day_Name', 'Hour']).size().unstack(fill_value=0)
        day_hour_data = day_hour_data.reindex(day_order)
        
        fig = px.imshow(
            day_hour_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Crime Count"),
            x=day_hour_data.columns,
            y=day_hour_data.index,
            color_continuous_scale='Reds',
            aspect='auto'
        )
        fig.update_layout(height=500, title='Crime Density: Day vs Hour')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No data available with current filters")

# Tab 4: Download
with tab4:
    st.header("💾 Download Temporal Data")
    
    st.markdown("""
    Download the filtered temporal clustering results for further analysis.
    """)
    
    if len(filtered_df) > 0:
        # Prepare download data
        download_cols = ['Date', 'Hour', 'Day_Name', 'Month_Name', 'Season',
                        'Is_Weekend', 'Is_Late_Night', 'Primary Type', 
                        'Crime_Severity', 'Arrest']
        
        if 'Temporal_Cluster' in filtered_df.columns:
            download_cols.append('Temporal_Cluster')
        if 'Temporal_Pattern' in filtered_df.columns:
            download_cols.append('Temporal_Pattern')
        
        download_df = filtered_df[download_cols].copy()
        
        csv = download_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name="temporal_patterns_filtered.csv",
            mime="text/csv"
        )
        
        # Display sample
        st.subheader("Data Preview")
        st.dataframe(download_df.head(100), use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(download_df):,}")
        
        with col2:
            st.metric("Date Range", f"{(filtered_df['Date'].max() - filtered_df['Date'].min()).days} days")
        
        with col3:
            st.metric("Crime Types", download_df['Primary Type'].nunique())
        
    else:
        st.warning("No data available for download with current filters")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Temporal Patterns Analysis | PatrolIQ Platform</p>
</div>
""", unsafe_allow_html=True)
