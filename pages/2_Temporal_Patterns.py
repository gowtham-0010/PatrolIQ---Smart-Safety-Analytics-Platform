# pages/2_Temporal_Patterns.py

"""
PatrolIQ - Temporal Crime Patterns Analysis
Page 2: Time-based pattern discovery and clustering results

Refactored to:
- Remove CSV dependencies
- Use synthetic temporal dataset
- Optionally use a pre-trained temporal clustering model from artifacts/temporal_model.pkl
"""

import os
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import joblib

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Temporal Crime Patterns - PatrolIQ",
    page_icon="⏰",
    layout="wide",
)

# -------------------------------------------------------------------
# Custom CSS
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.title("⏰ Temporal Crime Patterns")
st.markdown("### Explore when crimes occur across hours, days, and months")

# -------------------------------------------------------------------
# Optional pre-trained temporal model
# -------------------------------------------------------------------
TEMPORAL_MODEL_PATH = os.path.join("artifacts", "temporal_model.pkl")


@st.cache_resource
def load_temporal_model(path: str):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


temporal_model = load_temporal_model(TEMPORAL_MODEL_PATH)
if temporal_model is None:
    st.info(
        "No pre-trained temporal model found in `artifacts/temporal_model.pkl`. "
        "Using in-app KMeans clustering for temporal patterns."
    )

# -------------------------------------------------------------------
# Synthetic temporal data
# -------------------------------------------------------------------
@st.cache_data
def create_synthetic_temporal_data(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # Create random datetimes over one year
    start = dt.datetime(2023, 1, 1)
    timestamps = [
        start + dt.timedelta(
            days=int(d),
            hours=int(h),
            minutes=int(m),
        )
        for d, h, m in zip(
            rng.integers(0, 365, size=n_rows),
            rng.integers(0, 24, size=n_rows),
            rng.integers(0, 60, size=n_rows),
        )
    ]

    primary_types = rng.choice(
        ["THEFT", "BATTERY", "NARCOTICS", "BURGLARY", "ASSAULT", "CRIMINAL DAMAGE"],
        size=n_rows,
    )
    crime_severity = rng.integers(1, 6, size=n_rows)
    districts = rng.integers(1, 26, size=n_rows)
    wards = rng.integers(1, 51, size=n_rows)
    arrests = rng.integers(0, 2, size=n_rows)

    df = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(timestamps),
            "Primary Type": primary_types,
            "Crime_Severity": crime_severity,
            "District": districts,
            "Ward": wards,
            "Arrest": arrests,
        }
    )

    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["DayName"] = df["DateTime"].dt.day_name()
    df["Month"] = df["DateTime"].dt.month
    df["MonthName"] = df["DateTime"].dt.strftime("%b")
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6])

    return df


# -------------------------------------------------------------------
# Temporal clustering
# -------------------------------------------------------------------
@st.cache_data
def apply_temporal_clustering(df: pd.DataFrame) -> pd.DataFrame:
    X_time = df[["Hour", "DayOfWeek", "Month"]].values

    if temporal_model is not None and hasattr(temporal_model, "predict"):
        try:
            labels = temporal_model.predict(X_time)
        except Exception:
            km = KMeans(n_clusters=6, random_state=42)
            labels = km.fit_predict(X_time)
    else:
        km = KMeans(n_clusters=6, random_state=42)
        labels = km.fit_predict(X_time)

    df_out = df.copy()
    df_out["Temporal_Cluster"] = labels
    return df_out


# -------------------------------------------------------------------
# Build dataset
# -------------------------------------------------------------------
base_df = create_synthetic_temporal_data()
df = apply_temporal_clustering(base_df)

st.success(f"✅ Synthetic temporal dataset ready with {len(df):,} records")

# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------
st.sidebar.header("🔍 Filters")

cluster_ids = sorted(df["Temporal_Cluster"].unique())
selected_clusters = st.sidebar.multiselect(
    "Select Temporal Clusters",
    cluster_ids,
    default=cluster_ids[: min(5, len(cluster_ids))],
)

crime_types = sorted(df["Primary Type"].unique())
selected_crimes = st.sidebar.multiselect(
    "Filter by Crime Type",
    crime_types,
    default=[],
)

hour_range = st.sidebar.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=(0, 23),
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(df["DateTime"].dt.date.min(), df["DateTime"].dt.date.max()),
    min_value=df["DateTime"].dt.date.min(),
    max_value=df["DateTime"].dt.date.max(),
)

weekend_filter = st.sidebar.selectbox(
    "Day Type",
    ["All", "Weekdays", "Weekends"],
    index=0,
)

# -------------------------------------------------------------------
# Apply filters
# -------------------------------------------------------------------
filtered_df = df.copy()

if selected_clusters:
    filtered_df = filtered_df[filtered_df["Temporal_Cluster"].isin(selected_clusters)]

if selected_crimes:
    filtered_df = filtered_df[filtered_df["Primary Type"].isin(selected_crimes)]

filtered_df = filtered_df[
    (filtered_df["Hour"] >= hour_range[0]) & (filtered_df["Hour"] <= hour_range[1])
]

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["DateTime"].dt.date >= date_range[0])
        & (filtered_df["DateTime"].dt.date <= date_range[1])
    ]

if weekend_filter == "Weekdays":
    filtered_df = filtered_df[~filtered_df["IsWeekend"]]
elif weekend_filter == "Weekends":
    filtered_df = filtered_df[filtered_df["IsWeekend"]]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")

# -------------------------------------------------------------------
# Main content tabs
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "⏱️ Time Distributions", "📈 Cluster Analysis", "💾 Download"]
)

# ---------------- Tab 1: Overview ----------------
with tab1:
    st.header("Temporal Clustering Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(filtered_df):,}")

    with col2:
        n_clusters = filtered_df["Temporal_Cluster"].nunique()
        st.metric("Active Clusters", n_clusters)

    with col3:
        weekend_pct = (
            filtered_df["IsWeekend"].mean() * 100 if len(filtered_df) > 0 else 0
        )
        st.metric("Weekend Crimes", f"{weekend_pct:.1f}%")

    with col4:
        avg_severity = (
            filtered_df["Crime_Severity"].mean() if len(filtered_df) > 0 else 0
        )
        st.metric("Avg Severity", f"{avg_severity:.2f}/5")

    st.markdown("---")

    st.subheader("Top Crime Types by Temporal Cluster")
    if len(filtered_df) > 0:
        top_crime = (
            filtered_df.groupby("Temporal_Cluster")["Primary Type"]
            .agg(lambda x: x.value_counts().index[0])
            .reset_index()
        )
        st.dataframe(top_crime, use_container_width=True)
    else:
        st.warning("No data available with current filters")

# ---------------- Tab 2: Time Distributions ----------------
with tab2:
    st.header("Time-based Distributions")

    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Hourly Distribution")
            fig_hour = px.histogram(
                filtered_df,
                x="Hour",
                color="Temporal_Cluster",
                barmode="group",
                nbins=24,
                title="Crimes by Hour of Day",
            )
            fig_hour.update_layout(height=400)
            st.plotly_chart(fig_hour, use_container_width=True)

        with col2:
            st.subheader("Day of Week Distribution")
            order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            fig_day = px.histogram(
                filtered_df,
                x="DayName",
                category_orders={"DayName": order_days},
                color="Temporal_Cluster",
                barmode="group",
                title="Crimes by Day of Week",
            )
            fig_day.update_layout(height=400)
            st.plotly_chart(fig_day, use_container_width=True)

        st.subheader("Monthly Trend")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_counts = (
            filtered_df.groupby(["MonthName", "Temporal_Cluster"])
            .size()
            .reset_index(name="Count")
        )
        monthly_counts["MonthName"] = pd.Categorical(
            monthly_counts["MonthName"], categories=month_order, ordered=True
        )
        monthly_counts = monthly_counts.sort_values("MonthName")

        fig_month = px.line(
            monthly_counts,
            x="MonthName",
            y="Count",
            color="Temporal_Cluster",
            markers=True,
            title="Crimes per Month by Temporal Cluster",
        )
        fig_month.update_layout(height=400)
        st.plotly_chart(fig_month, use_container_width=True)
    else:
        st.warning("No data available with current filters")

# ---------------- Tab 3: Cluster Analysis ----------------
with tab3:
    st.header("Detailed Temporal Cluster Analysis")

    if len(filtered_df) > 0:
        st.subheader("Cluster Size and Severity")
        cluster_stats = (
            filtered_df.groupby("Temporal_Cluster")
            .agg(
                Crime_Count=("Primary Type", "count"),
                Avg_Severity=("Crime_Severity", "mean"),
                Arrest_Rate=("Arrest", "mean"),
            )
            .reset_index()
        )
        cluster_stats["Avg_Severity"] = cluster_stats["Avg_Severity"].round(2)
        cluster_stats["Arrest_Rate"] = (cluster_stats["Arrest_Rate"] * 100).round(1)

        st.dataframe(cluster_stats, use_container_width=True)

        fig_cluster = px.bar(
            cluster_stats,
            x="Temporal_Cluster",
            y="Crime_Count",
            color="Avg_Severity",
            color_continuous_scale="RdYlGn_r",
            title="Crime Count by Temporal Cluster (colored by severity)",
            labels={"Crime_Count": "Number of Crimes", "Temporal_Cluster": "Cluster"},
        )
        fig_cluster.update_layout(height=400)
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.subheader("Hourly Heatmap by Cluster")
        heat_df = (
            filtered_df.groupby(["Temporal_Cluster", "Hour"])
            .size()
            .reset_index(name="Count")
        )
        fig_heat = px.density_heatmap(
            heat_df,
            x="Hour",
            y="Temporal_Cluster",
            z="Count",
            color_continuous_scale="Viridis",
            title="Heatmap of Crimes by Hour and Temporal Cluster",
        )
        fig_heat.update_layout(height=450)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("No data available with current filters")

# ---------------- Tab 4: Download ----------------
with tab4:
    st.header("💾 Download Temporal Data")

    st.markdown(
        """
        Download the filtered temporal clustering results for further analysis.
        The CSV file includes all records with their assigned temporal clusters.
        """
    )

    if len(filtered_df) > 0:
        download_df = filtered_df[
            [
                "DateTime",
                "Hour",
                "DayName",
                "MonthName",
                "Primary Type",
                "Crime_Severity",
                "District",
                "Ward",
                "Arrest",
                "Temporal_Cluster",
            ]
        ].copy()

        csv = download_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Filtered Temporal Data (CSV)",
            data=csv,
            file_name="temporal_clusters_filtered.csv",
            mime="text/csv",
        )

        st.subheader("Data Preview")
        st.dataframe(download_df.head(100), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(download_df):,}")
        with col2:
            st.metric("Clusters", download_df["Temporal_Cluster"].nunique())
        with col3:
            st.metric("Crime Types", download_df["Primary Type"].nunique())
    else:
        st.warning("No data available for download with current filters")

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>Temporal Crime Patterns Analysis | PatrolIQ Platform</p>
    </div>
    """,
    unsafe_allow_html=True,
)