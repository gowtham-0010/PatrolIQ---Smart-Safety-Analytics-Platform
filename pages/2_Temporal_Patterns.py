# Temporal_patterns.py  (pages/2_Temporal_Patterns.py)

"""
PatrolIQ - Temporal Crime Patterns Analysis
Page 2: Time-based pattern discovery and clustering results

Final architecture:
- Notebook trains on full dataset, writes artifacts/temporal_clustered_sample.csv
- Streamlit ONLY reads preprocessed temporal sample CSV
- No synthetic data, no retraining in the app
"""

import os

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
# Load preprocessed temporal sample
# -------------------------------------------------------------------
TEMPORAL_SAMPLE_PATH = "artifacts/temporal_clustered_sample.csv"


@st.cache_data
def load_temporal_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df


df = load_temporal_data(TEMPORAL_SAMPLE_PATH)

if df.empty:
    st.error(
        "Temporal clustered sample not found or empty.\n"
        "Expected: artifacts/temporal_clustered_sample.csv"
    )
else:
    st.success(f"✅ Loaded {len(df):,} preprocessed temporal records from artifacts")

# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------
st.sidebar.header("🔍 Filters")

if not df.empty:
    clusters = sorted(df["Temporal_Cluster"].unique())
    selected_clusters = st.sidebar.multiselect(
        "Select Temporal Clusters",
        clusters,
        default=clusters[: min(5, len(clusters))],
    )

    crime_types = (
        sorted(df["Primary Type"].dropna().unique())
        if "Primary Type" in df.columns
        else []
    )
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

    if "DateTime" in df.columns:
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(df["DateTime"].dt.date.min(), df["DateTime"].dt.date.max()),
            min_value=df["DateTime"].dt.date.min(),
            max_value=df["DateTime"].dt.date.max(),
        )
    else:
        date_range = ()

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

    if "Hour" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["Hour"] >= hour_range[0])
            & (filtered_df["Hour"] <= hour_range[1])
        ]

    if len(date_range) == 2 and "DateTime" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["DateTime"].dt.date >= date_range[0])
            & (filtered_df["DateTime"].dt.date <= date_range[1])
        ]

    if "IsWeekend" in filtered_df.columns:
        if weekend_filter == "Weekdays":
            filtered_df = filtered_df[~filtered_df["IsWeekend"]]
        elif weekend_filter == "Weekends":
            filtered_df = filtered_df[filtered_df["IsWeekend"]]

    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")

    # -------------------------------------------------------------------
    # Main content
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
                filtered_df["IsWeekend"].mean() * 100
                if len(filtered_df) > 0 and "IsWeekend" in filtered_df.columns
                else 0
            )
            st.metric("Weekend Crimes", f"{weekend_pct:.1f}%")

        with col4:
            avg_severity = (
                filtered_df["Crime_Severity"].mean()
                if len(filtered_df) > 0 and "Crime_Severity" in filtered_df.columns
                else 0
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

            if "Hour" in filtered_df.columns:
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

            if "DayName" in filtered_df.columns:
                with col2:
                    st.subheader("Day of Week Distribution")
                    order_days = [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ]
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

            if "MonthName" in filtered_df.columns:
                st.subheader("Monthly Trend")
                month_order = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                monthly_counts = (
                    filtered_df.groupby(["MonthName", "Temporal_Cluster"])
                    .size()
                    .reset_index(name="Count")
                )
                monthly_counts["MonthName"] = pd.Categorical(
                    monthly_counts["MonthName"],
                    categories=month_order,
                    ordered=True,
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
            cluster_stats["Avg_Severity"] = (
                cluster_stats["Avg_Severity"].fillna(0).round(2)
            )
            cluster_stats["Arrest_Rate"] = (
                cluster_stats["Arrest_Rate"].fillna(0) * 100
            ).round(1)

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

            if "Hour" in filtered_df.columns:
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
            cols = [
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
            cols = [c for c in cols if c in filtered_df.columns]

            download_df = filtered_df[cols].copy()
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
                st.metric(
                    "Clusters",
                    download_df["Temporal_Cluster"].nunique()
                    if "Temporal_Cluster" in download_df.columns
                    else 0,
                )
            with col3:
                st.metric(
                    "Crime Types",
                    download_df["Primary Type"].nunique()
                    if "Primary Type" in download_df.columns
                    else 0,
                )
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