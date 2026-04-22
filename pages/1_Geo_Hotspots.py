"""
PatrolIQ - Geographic Crime Hotspots Analysis
Page 1: Interactive geographic visualization and clustering results

Refactored to:
- Load pre-trained clustering models from artifacts/*.pkl
- Generate a small synthetic dataset for visualization
- Remove dependency on large CSV files
"""

import os
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import joblib
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Geographic Hotspots - PatrolIQ",
    page_icon="🗺️",
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
st.title("🗺️ Geographic Crime Hotspots")
st.markdown("### Interactive crime hotspot analysis across Chicago")

# -------------------------------------------------------------------
# Model paths
# -------------------------------------------------------------------
MODEL_DIR = "artifacts"
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
DBSCAN_PATH = os.path.join(MODEL_DIR, "dbscan_model.pkl")
HIER_PATH = os.path.join(MODEL_DIR, "hierarchical_model.pkl")

# -------------------------------------------------------------------
# Cached model loading
# -------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


kmeans_model = load_model(KMEANS_PATH)
dbscan_model = load_model(DBSCAN_PATH)
hier_model = load_model(HIER_PATH)

if not any([kmeans_model, dbscan_model, hier_model]):
    st.warning(
        "No clustering model files found in `artifacts/`. "
        "Please ensure kmeans_model.pkl, dbscan_model.pkl, hierarchical_model.pkl are present."
    )

# -------------------------------------------------------------------
# Synthetic data generation (small, in-memory demo set)
# -------------------------------------------------------------------
@st.cache_data
def create_synthetic_geo_data(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a compact synthetic dataset that mimics Chicago crime data
    for demonstration on Streamlit Cloud (no large CSVs).
    """
    rng = np.random.default_rng(random_state)

    # Rough bounding box for Chicago
    latitudes = rng.uniform(41.6, 42.05, size=n_rows)
    longitudes = rng.uniform(-87.95, -87.5, size=n_rows)

    # Dates over one year
    start_date = dt.date(2023, 1, 1)
    dates = [start_date + dt.timedelta(days=int(d)) for d in rng.integers(0, 365, size=n_rows)]

    primary_types = rng.choice(
        ["THEFT", "BATTERY", "NARCOTICS", "BURGLARY", "ASSAULT", "CRIMINAL DAMAGE"],
        size=n_rows,
    )
    crime_severity = rng.integers(1, 6, size=n_rows)  # 1–5
    districts = rng.integers(1, 26, size=n_rows)
    wards = rng.integers(1, 51, size=n_rows)
    arrests = rng.integers(0, 2, size=n_rows)

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Latitude": latitudes,
            "Longitude": longitudes,
            "Primary Type": primary_types,
            "Crime_Severity": crime_severity,
            "District": districts,
            "Ward": wards,
            "Arrest": arrests,
        }
    )

    return df


# -------------------------------------------------------------------
# Apply clustering using loaded models (with safe fallbacks)
# -------------------------------------------------------------------
@st.cache_data
def apply_geo_clustering(
    base_df: pd.DataFrame,
    kmeans_model,
    dbscan_model,
    hier_model,
) -> pd.DataFrame:
    """
    Use pre-trained models when available. If a model is missing or does not
    support predict, fall back to a small, in-memory fit on the synthetic data.
    """
    df = base_df.copy()
    X_geo = df[["Latitude", "Longitude"]].values

    # KMeans
    if kmeans_model is not None and hasattr(kmeans_model, "predict"):
        try:
            df["KMeans_Cluster"] = kmeans_model.predict(X_geo)
        except Exception:
            km_fallback = KMeans(n_clusters=6, random_state=42)
            df["KMeans_Cluster"] = km_fallback.fit_predict(X_geo)
    else:
        km_fallback = KMeans(n_clusters=6, random_state=42)
        df["KMeans_Cluster"] = km_fallback.fit_predict(X_geo)

    # DBSCAN
    if dbscan_model is not None and hasattr(dbscan_model, "fit_predict"):
        try:
            df["DBSCAN_Cluster"] = dbscan_model.fit_predict(X_geo)
        except Exception:
            db_fallback = DBSCAN(eps=0.01, min_samples=10)
            df["DBSCAN_Cluster"] = db_fallback.fit_predict(X_geo)
    else:
        db_fallback = DBSCAN(eps=0.01, min_samples=10)
        df["DBSCAN_Cluster"] = db_fallback.fit_predict(X_geo)

    # Hierarchical
    if hier_model is not None and hasattr(hier_model, "fit_predict"):
        try:
            df["Hierarchical_Cluster"] = hier_model.fit_predict(X_geo)
        except Exception:
            hier_fallback = AgglomerativeClustering(n_clusters=6, linkage="ward")
            df["Hierarchical_Cluster"] = hier_fallback.fit_predict(X_geo)
    else:
        hier_fallback = AgglomerativeClustering(n_clusters=6, linkage="ward")
        df["Hierarchical_Cluster"] = hier_fallback.fit_predict(X_geo)

    # Final cluster (simple choice: same as KMeans for demo)
    df["Final_Geo_Cluster"] = df["KMeans_Cluster"]

    return df


# -------------------------------------------------------------------
# Filtering helpers (same UI logic, new backend)
# -------------------------------------------------------------------
@st.cache_data
def prepare_filtered_data(
    df: pd.DataFrame,
    selected_algorithm: str,
    selected_clusters,
    selected_crimes,
    date_range,
) -> pd.DataFrame:
    filtered_df = df.copy()

    if selected_clusters:
        filtered_df = filtered_df[filtered_df[selected_algorithm].isin(selected_clusters)]

    if selected_crimes:
        filtered_df = filtered_df[filtered_df["Primary Type"].isin(selected_crimes)]

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["Date"].dt.date >= date_range[0])
            & (filtered_df["Date"].dt.date <= date_range[1])
        ]

    return filtered_df


@st.cache_data
def prepare_map_data(filtered_df: pd.DataFrame, selected_algorithm: str) -> pd.DataFrame:
    map_cols = ["Latitude", "Longitude", selected_algorithm]
    optional_cols = ["Primary Type", "Crime_Severity", "Arrest"]

    available_cols = map_cols + [col for col in optional_cols if col in filtered_df.columns]
    map_df = filtered_df[available_cols].copy()

    if len(map_df) > 3000:
        map_df = map_df.sample(n=3000, random_state=42).copy()

    return map_df


# -------------------------------------------------------------------
# Build in-memory dataset with clusters
# -------------------------------------------------------------------
base_df = create_synthetic_geo_data()
df = apply_geo_clustering(base_df, kmeans_model, dbscan_model, hier_model)

st.success(f"✅ Synthetic dataset ready with {len(df):,} records and model-based clusters")

# -------------------------------------------------------------------
# Sidebar filters (unchanged UI)
# -------------------------------------------------------------------
st.sidebar.header("🔍 Filters")

clustering_algorithms = [
    "KMeans_Cluster",
    "DBSCAN_Cluster",
    "Hierarchical_Cluster",
    "Final_Geo_Cluster",
]
available_algorithms = [col for col in clustering_algorithms if col in df.columns]

if not available_algorithms:
    st.warning("No clustering columns available to display.")
else:
    selected_algorithm = st.sidebar.selectbox(
        "Select Clustering Algorithm",
        available_algorithms,
        index=len(available_algorithms) - 1
        if "Final_Geo_Cluster" in available_algorithms
        else 0,
    )

    # Unique clusters (exclude -1 for DBSCAN noise)
    if selected_algorithm == "DBSCAN_Cluster":
        unique_clusters = sorted(
            [c for c in df[selected_algorithm].dropna().unique() if c != -1]
        )
    else:
        unique_clusters = sorted(df[selected_algorithm].dropna().unique())

    selected_clusters = st.sidebar.multiselect(
        "Select Clusters",
        unique_clusters,
        default=unique_clusters[: min(5, len(unique_clusters))],
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

    if "Date" in df.columns and not df.empty:
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(df["Date"].min(), df["Date"].max()),
            min_value=df["Date"].min(),
            max_value=df["Date"].max(),
        )
    else:
        date_range = ()

    filtered_df = prepare_filtered_data(
        df,
        selected_algorithm,
        tuple(selected_clusters) if selected_clusters else tuple(),
        tuple(selected_crimes) if selected_crimes else tuple(),
        tuple(date_range) if len(date_range) == 2 else tuple(),
    )

    map_df = prepare_map_data(filtered_df, selected_algorithm)

    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")

    # -------------------------------------------------------------------
    # Main content – Tabs (UI unchanged)
    # -------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "🗺️ Interactive Map", "📈 Cluster Analysis", "💾 Download"]
    )

    # ---------------- Tab 1: Overview ----------------
    with tab1:
        st.header("Clustering Results Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n_clusters = len(unique_clusters)
            st.metric("Total Clusters", n_clusters)

        with col2:
            avg_cluster_size = (
                len(filtered_df) // len(selected_clusters) if selected_clusters else 0
            )
            st.metric("Avg Cluster Size", f"{avg_cluster_size:,}")

        with col3:
            arrest_rate = (
                filtered_df["Arrest"].sum() / len(filtered_df) * 100
                if len(filtered_df) > 0
                else 0
            )
            st.metric("Arrest Rate", f"{arrest_rate:.1f}%")

        with col4:
            avg_severity = (
                filtered_df["Crime_Severity"].mean() if len(filtered_df) > 0 else 0
            )
            st.metric("Avg Severity", f"{avg_severity:.2f}/5")

        st.markdown("---")
        st.subheader("Cluster Statistics")

        if len(filtered_df) > 0:
            cluster_stats = (
                filtered_df.groupby(selected_algorithm)
                .agg(
                    {
                        "Latitude": "count",
                        "Arrest": "mean",
                        "Crime_Severity": "mean",
                        "Primary Type": lambda x: x.value_counts().index[0],
                    }
                )
                .reset_index()
            )

            cluster_stats.columns = [
                "Cluster",
                "Crime Count",
                "Arrest Rate",
                "Avg Severity",
                "Top Crime Type",
            ]
            cluster_stats["Arrest Rate"] = (
                cluster_stats["Arrest Rate"] * 100
            ).round(2)
            cluster_stats["Avg Severity"] = cluster_stats["Avg Severity"].round(2)
            cluster_stats = cluster_stats.sort_values("Crime Count", ascending=False)

            st.dataframe(cluster_stats, use_container_width=True)

            st.subheader("Cluster Size Distribution")
            fig = px.bar(
                cluster_stats,
                x="Cluster",
                y="Crime Count",
                color="Avg Severity",
                color_continuous_scale="RdYlGn_r",
                title="Crime Count by Cluster (colored by severity)",
                labels={"Crime Count": "Number of Crimes", "Cluster": "Cluster ID"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available with current filters")

    # ---------------- Tab 2: Interactive Map ----------------
    with tab2:
        st.header("Interactive Crime Hotspot Map")

        if len(filtered_df) > 0:
            st.info(
                f"Displaying {len(map_df):,} sampled crime locations "
                f"from {len(filtered_df):,} filtered records"
            )

            center_lat = map_df["Latitude"].mean()
            center_lon = map_df["Longitude"].mean()

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=11,
                tiles="OpenStreetMap",
            )

            colors = [
                "red",
                "blue",
                "green",
                "purple",
                "orange",
                "darkred",
                "lightred",
                "beige",
                "darkblue",
                "darkgreen",
                "cadetblue",
                "darkpurple",
                "white",
                "pink",
                "lightblue",
                "lightgreen",
                "gray",
                "black",
                "lightgray",
            ]

            for cluster_id in map_df[selected_algorithm].dropna().unique():
                cluster_data = map_df[map_df[selected_algorithm] == cluster_id]
                color = colors[int(cluster_id) % len(colors)]

                cluster_group = MarkerCluster(
                    name=f"Cluster {cluster_id}"
                ).add_to(m)

                for row in cluster_data.itertuples(index=False):
                    popup_text = f"Cluster: {getattr(row, selected_algorithm)}"
                    if "Primary Type" in map_df.columns:
                        popup_text += f"<br>Crime: {getattr(row, 'Primary Type')}"
                    if "Crime_Severity" in map_df.columns:
                        popup_text += f"<br>Severity: {getattr(row, 'Crime_Severity')}"

                    folium.Marker(
                        location=[getattr(row, "Latitude"), getattr(row, "Longitude")],
                        popup=folium.Popup(popup_text, max_width=250),
                        icon=folium.Icon(color=color, icon="info-sign"),
                    ).add_to(cluster_group)

            folium.LayerControl().add_to(m)
            st_folium(m, width=1400, height=600)

            st.subheader("Scatter Map Visualization")

            hover_cols = [
                col
                for col in ["Primary Type", "Crime_Severity", "Arrest"]
                if col in map_df.columns
            ]

            fig = px.scatter_mapbox(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color=selected_algorithm,
                hover_data=hover_cols,
                zoom=10,
                height=600,
                title=f"Crime Locations by {selected_algorithm}",
            )

            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available with current filters")

    # ---------------- Tab 3: Cluster Analysis ----------------
    with tab3:
        st.header("Detailed Cluster Analysis")

        if len(filtered_df) > 0:
            st.subheader("Crime Type Distribution by Cluster")

            crime_by_cluster = (
                filtered_df.groupby([selected_algorithm, "Primary Type"])
                .size()
                .reset_index(name="Count")
            )

            fig = px.bar(
                crime_by_cluster,
                x=selected_algorithm,
                y="Count",
                color="Primary Type",
                title="Crime Types within Each Cluster",
                labels={"Count": "Number of Crimes", selected_algorithm: "Cluster ID"},
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("District Distribution by Cluster")

            col1, col2 = st.columns(2)

            with col1:
                district_data = (
                    filtered_df.groupby([selected_algorithm, "District"])
                    .size()
                    .reset_index(name="Count")
                )
                fig = px.sunburst(
                    district_data,
                    path=[selected_algorithm, "District"],
                    values="Count",
                    title="Hierarchical View: Clusters → Districts",
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.box(
                    filtered_df,
                    x=selected_algorithm,
                    y="Crime_Severity",
                    color=selected_algorithm,
                    title="Crime Severity Distribution by Cluster",
                    labels={
                        "Crime_Severity": "Severity Score (1-5)",
                        selected_algorithm: "Cluster ID",
                    },
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Arrest Rate Comparison")

            arrest_by_cluster = filtered_df.groupby(selected_algorithm)["Arrest"].mean() * 100

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=arrest_by_cluster.index,
                        y=arrest_by_cluster.values,
                        marker_color="steelblue",
                        text=arrest_by_cluster.values.round(1),
                        textposition="outside",
                    )
                ]
            )

            fig.update_layout(
                title="Arrest Rate by Cluster (%)",
                xaxis_title="Cluster ID",
                yaxis_title="Arrest Rate (%)",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available with current filters")

    # ---------------- Tab 4: Download ----------------
    with tab4:
        st.header("💾 Download Cluster Data")

        st.markdown(
            """
            Download the filtered geographic clustering results for further analysis.
            The CSV file includes all records with their assigned clusters.
            """
        )

        if len(filtered_df) > 0:
            download_df = filtered_df[
                [
                    "Date",
                    "Latitude",
                    "Longitude",
                    "Primary Type",
                    "Crime_Severity",
                    "District",
                    "Ward",
                    "Arrest",
                    selected_algorithm,
                ]
            ].copy()

            csv = download_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"geo_clusters_filtered_{selected_algorithm}.csv",
                mime="text/csv",
            )

            st.subheader("Data Preview")
            st.dataframe(download_df.head(100), use_container_width=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Records", f"{len(download_df):,}")

            with col2:
                st.metric("Clusters", download_df[selected_algorithm].nunique())

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
        <p>Geographic Hotspots Analysis | PatrolIQ Platform</p>
    </div>
    """,
    unsafe_allow_html=True,
)