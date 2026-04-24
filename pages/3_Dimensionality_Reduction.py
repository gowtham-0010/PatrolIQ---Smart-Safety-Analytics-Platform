# Dimensionality_reduction.py  (pages/3_Dimensionality_Reduction.py)

"""
PatrolIQ - Dimensionality Reduction Analysis
Page 3: PCA and t-SNE visualization of crime patterns

Final architecture:
- Notebook trains on full dataset, writes artifacts/pca_tsne_sample.csv
- Streamlit ONLY reads preprocessed DR sample CSV
- No synthetic data, no retraining in the app
"""

import os

import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dimensionality Reduction - PatrolIQ",
    page_icon="🔍",
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
st.title("🔍 Dimensionality Reduction")
st.markdown("### Visualize high-dimensional crime patterns using PCA and t-SNE")

# -------------------------------------------------------------------
# Load preprocessed PCA/t-SNE sample
# -------------------------------------------------------------------
DR_SAMPLE_PATH = "artifacts/pca_tsne_sample.csv"


@st.cache_data
def load_dr_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


df = load_dr_data(DR_SAMPLE_PATH)

if df.empty:
    st.error(
        "Dimensionality reduction sample not found or empty.\n"
        "Expected: artifacts/pca_tsne_sample.csv"
    )
else:
    st.success(f"✅ Loaded {len(df):,} PCA/t-SNE records from artifacts")

# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------
st.sidebar.header("🔍 Filters")

if not df.empty:
    clusters = sorted(df["ML_Cluster"].unique()) if "ML_Cluster" in df.columns else []
    selected_clusters = st.sidebar.multiselect(
        "Select Clusters",
        clusters,
        default=clusters[: min(5, len(clusters))] if clusters else [],
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

    if "Crime_Severity" in df.columns:
        min_sev = int(df["Crime_Severity"].min())
        max_sev = int(df["Crime_Severity"].max())
    else:
        min_sev, max_sev = 1, 5

    severity_range = st.sidebar.slider(
        "Crime Severity",
        min_value=min_sev,
        max_value=max_sev,
        value=(min_sev, max_sev),
    )

    arrest_filter = st.sidebar.selectbox(
        "Arrest Status",
        ["All", "Arrested", "Not Arrested"],
        index=0,
    )

    # -------------------------------------------------------------------
    # Apply filters
    # -------------------------------------------------------------------
    filtered_df = df.copy()

    if selected_clusters and "ML_Cluster" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["ML_Cluster"].isin(selected_clusters)]

    if selected_crimes:
        filtered_df = filtered_df[filtered_df["Primary Type"].isin(selected_crimes)]

    if "Crime_Severity" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["Crime_Severity"] >= severity_range[0])
            & (filtered_df["Crime_Severity"] <= severity_range[1])
        ]

    if arrest_filter == "Arrested" and "Arrest" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Arrest"] == 1]
    elif arrest_filter == "Not Arrested" and "Arrest" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Arrest"] == 0]

    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")

    # -------------------------------------------------------------------
    # Main content
    # -------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🧮 PCA View", "🌈 t-SNE View", "📊 Feature Space", "💾 Download"]
    )

    # ---------------- Tab 1: PCA View ----------------
    with tab1:
        st.header("PCA (2D) Projection")

        if len(filtered_df) > 0 and {"PCA_1", "PCA_2"}.issubset(filtered_df.columns):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("PCA Colored by Cluster")
                fig_pca_cluster = px.scatter(
                    filtered_df,
                    x="PCA_1",
                    y="PCA_2",
                    color="ML_Cluster" if "ML_Cluster" in filtered_df.columns else None,
                    hover_data=[
                        c
                        for c in ["Primary Type", "Crime_Severity", "Arrest"]
                        if c in filtered_df.columns
                    ],
                    title="PCA Projection (Colored by ML Cluster)",
                )
                fig_pca_cluster.update_layout(height=500)
                st.plotly_chart(fig_pca_cluster, use_container_width=True)

            with col2:
                st.subheader("PCA Colored by Crime Type")
                fig_pca_type = px.scatter(
                    filtered_df,
                    x="PCA_1",
                    y="PCA_2",
                    color="Primary Type" if "Primary Type" in filtered_df.columns else None,
                    hover_data=[
                        c
                        for c in ["Crime_Severity", "Arrest"]
                        if c in filtered_df.columns
                    ],
                    title="PCA Projection (Colored by Crime Type)",
                )
                fig_pca_type.update_layout(height=500)
                st.plotly_chart(fig_pca_type, use_container_width=True)
        else:
            st.warning("PCA columns (PCA_1, PCA_2) not available in the dataset")

    # ---------------- Tab 2: t-SNE View ----------------
    with tab2:
        st.header("t-SNE (2D) Embedding")

        if len(filtered_df) > 0 and {"TSNE_1", "TSNE_2"}.issubset(filtered_df.columns):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("t-SNE Colored by Cluster")
                fig_tsne_cluster = px.scatter(
                    filtered_df,
                    x="TSNE_1",
                    y="TSNE_2",
                    color="ML_Cluster" if "ML_Cluster" in filtered_df.columns else None,
                    hover_data=[
                        c
                        for c in ["Primary Type", "Crime_Severity", "Arrest"]
                        if c in filtered_df.columns
                    ],
                    title="t-SNE Embedding (Colored by ML Cluster)",
                )
                fig_tsne_cluster.update_layout(height=500)
                st.plotly_chart(fig_tsne_cluster, use_container_width=True)

            with col2:
                st.subheader("t-SNE Colored by Crime Type")
                fig_tsne_type = px.scatter(
                    filtered_df,
                    x="TSNE_1",
                    y="TSNE_2",
                    color="Primary Type" if "Primary Type" in filtered_df.columns else None,
                    hover_data=[
                        c
                        for c in ["Crime_Severity", "Arrest"]
                        if c in filtered_df.columns
                    ],
                    title="t-SNE Embedding (Colored by Crime Type)",
                )
                fig_tsne_type.update_layout(height=500)
                st.plotly_chart(fig_tsne_type, use_container_width=True)
        else:
            st.warning("t-SNE columns (TSNE_1, TSNE_2) not available in the dataset")

    # ---------------- Tab 3: Feature Space ----------------
    with tab3:
        st.header("Feature Space Distributions")

        if len(filtered_df) > 0:
            col1, col2 = st.columns(2)

            if {"ML_Cluster", "Crime_Severity"}.issubset(filtered_df.columns):
                with col1:
                    st.subheader("Crime Severity by Cluster")
                    fig_sev = px.box(
                        filtered_df,
                        x="ML_Cluster",
                        y="Crime_Severity",
                        color="ML_Cluster",
                        title="Crime Severity Distribution by Cluster",
                        labels={"ML_Cluster": "Cluster"},
                    )
                    fig_sev.update_layout(height=450)
                    st.plotly_chart(fig_sev, use_container_width=True)

            if {"ML_Cluster", "District"}.issubset(filtered_df.columns):
                with col2:
                    st.subheader("District Distribution")
                    fig_dist = px.histogram(
                        filtered_df,
                        x="District",
                        color="ML_Cluster",
                        barmode="group",
                        title="District Distribution by Cluster",
                    )
                    fig_dist.update_layout(height=450)
                    st.plotly_chart(fig_dist, use_container_width=True)

            if {"Ward", "Crime_Severity", "ML_Cluster"}.issubset(filtered_df.columns):
                st.subheader("Ward vs Severity (Colored by Cluster)")
                fig_scatter_fs = px.scatter(
                    filtered_df,
                    x="Ward",
                    y="Crime_Severity",
                    color="ML_Cluster",
                    hover_data=[
                        c
                        for c in ["Primary Type", "Arrest"]
                        if c in filtered_df.columns
                    ],
                    title="Ward vs Crime Severity by Cluster",
                )
                fig_scatter_fs.update_layout(height=450)
                st.plotly_chart(fig_scatter_fs, use_container_width=True)
        else:
            st.warning("No data available with current filters")

    # ---------------- Tab 4: Download ----------------
    with tab4:
        st.header("💾 Download Embedding Data")

        st.markdown(
            """
            Download the filtered dimensionality reduction results for further analysis.
            The CSV file includes PCA and t-SNE coordinates for each record.
            """
        )

        if len(filtered_df) > 0:
            cols = [
                "Latitude",
                "Longitude",
                "Primary Type",
                "Crime_Severity",
                "District",
                "Ward",
                "Arrest",
                "ML_Cluster",
                "PCA_1",
                "PCA_2",
                "TSNE_1",
                "TSNE_2",
            ]
            cols = [c for c in cols if c in filtered_df.columns]

            download_df = filtered_df[cols].copy()
            csv = download_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Embedding Data (CSV)",
                data=csv,
                file_name="dimensionality_reduction_filtered.csv",
                mime="text/csv",
            )

            st.subheader("Data Preview")
            st.dataframe(download_df.head(100), use_container_width=True)
        else:
            st.warning("No data available for download with current filters")

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>Dimensionality Reduction Analysis | PatrolIQ Platform</p>
    </div>
    """,
    unsafe_allow_html=True,
)