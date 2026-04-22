# pages/3_Dimensionality_Reduction.py

"""
PatrolIQ - Dimensionality Reduction Analysis
Page 3: PCA and t-SNE visualization of crime patterns

Refactored to:
- Remove CSV dependencies
- Use synthetic ML dataset
- Apply PCA (2D) and t-SNE (2D) on numeric features
- Optionally use pre-trained PCA/TSNE models from artifacts/*.pkl
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

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
# Optional pre-trained models
# -------------------------------------------------------------------
PCA_MODEL_PATH = os.path.join("artifacts", "pca_model.pkl")
TSNE_MODEL_PATH = os.path.join("artifacts", "tsne_model.pkl")


@st.cache_resource
def load_reduction_model(path: str):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


pca_model_pretrained = load_reduction_model(PCA_MODEL_PATH)
tsne_model_pretrained = load_reduction_model(TSNE_MODEL_PATH)

if pca_model_pretrained is None:
    st.info(
        "No pre-trained PCA model found in `artifacts/pca_model.pkl`. "
        "Using in-app PCA for visualization."
    )
if tsne_model_pretrained is None:
    st.info(
        "No pre-trained t-SNE model found in `artifacts/tsne_model.pkl`. "
        "Using in-app t-SNE for visualization."
    )

# -------------------------------------------------------------------
# Synthetic ML dataset
# -------------------------------------------------------------------
@st.cache_data
def create_synthetic_ml_data(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    latitudes = rng.uniform(41.6, 42.05, size=n_rows)
    longitudes = rng.uniform(-87.95, -87.5, size=n_rows)
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
            "Latitude": latitudes,
            "Longitude": longitudes,
            "Primary Type": primary_types,
            "Crime_Severity": crime_severity,
            "District": districts,
            "Ward": wards,
            "Arrest": arrests,
        }
    )

    # Simple clustering for coloring (e.g., spatial clusters)
    X_cluster = df[["Latitude", "Longitude"]].values
    km = KMeans(n_clusters=6, random_state=42)
    df["ML_Cluster"] = km.fit_predict(X_cluster)

    return df


# -------------------------------------------------------------------
# Apply PCA and t-SNE
# -------------------------------------------------------------------
@st.cache_data
def apply_pca_tsne(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["Crime_Severity", "Arrest", "District", "Ward"]
    X_num = df[numeric_cols].astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    # PCA
    if pca_model_pretrained is not None and hasattr(pca_model_pretrained, "transform"):
        try:
            pca_result = pca_model_pretrained.transform(X_scaled)
            pca_result = pca_result[:, :2]
        except Exception:
            pca = PCA(n_components=2, random_state=42)
            pca_result = pca.fit_transform(X_scaled)
    else:
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(X_scaled)

    # t-SNE
    if tsne_model_pretrained is not None and hasattr(tsne_model_pretrained, "fit_transform"):
        try:
            tsne_result = tsne_model_pretrained.fit_transform(X_scaled)
        except Exception:
            tsne = TSNE(
                n_components=2,
                learning_rate="auto",
                init="random",
                perplexity=30,
                random_state=42,
            )
            tsne_result = tsne.fit_transform(X_scaled)
    else:
        tsne = TSNE(
            n_components=2,
            learning_rate="auto",
            init="random",
            perplexity=30,
            random_state=42,
        )
        tsne_result = tsne.fit_transform(X_scaled)

    df_out = df.copy()
    df_out["PCA_1"] = pca_result[:, 0]
    df_out["PCA_2"] = pca_result[:, 1]
    df_out["TSNE_1"] = tsne_result[:, 0]
    df_out["TSNE_2"] = tsne_result[:, 1]

    return df_out


# -------------------------------------------------------------------
# Build dataset
# -------------------------------------------------------------------
base_df = create_synthetic_ml_data()
df = apply_pca_tsne(base_df)

st.success(f"✅ Synthetic ML dataset ready with {len(df):,} records and PCA/t-SNE embeddings")

# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------
st.sidebar.header("🔍 Filters")

cluster_ids = sorted(df["ML_Cluster"].unique())
selected_clusters = st.sidebar.multiselect(
    "Select Clusters",
    cluster_ids,
    default=cluster_ids[: min(5, len(cluster_ids))],
)

crime_types = sorted(df["Primary Type"].unique())
selected_crimes = st.sidebar.multiselect(
    "Filter by Crime Type",
    crime_types,
    default=[],
)

severity_range = st.sidebar.slider(
    "Crime Severity",
    min_value=1,
    max_value=5,
    value=(1, 5),
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

if selected_clusters:
    filtered_df = filtered_df[filtered_df["ML_Cluster"].isin(selected_clusters)]

if selected_crimes:
    filtered_df = filtered_df[filtered_df["Primary Type"].isin(selected_crimes)]

filtered_df = filtered_df[
    (filtered_df["Crime_Severity"] >= severity_range[0])
    & (filtered_df["Crime_Severity"] <= severity_range[1])
]

if arrest_filter == "Arrested":
    filtered_df = filtered_df[filtered_df["Arrest"] == 1]
elif arrest_filter == "Not Arrested":
    filtered_df = filtered_df[filtered_df["Arrest"] == 0]

st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")

# -------------------------------------------------------------------
# Main content tabs
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🧮 PCA View", "🌈 t-SNE View", "📊 Feature Space", "💾 Download"]
)

# ---------------- Tab 1: PCA View ----------------
with tab1:
    st.header("PCA (2D) Projection")

    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("PCA Colored by Cluster")
            fig_pca_cluster = px.scatter(
                filtered_df,
                x="PCA_1",
                y="PCA_2",
                color="ML_Cluster",
                hover_data=["Primary Type", "Crime_Severity", "Arrest"],
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
                color="Primary Type",
                hover_data=["Crime_Severity", "Arrest"],
                title="PCA Projection (Colored by Crime Type)",
            )
            fig_pca_type.update_layout(height=500)
            st.plotly_chart(fig_pca_type, use_container_width=True)
    else:
        st.warning("No data available with current filters")

# ---------------- Tab 2: t-SNE View ----------------
with tab2:
    st.header("t-SNE (2D) Embedding")

    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("t-SNE Colored by Cluster")
            fig_tsne_cluster = px.scatter(
                filtered_df,
                x="TSNE_1",
                y="TSNE_2",
                color="ML_Cluster",
                hover_data=["Primary Type", "Crime_Severity", "Arrest"],
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
                color="Primary Type",
                hover_data=["Crime_Severity", "Arrest"],
                title="t-SNE Embedding (Colored by Crime Type)",
            )
            fig_tsne_type.update_layout(height=500)
            st.plotly_chart(fig_tsne_type, use_container_width=True)
    else:
        st.warning("No data available with current filters")

# ---------------- Tab 3: Feature Space ----------------
with tab3:
    st.header("Feature Space Distributions")

    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)

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

        st.subheader("Ward vs Severity (Colored by Cluster)")
        fig_scatter_fs = px.scatter(
            filtered_df,
            x="Ward",
            y="Crime_Severity",
            color="ML_Cluster",
            hover_data=["Primary Type", "Arrest"],
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
        download_df = filtered_df[
            [
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
        ].copy()

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