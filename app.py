# app.py

"""
PatrolIQ - Smart Safety Analytics Platform
Main application home page

Final architecture:
- 01_data_preprocessing.ipynb trains/cleans on FULL dataset
- Notebook saves a sampled, processed dataset to artifacts/processed_crime_data.csv
- This app ONLY reads that processed sample (no heavy ETL or training here)
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="PatrolIQ - Smart Safety Analytics Platform",
    page_icon="🚨",
    layout="wide",
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
# Base directory = directory containing this app.py
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = BASE_DIR / "artifacts" / "processed_crime_data.csv"

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
@st.cache_data
def load_processed_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Debug info to help when deployed
        st.error(
            f"Processed dataset not found at: {path}\n"
            "Please run 01_data_preprocessing.ipynb locally and commit "
            "artifacts/processed_crime_data.csv."
        )
        st.info(f"Current working directory: {Path.cwd()}")
        st.info(f"Files in artifacts/: {list((BASE_DIR / 'artifacts').glob('*'))}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Optional: parse dates if present
    for col in ["Date", "DateTime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


df_processed = load_processed_data(PROCESSED_PATH)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-bottom: 2rem;">
        <h1 style="margin-bottom:0.25rem;">🚨 PatrolIQ</h1>
        <h3 style="color:#9ca3af; font-weight:400; margin-top:0;">
            Smart Urban Safety Analytics Platform
        </h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mission section
st.markdown(
    """
    <div style="
        background: rgba(31, 41, 55, 0.85);
        border-radius: 14px;
        padding: 1.75rem 2rem;
        border: 1px solid rgba(59, 130, 246, 0.35);
        margin-bottom: 2rem;
    ">
        <h3 style="margin: 0 0 0.75rem 0; color: #e5e7eb;">🎯 Mission</h3>
        <p style="margin: 0; color: #d1d5db; line-height: 1.75;">
            Leverage unsupervised machine learning to analyze crime patterns and optimize
            police resource allocation in Chicago. This platform provides actionable
            insights to make cities safer through data-driven decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Content depending on data
# -------------------------------------------------------------------
if df_processed.empty:
    st.warning(
        "No processed data available. Once you generate and commit "
        "`artifacts/processed_crime_data.csv`, this page will show "
        "high-level metrics and overview charts."
    )
else:
    st.success(f"✅ Loaded {len(df_processed):,} processed crime records")

    # Sidebar filters (keep light – heavy filtering is on pages)
    st.sidebar.header("🔍 Quick Filters")

    crime_types = (
        sorted(df_processed["Primary Type"].dropna().unique())
        if "Primary Type" in df_processed.columns
        else []
    )
    selected_crimes = st.sidebar.multiselect(
        "Crime Type",
        crime_types,
        default=crime_types[:5] if crime_types else [],
    )

    if "District" in df_processed.columns:
        districts = sorted(df_processed["District"].dropna().unique())
        selected_districts = st.sidebar.multiselect(
            "District",
            districts,
            default=[],
        )
    else:
        selected_districts = []

    filtered_df = df_processed.copy()

    if selected_crimes:
        filtered_df = filtered_df[filtered_df["Primary Type"].isin(selected_crimes)]
    if selected_districts and "District" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["District"].isin(selected_districts)]

    # -------------------------------------------------------------------
    # KPI cards
    # -------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Crimes (sample)", f"{len(filtered_df):,}")

    with col2:
        if "Crime_Severity" in filtered_df.columns:
            avg_severity = filtered_df["Crime_Severity"].mean()
            st.metric("Avg Severity", f"{avg_severity:.2f}/5")
        else:
            st.metric("Avg Severity", "N/A")

    with col3:
        if "Arrest" in filtered_df.columns and len(filtered_df) > 0:
            arrest_rate = filtered_df["Arrest"].mean() * 100
            st.metric("Arrest Rate", f"{arrest_rate:.1f}%")
        else:
            st.metric("Arrest Rate", "N/A")

    with col4:
        if "District" in filtered_df.columns:
            st.metric("Districts Covered", filtered_df["District"].nunique())
        else:
            st.metric("Districts Covered", "N/A")

    st.markdown("---")

    # -------------------------------------------------------------------
    # Overview charts
    # -------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Crime Breakdown", "📈 Temporal Trend"])

    with tab1:
        st.subheader("Top Crime Types")
        if "Primary Type" in filtered_df.columns:
            ct = (
                filtered_df["Primary Type"]
                .value_counts()
                .reset_index(name="Count")
                .rename(columns={"index": "Primary Type"})
                .head(15)
            )
            fig = px.bar(
                ct,
                x="Primary Type",
                y="Count",
                title="Top Crime Types (sample)",
            )
            fig.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'Primary Type' not found in processed dataset.")

    with tab2:
        st.subheader("Crimes Over Time")
        if "Date" in filtered_df.columns:
            ts = (
                filtered_df.set_index("Date")
                .resample("W")
                .size()
                .reset_index(name="Count")
            )
            fig = px.line(
                ts,
                x="Date",
                y="Count",
                title="Weekly Crime Count (sample)",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        elif "DateTime" in filtered_df.columns:
            ts = (
                filtered_df.set_index("DateTime")
                .resample("W")
                .size()
                .reset_index(name="Count")
            )
            fig = px.line(
                ts,
                x="DateTime",
                y="Count",
                title="Weekly Crime Count (sample)",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No datetime column ('Date' or 'DateTime') found for temporal trend.")

    st.markdown("---")
    st.caption("PatrolIQ · Smart Urban Safety Analytics Platform")