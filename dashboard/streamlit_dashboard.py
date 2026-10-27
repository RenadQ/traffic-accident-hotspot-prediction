# Run from the project root:  streamlit run dashboard/streamlit_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from io import BytesIO

# === PAGE CONFIG ===
st.set_page_config(page_title="Accident Data Dashboard", layout="wide")

st.title("🚦 Interactive Accident Data Dashboard")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")
if uploaded_file:
    
    # Load Excel
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ Loaded {uploaded_file.name} with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # --- Convert numeric columns stored as text ---
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # keep non-numeric columns as they are
    
    # --- Convert datetime columns ---
    datetime_cols = [col for col in df.columns if 'Time' in col]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    if 'Start_Time' in df.columns:
        df['Year'] = df['Start_Time'].dt.year
        df['Month'] = df['Start_Time'].dt.month
        df['Day'] = df['Start_Time'].dt.day
        df['Hour'] = df['Start_Time'].dt.hour
        df['Weekday'] = df['Start_Time'].dt.dayofweek
    
    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    
    year_list = sorted(df['Year'].dropna().unique().tolist()) if 'Year' in df.columns else []
    selected_year = st.sidebar.selectbox("Select Year", options=year_list)
    filtered_df = df[df['Year'] == selected_year] if selected_year else df
    
    weather_list = filtered_df['Weather_Condition'].dropna().unique().tolist() if 'Weather_Condition' in filtered_df.columns else []
    selected_weather = st.sidebar.multiselect("Select Weather Condition(s)", options=weather_list, default=weather_list)
    filtered_df = filtered_df[filtered_df['Weather_Condition'].isin(selected_weather)] if selected_weather else filtered_df
    
    st.sidebar.write(f"Filtered dataset has {filtered_df.shape[0]} rows.")
    
    # === SUMMARY STATISTICS ===
    st.subheader("📊 Dataset Summary")
    st.write("**Numeric Summary:**")
    st.dataframe(filtered_df.describe())
    
    st.write("**Missing Values per Column:**")
    st.dataframe(filtered_df.isnull().sum())
    
    # --- OUTLIERS ---
    st.write("**Outlier Detection (IQR Method)**")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_info = []
    for col in numeric_cols:
        Q1 = filtered_df[col].quantile(0.25)
        Q3 = filtered_df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = filtered_df[(filtered_df[col] < Q1 - 1.5*IQR) | (filtered_df[col] > Q3 + 1.5*IQR)]
        outlier_info.append({'Column': col, 'Outliers': len(outliers), 'Percent': len(outliers)/len(filtered_df)*100})
    st.dataframe(pd.DataFrame(outlier_info))
    
    # === HISTOGRAMS ===
    st.subheader("📈 Histograms")
    for col in numeric_cols:
        fig = px.histogram(filtered_df, x=col, nbins=30, title=f"Histogram: {col}", marginal="box", color_discrete_sequence=['skyblue'])
        st.plotly_chart(fig, use_container_width=True)
    
    # === BOX PLOTS ===
    st.subheader("📦 Box Plots")
    for col in numeric_cols:
        fig = px.box(filtered_df, y=col, color_discrete_sequence=['coral'], title=f"Box Plot: {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    # === SCATTER PLOTS vs Severity ===
    if 'Severity' in filtered_df.columns:
        st.subheader("⚡ Scatter Plots vs Severity")
        for col in numeric_cols:
            if col != 'Severity':
                fig = px.scatter(filtered_df, x=col, y='Severity', color='Weather_Condition', hover_data=['City','Description'],
                                 title=f"{col} vs Severity")
                st.plotly_chart(fig, use_container_width=True)
    
    # === TIME-BASED VISUALIZATIONS ===
    st.subheader("⏰ Time-based Visualizations")
    time_cols = ['Hour','Day','Month','Weekday','Year']
    for col in time_cols:
        if col in filtered_df.columns:
            fig = px.histogram(filtered_df, x=col, nbins=24 if col=='Hour' else 12, title=f"Accidents by {col}", color='Severity')
            st.plotly_chart(fig, use_container_width=True)
    
    # === WEATHER VS SEVERITY HEATMAP ===
    if 'Weather_Condition' in filtered_df.columns and 'Severity' in filtered_df.columns:
        st.subheader("🌦️ Weather vs Severity")
        heatmap_data = pd.crosstab(filtered_df['Weather_Condition'], filtered_df['Severity'])
        fig = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Viridis', title="Weather vs Severity Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    
    # === GEOSPATIAL SCATTER ===
    st.subheader("🗺️ Accident Locations")
    if 'Start_Lat' in filtered_df.columns and 'Start_Lng' in filtered_df.columns:
        # plotly >= 6 renamed scatter_mapbox to scatter_map; support both
        new_api = hasattr(px, "scatter_map")
        map_fn = px.scatter_map if new_api else px.scatter_mapbox
        fig = map_fn(filtered_df, lat="Start_Lat", lon="Start_Lng", hover_name="Description",
                     hover_data=["City","Severity","Distance(mi)"], color="Severity",
                     color_continuous_scale=px.colors.cyclical.IceFire, zoom=3, height=500)
        if new_api:
            fig.update_layout(map_style="open-street-map")
        else:
            fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    
    # === DOWNLOAD AUDIT REPORT ===
    st.subheader("📄 Download EDA Audit Report")
    audit_lines = []
    audit_lines.append(f"Dataset: {uploaded_file.name}\nRows: {filtered_df.shape[0]}, Columns: {filtered_df.shape[1]}\n")
    audit_lines.append("Numeric Summary:\n" + filtered_df.describe().to_string() + "\n")
    audit_lines.append("Missing Values:\n" + filtered_df.isnull().sum().to_string() + "\n")
    audit_lines.append("Outliers per numeric column:\n" + pd.DataFrame(outlier_info).to_string() + "\n")
    
    audit_txt = "\n".join(audit_lines)
    
    b = BytesIO()
    b.write(audit_txt.encode())
    b.seek(0)
    
    st.download_button("Download EDA Audit Report (.txt)", b, file_name="EDA_audit_report.txt")
