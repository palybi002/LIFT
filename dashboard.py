import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(layout="wide", page_title="Experiment Results Dashboard")

st.title("Contrast Experiment Analysis Dashboard")
st.markdown("""
This dashboard visualizes the results from `comparison_results.csv`. 
You can filter by dataset and compare model performance.
""")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('comparison_results.csv')
        # Filter out rows where MSE is missing
        df_clean = df.dropna(subset=['MSE', 'MAE'])
        return df, df_clean
    except FileNotFoundError:
        return None, None

df_raw, df = load_data()

if df_raw is None:
    st.error("File `comparison_results.csv` not found. Please run the analysis script first.")
else:
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Dataset selection
    datasets = sorted(df['Dataset'].unique())
    selected_dataset = st.sidebar.selectbox("Select Dataset", datasets)
    
    # Filter data
    filtered_df = df[df['Dataset'] == selected_dataset]

    # Show raw data toggle
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(df_raw)

    if filtered_df.empty:
        st.warning(f"No valid results found for dataset: {selected_dataset}")
    else:
        # Key Metrics Overview
        st.header(f"Results for {selected_dataset}")
        
        # Best Model Calculation
        best_mse_idx = filtered_df['MSE'].idxmin()
        best_model = filtered_df.loc[best_mse_idx]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model (MSE)", best_model['Model'])
        col1.markdown(f"**MSE**: {best_model['MSE']:.4f}")
        
        col2.metric("Best Model (MAE)", best_model['Model']) # Simplified, usually same model
        col2.markdown(f"**MAE**: {best_model['MAE']:.4f}")

        if 'TrainTime' in filtered_df.columns and not filtered_df['TrainTime'].isnull().all():
             fastest_idx = filtered_df['TrainTime'].idxmin()
             fastest = filtered_df.loc[fastest_idx]
             col3.metric("Fastest Model", fastest['Model'])
             col3.markdown(f"**Time**: {fastest['TrainTime']:.2f}s")

        st.divider()

        # 1. Performance Charts (MSE & MAE)
        st.subheader("Performance Comparison (Lower is Better)")
        
        # Prepare data for Altair
        # We want grouped bars if multiple entries exist, or just bars.
        # Let's aggregate by Model to get Mean performance
        grouped = filtered_df.groupby('Model')[['MSE', 'MAE', 'TrainTime']].mean().reset_index()
        
        # Melt for side-by-side plotting of MSE/MAE if desired, but separate is cleaner
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### MSE Comparison")
            chart_mse = alt.Chart(grouped).mark_bar().encode(
                x=alt.X('Model', sort='y'),
                y='MSE',
                color='Model',
                tooltip=['Model', 'MSE']
            ).interactive()
            st.altair_chart(chart_mse, use_container_width=True)

        with c2:
            st.markdown("### MAE Comparison")
            chart_mae = alt.Chart(grouped).mark_bar().encode(
                x=alt.X('Model', sort='y'),
                y='MAE',
                color='Model',
                tooltip=['Model', 'MAE']
            ).interactive()
            st.altair_chart(chart_mae, use_container_width=True)

        # 2. Efficiency Chart
        if 'TrainTime' in grouped.columns and grouped['TrainTime'].notna().any():
            st.subheader("Efficiency Comparison (Training Time)")
            chart_time = alt.Chart(grouped).mark_bar().encode(
                x=alt.X('Model', sort='y'),
                y=alt.Y('TrainTime', title='Training Time (s)'),
                color='Model',
                tooltip=['Model', 'TrainTime']
            ).interactive()
            st.altair_chart(chart_time, use_container_width=True)
            
        # 3. Scatter Plot: Accuracy vs Efficiency
        if 'TrainTime' in filtered_df.columns:
            st.subheader("Accuracy vs Efficiency Trade-off")
            scatter = alt.Chart(filtered_df).mark_circle(size=100).encode(
                x=alt.X('TrainTime', title='Training Time (s)'),
                y=alt.Y('MSE', title='MSE (Lower is better)'),
                color='Model',
                tooltip=['Model', 'MSE', 'TrainTime', 'Params']
            ).interactive()
            st.altair_chart(scatter, use_container_width=True)

        # 4. Detailed Table
        st.subheader("Detailed Results Table")
        st.dataframe(filtered_df.style.highlight_min(subset=['MSE', 'MAE'], color='lightgreen'))

