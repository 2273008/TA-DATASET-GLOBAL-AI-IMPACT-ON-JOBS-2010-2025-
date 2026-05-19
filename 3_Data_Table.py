import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Data Viewer", page_icon=":clipboard:", layout="wide")

st.title("Data Viewer & Export")
st.markdown("View and export historical data and prediction results")

# Load data
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    file_path = os.path.join(DATA_DIR, 'ai_impact_jobs_2010_2025.csv')
    df = pd.read_csv(file_path)
    return df

@st.cache_resource
def load_predictions():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    pred_path = os.path.join(MODELS_DIR, 'predictions.pkl')
    
    if os.path.exists(pred_path):
        return joblib.load(pred_path)
    return None

df = load_data()
predictions = load_predictions()

# Tabs
tab1, tab2, tab3 = st.tabs(["Historical Data", "Prediction Results", "Export Data"])

with tab1:
    st.markdown("### Historical AI Impact Jobs Data")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_industry = st.selectbox("Filter Industry", ['All'] + sorted(df['industry'].unique().tolist()))
    with col2:
        if 'year' in df.columns:
            selected_year = st.selectbox("Filter Year", ['All'] + sorted(df['year'].unique().tolist()))
        else:
            selected_year = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    if selected_industry != 'All':
        filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
    if selected_year != 'All' and 'year' in df.columns:
        filtered_df = filtered_df[filtered_df['year'] == selected_year]
    
    st.dataframe(filtered_df, use_container_width=True)
    st.caption(f"Showing {len(filtered_df)} rows of data")

with tab2:
    st.markdown("### Prediction Results by Industry")
    
    if predictions:
        # Create prediction dataframe
        pred_list = []
        for industry, data in predictions.items():
            for i, year in enumerate(data['years']):
                pred_list.append({
                    'industry': industry,
                    'year': year,
                    'predicted_intensity': data['predictions'][i],
                    'slope': data['slope'],
                    'r2_score': data['r2_score']
                })
        
        pred_df = pd.DataFrame(pred_list)
        st.dataframe(pred_df, use_container_width=True)
        
        # Chart
        fig = px.line(
            pred_df,
            x='year',
            y='predicted_intensity',
            color='industry',
            title='Predicted AI Intensity Score by Industry (2026-2030)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No prediction data available. Run python model_training.py first.")

with tab3:
    st.markdown("### Export Data")
    
    # Option to export historical data
    if st.button("Export Historical Data (CSV)"):
        csv_hist = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_hist,
            file_name="ai_historical_data.csv",
            mime="text/csv"
        )
    
    # Option to export predictions
    if predictions and st.button("Export Prediction Data (CSV)"):
        pred_list = []
        for industry, data in predictions.items():
            for i, year in enumerate(data['years']):
                pred_list.append({
                    'industry': industry,
                    'year': year,
                    'predicted_intensity': data['predictions'][i],
                    'slope': data['slope'],
                    'r2_score': data['r2_score']
                })
        pred_export_df = pd.DataFrame(pred_list)
        csv_pred = pred_export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_pred,
            file_name="ai_predictions.csv",
            mime="text/csv"
        )
    
    st.info("Tip: Data can be imported into Excel or other analysis tools.")