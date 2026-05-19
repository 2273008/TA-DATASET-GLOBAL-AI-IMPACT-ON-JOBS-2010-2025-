import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(page_title="Comparison", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Industry Comparison")
st.markdown("Compare AI adoption predictions across industry sectors")

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

if predictions is None:
    st.error("Models not available. Please run python model_training.py first.")
    st.stop()

# Multi-select for industries to compare
industries = list(predictions.keys())
selected_industries = st.multiselect(
    "Select industries to compare",
    industries,
    default=industries[:3]
)

if not selected_industries:
    st.warning("Please select at least 1 industry to compare")
    st.stop()

# Prepare comparison data
comparison_data = []
for industry in selected_industries:
    pred_data = predictions[industry]
    pred_years = pred_data['years']
    pred_values = pred_data['predictions']
    
    for year, value in zip(pred_years, pred_values):
        comparison_data.append({
            'industry': industry,
            'year': year,
            'predicted_value': value,
            'slope': pred_data['slope']
        })

comparison_df = pd.DataFrame(comparison_data)

# Line chart comparison
st.markdown("### Prediction Comparison Chart")

fig = px.line(
    comparison_df,
    x='year',
    y='predicted_value',
    color='industry',
    title='AI Intensity Score Prediction Comparison',
    labels={'year': 'Year', 'predicted_value': 'AI Intensity Score', 'industry': 'Industry'},
    markers=True
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Bar chart for 2030
st.markdown("### Predicted Values for 2030")

pred_2030_data = comparison_df[comparison_df['year'] == 2030].copy()
if not pred_2030_data.empty:
    fig_bar = px.bar(
        pred_2030_data,
        x='industry',
        y='predicted_value',
        title='Predicted AI Intensity Score for 2030',
        labels={'predicted_value': '2030 Prediction', 'industry': 'Industry'},
        color='predicted_value',
        color_continuous_scale='Viridis',
        text='predicted_value'
    )
    fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_bar.update_layout(height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

# Ranking table
st.markdown("---")
st.markdown("### Industry Ranking")

ranking_data = []
for industry in predictions:
    pred_data = predictions[industry]
    ranking_data.append({
        'Industry': industry,
        'Trend (slope)': f"{pred_data['slope']:.4f}",
        'R2 Score': f"{pred_data['r2_score']:.3f}",
        '2030 Prediction': f"{pred_data['predictions'][-1]:.4f}",
        'Growth %': f"{((pred_data['predictions'][-1] - pred_data['predictions'][0]) / pred_data['predictions'][0] * 100):.1f}%"
    })

ranking_df = pd.DataFrame(ranking_data)
ranking_df = ranking_df.sort_values('2030 Prediction', ascending=False)

st.dataframe(ranking_df, use_container_width=True, hide_index=True)

# Download button
csv = ranking_df.to_csv(index=False)
st.download_button(
    label="Download Comparison Data (CSV)",
    data=csv,
    file_name="ai_prediction_comparison.csv",
    mime="text/csv"
)