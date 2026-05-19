import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(page_title="Prediction Detail", page_icon=":bar_chart:", layout="wide")

st.title("Prediction Detail per Industry")
st.markdown("Detailed analysis for each industry")

# Load models
@st.cache_resource
def load_predictions():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    pred_path = os.path.join(MODELS_DIR, 'predictions.pkl')
    
    if os.path.exists(pred_path):
        return joblib.load(pred_path)
    return None

predictions = load_predictions()

if predictions is None:
    st.error("Models not available. Please run python model_training.py first.")
    st.stop()

# Create comparison dataframe
comparison_data = []
for industry, data in predictions.items():
    growth = ((data['predictions'][-1] - data['predictions'][0]) / data['predictions'][0]) * 100
    comparison_data.append({
        'industry': industry,
        'slope': data['slope'],
        'r2_score': data['r2_score'],
        'pred_2026': data['predictions'][0],
        'pred_2030': data['predictions'][-1],
        'growth_percent': growth,
        'trend': 'Increasing' if data['slope'] > 0 else 'Decreasing'
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('slope', ascending=False)

# Display all industries in grid
st.markdown("### Industry Ranking Based on Growth")

cols = st.columns(3)
for idx, (_, row) in enumerate(comparison_df.head(9).iterrows()):
    with cols[idx % 3]:
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h4>{row['industry']}</h4>
            <p>Trend: {row['slope']:.4f}/year</p>
            <p>R2 Score: {row['r2_score']:.3f}</p>
            <p>Growth: {row['growth_percent']:.1f}%</p>
            <p>{row['trend']}</p>
        </div>
        """, unsafe_allow_html=True)

# Full table
st.markdown("---")
st.markdown("### Complete Table of All Industries")

st.dataframe(
    comparison_df.style.format({
        'slope': '{:.4f}',
        'r2_score': '{:.3f}',
        'pred_2026': '{:.4f}',
        'pred_2030': '{:.4f}',
        'growth_percent': '{:.1f}%'
    }),
    use_container_width=True,
    hide_index=True
)

# Chart all industries
st.markdown("---")
st.markdown("### Trend Comparison of All Industries")

fig = px.bar(
    comparison_df, 
    x='industry', 
    y='slope',
    title='AI Growth Trend per Year (Slope) - All Industries',
    labels={'slope': 'Trend (change per year)', 'industry': 'Industry'},
    color='slope',
    color_continuous_scale='RdYlGn',
    text='slope'
)
fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Growth chart
fig2 = px.bar(
    comparison_df,
    x='industry',
    y='growth_percent',
    title='Predicted Growth 2026 -> 2030 (%) - All Industries',
    labels={'growth_percent': 'Growth (%)', 'industry': 'Industry'},
    color='growth_percent',
    color_continuous_scale='Viridis',
    text='growth_percent'
)
fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig2.update_layout(height=500)
st.plotly_chart(fig2, use_container_width=True)