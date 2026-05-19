import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI PATH ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

st.set_page_config(
    page_title="AI Impact on Jobs Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .main-header p {
        color: #a0a0a0;
        margin: 0;
    }
    .method-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #4a4a5a;
    }
    .method-card h4 {
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .method-card p {
        color: #d0d0d0;
        margin: 0.3rem 0;
    }
    .method-card hr {
        margin: 0.8rem 0;
        border-color: #4a4a5a;
    }
    .best-method {
        background-color: #00cc96;
        color: #1a1a2e;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.75rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #00cc96;
    }
    .prediction-value {
        font-size: 1.3rem;
        font-weight: bold;
        color: #ff6b6b;
    }
    .nav-button {
        background-color: #00cc96;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.2rem 0;
        cursor: pointer;
    }
    .section-title {
        background-color: #2d2d3d;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .filter-box {
        background-color: #1e1e2a;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-header">
    <h1>AI Impact on Jobs Dashboard</h1>
    <p>Analisis Dampak AI terhadap Pekerjaan | Prediksi Adopsi AI 2026-2030 | 4 Metode Prediksi</p>
</div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    file_path = os.path.join(DATA_DIR, 'ai_impact_jobs_2010_2025.csv')
    df = pd.read_csv(file_path)
    return df

@st.cache_resource
def load_models():
    model_path = os.path.join(MODELS_DIR, 'all_models.pkl')
    pred_path = os.path.join(MODELS_DIR, 'all_predictions.pkl')
    comparison_path = os.path.join(MODELS_DIR, 'model_comparison.csv')
    ranking_path = os.path.join(MODELS_DIR, 'method_ranking.csv')
    
    if not os.path.exists(model_path):
        st.error("Models not available. Please run python model_training_4methods.py first.")
        return None, None, None, None
    
    models = joblib.load(model_path)
    predictions = joblib.load(pred_path)
    
    comparison_df = pd.read_csv(comparison_path) if os.path.exists(comparison_path) else None
    ranking_df = pd.read_csv(ranking_path) if os.path.exists(ranking_path) else None
    
    return models, predictions, comparison_df, ranking_df

# Load all data
df = load_data()
models, predictions, comparison_df, ranking_df = load_models()

if models is None:
    st.stop()

# ==================== SIDEBAR NAVIGATION ====================
with st.sidebar:
    st.markdown("## Navigasi")
    
    # Navigation buttons
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <a href="#data-overview" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center;">
                Data Overview
            </div>
        </a>
        <a href="#automation-risk" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center;">
                Automation Risk Analysis
            </div>
        </a>
        <a href="#displacement-risk" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center;">
                Displacement Risk by Industry
            </div>
        </a>
        <a href="#ai-intensity" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center;">
                AI Intensity Analysis
            </div>
        </a>
        <a href="#adoption-stage" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center;">
                Adoption Stage
            </div>
        </a>
        <a href="#salary-analysis" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center;">
                Salary Analysis
            </div>
        </a>
        <a href="#reskilling" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center;">
                Reskilling Requirement
            </div>
        </a>
        <a href="#prediction" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center;">
                4 Methods Prediction
            </div>
        </a>
        <a href="#comparison" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center;">
                Global Comparison
            </div>
        </a>
        <a href="#unique-data" style="text-decoration: none;">
            <div style="background: linear-gradient(135deg, #ff6b6b, #ffa500); padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: white; font-weight: bold;">
                Unique Data Stories
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== FILTERS ====================
    st.markdown("## Filter Data")
    
    # Industry filter
    industries = df['industry'].unique().tolist()
    selected_industries = st.multiselect(
        "Industry",
        industries,
        default=industries[:3]
    )
    
    # Job Title filter (from original dataset)
    if 'job_title' in df.columns:
        job_titles = df['job_title'].unique().tolist()
        selected_job_titles = st.multiselect(
            "Job Title",
            job_titles,
            default=[]
        )
    
    # Company Size filter
    if 'company_size' in df.columns:
        company_sizes = df['company_size'].unique().tolist()
        selected_sizes = st.multiselect(
            "Company Size",
            company_sizes,
            default=[]
        )
    
    # Seniority Level filter
    if 'seniority_level' in df.columns:
        seniority_levels = df['seniority_level'].unique().tolist()
        selected_seniority = st.multiselect(
            "Seniority Level",
            seniority_levels,
            default=[]
        )
    
    # AI Displacement Risk filter
    displacement_risks = df['ai_job_displacement_risk'].unique().tolist()
    selected_risks = st.multiselect(
        "Displacement Risk",
        displacement_risks,
        default=displacement_risks
    )
    
    # Adoption Stage filter
    adoption_stages = df['industry_ai_adoption_stage'].unique().tolist()
    selected_stages = st.multiselect(
        "Adoption Stage",
        adoption_stages,
        default=adoption_stages
    )
    
    st.markdown("---")
    st.markdown("### Metode Prediksi")
    st.info("""
    1. Linear Regression - Simple linear trend
    2. Random Forest - Ensemble of decision trees
    3. Gradient Boosting - Sequential tree building
    4. Polynomial Regression - Curve fitting (degree 2)
    """)

# Apply filters
filtered_df = df.copy()
if selected_industries:
    filtered_df = filtered_df[filtered_df['industry'].isin(selected_industries)]
if selected_job_titles and 'job_title' in df.columns:
    filtered_df = filtered_df[filtered_df['job_title'].isin(selected_job_titles)]
if selected_sizes and 'company_size' in df.columns:
    filtered_df = filtered_df[filtered_df['company_size'].isin(selected_sizes)]
if selected_seniority and 'seniority_level' in df.columns:
    filtered_df = filtered_df[filtered_df['seniority_level'].isin(selected_seniority)]
if selected_risks:
    filtered_df = filtered_df[filtered_df['ai_job_displacement_risk'].isin(selected_risks)]
if selected_stages:
    filtered_df = filtered_df[filtered_df['industry_ai_adoption_stage'].isin(selected_stages)]

# ==================== SECTION 1: DATA OVERVIEW ====================
st.markdown('<div id="data-overview" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## Data Overview")
st.markdown(f"Menampilkan **{len(filtered_df)}** dari **{len(df)}** total records")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    st.metric("Industries", df['industry'].nunique())
with col3:
    st.metric("Unique Job Titles", df['job_title'].nunique() if 'job_title' in df.columns else "N/A")
with col4:
    st.metric("Avg Salary (USD)", f"${df['salary_usd'].mean():,.0f}")

# Show filtered data preview
with st.expander("View Filtered Data"):
    st.dataframe(filtered_df.head(100), use_container_width=True)

st.markdown("---")

# ==================== SECTION 2: AUTOMATION RISK ====================
st.markdown('<div id="automation-risk" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## Automation Risk Analysis")

col_auto1, col_auto2 = st.columns(2)

with col_auto1:
    fig_auto_box = px.box(
        filtered_df, x='industry', y='automation_risk_score',
        title='Automation Risk Score by Industry',
        color='industry',
        labels={'automation_risk_score': 'Risk Score (0-1)', 'industry': 'Industry'}
    )
    fig_auto_box.update_layout(height=500, template='plotly_dark')
    st.plotly_chart(fig_auto_box, use_container_width=True)

with col_auto2:
    # Summary statistics by industry
    auto_stats = filtered_df.groupby('industry')['automation_risk_score'].agg(['mean', 'std', 'min', 'max']).round(3)
    auto_stats = auto_stats.sort_values('mean', ascending=False)
    st.dataframe(auto_stats, use_container_width=True)
    
    # Highest risk industry
    highest_risk = auto_stats.index[0] if len(auto_stats) > 0 else "N/A"
    st.info(f"**Industri dengan risiko otomatisasi tertinggi:** {highest_risk}")

st.markdown("---")

# ==================== SECTION 3: DISPLACEMENT RISK ====================
st.markdown('<div id="displacement-risk" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## AI Job Displacement Risk by Industry")

# Create displacement risk by industry
displacement_by_industry = filtered_df.groupby(['industry', 'ai_job_displacement_risk']).size().reset_index(name='count')

fig_displacement = px.bar(
    displacement_by_industry,
    x='industry',
    y='count',
    color='ai_job_displacement_risk',
    title='AI Job Displacement Risk Distribution by Industry',
    labels={'count': 'Number of Jobs', 'industry': 'Industry', 'ai_job_displacement_risk': 'Displacement Risk'},
    barmode='stack',
    color_discrete_map={'Low': '#00cc96', 'Medium': '#ffa500', 'High': '#ff6b6b'}
)
fig_displacement.update_layout(height=500, template='plotly_dark')
st.plotly_chart(fig_displacement, use_container_width=True)

# Percentage table
displacement_pct = pd.crosstab(
    filtered_df['industry'], 
    filtered_df['ai_job_displacement_risk'], 
    normalize='index'
).mul(100).round(1)

st.markdown("### Persentase Risiko Perpindahan Pekerjaan per Industri (%)")
st.dataframe(displacement_pct, use_container_width=True)

st.markdown("---")

# ==================== SECTION 4: AI INTENSITY ====================
st.markdown('<div id="ai-intensity" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## AI Intensity Analysis")

col_int1, col_int2 = st.columns(2)

with col_int1:
    fig_intensity_box = px.box(
        filtered_df, x='industry', y='ai_intensity_score',
        title='AI Intensity Score by Industry',
        color='industry',
        labels={'ai_intensity_score': 'AI Intensity (0-1)', 'industry': 'Industry'}
    )
    fig_intensity_box.update_layout(height=500, template='plotly_dark')
    st.plotly_chart(fig_intensity_box, use_container_width=True)

with col_int2:
    intensity_stats = filtered_df.groupby('industry')['ai_intensity_score'].agg(['mean', 'std', 'min', 'max']).round(3)
    intensity_stats = intensity_stats.sort_values('mean', ascending=False)
    st.dataframe(intensity_stats, use_container_width=True)
    
    highest_intensity = intensity_stats.index[0] if len(intensity_stats) > 0 else "N/A"
    st.success(f"**Industri dengan intensitas AI tertinggi:** {highest_intensity}")

st.markdown("---")

# ==================== SECTION 5: ADOPTION STAGE ====================
st.markdown('<div id="adoption-stage" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## AI Adoption Stage per Industry")

adoption_by_industry = filtered_df.groupby(['industry', 'industry_ai_adoption_stage']).size().reset_index(name='count')

fig_adoption = px.bar(
    adoption_by_industry,
    x='industry',
    y='count',
    color='industry_ai_adoption_stage',
    title='AI Adoption Stage by Industry',
    labels={'count': 'Number of Jobs', 'industry': 'Industry', 'industry_ai_adoption_stage': 'Adoption Stage'},
    barmode='stack',
    category_orders={'industry_ai_adoption_stage': ['Emerging', 'Growing', 'Mature']}
)
fig_adoption.update_layout(height=500, template='plotly_dark')
st.plotly_chart(fig_adoption, use_container_width=True)

# Pie chart for overall adoption
adoption_overall = filtered_df['industry_ai_adoption_stage'].value_counts().reset_index()
adoption_overall.columns = ['stage', 'count']

fig_adoption_pie = px.pie(
    adoption_overall,
    values='count',
    names='stage',
    title='Overall AI Adoption Stage Distribution',
    color_discrete_sequence=['#ffa500', '#00cc96', '#ff6b6b']
)
fig_adoption_pie.update_layout(height=450, template='plotly_dark')
st.plotly_chart(fig_adoption_pie, use_container_width=True)

st.markdown("---")

# ==================== SECTION 6: SALARY ANALYSIS ====================
st.markdown('<div id="salary-analysis" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## Salary Analysis")

col_sal1, col_sal2 = st.columns(2)

with col_sal1:
    fig_salary_box = px.box(
        filtered_df, x='industry', y='salary_usd',
        title='Salary Distribution by Industry (USD)',
        color='industry',
        labels={'salary_usd': 'Salary (USD)', 'industry': 'Industry'}
    )
    fig_salary_box.update_layout(height=500, template='plotly_dark')
    st.plotly_chart(fig_salary_box, use_container_width=True)

with col_sal2:
    salary_stats = filtered_df.groupby('industry')['salary_usd'].agg(['mean', 'median', 'min', 'max']).round(0).astype(int)
    salary_stats = salary_stats.sort_values('mean', ascending=False)
    st.dataframe(salary_stats, use_container_width=True)
    
    highest_salary = salary_stats.index[0] if len(salary_stats) > 0 else "N/A"
    st.success(f"**Industri dengan gaji tertinggi:** {highest_salary}")

# Salary by displacement risk
st.markdown("### Salary vs Displacement Risk")
fig_salary_risk = px.box(
    filtered_df, x='ai_job_displacement_risk', y='salary_usd',
    title='Salary Distribution by Displacement Risk Level',
    color='ai_job_displacement_risk',
    labels={'salary_usd': 'Salary (USD)', 'ai_job_displacement_risk': 'Displacement Risk'},
    color_discrete_map={'Low': '#00cc96', 'Medium': '#ffa500', 'High': '#ff6b6b'}
)
fig_salary_risk.update_layout(height=450, template='plotly_dark')
st.plotly_chart(fig_salary_risk, use_container_width=True)

st.markdown("---")

# ==================== SECTION 7: RESKILLING ====================
st.markdown('<div id="reskilling" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## Reskilling Requirement")

col_res1, col_res2 = st.columns(2)

with col_res1:
    reskilling_by_industry = filtered_df.groupby(['industry', 'reskilling_required']).size().reset_index(name='count')
    
    fig_reskilling = px.bar(
        reskilling_by_industry,
        x='industry',
        y='count',
        color='reskilling_required',
        title='Reskilling Requirement by Industry',
        labels={'count': 'Number of Jobs', 'industry': 'Industry', 'reskilling_required': 'Reskilling Required'},
        barmode='stack',
        color_discrete_map={True: '#ff6b6b', False: '#00cc96'}
    )
    fig_reskilling.update_layout(height=500, template='plotly_dark')
    st.plotly_chart(fig_reskilling, use_container_width=True)

with col_res2:
    reskilling_pct = pd.crosstab(
        filtered_df['industry'], 
        filtered_df['reskilling_required'], 
        normalize='index'
    ).mul(100).round(1)
    reskilling_pct.columns = ['No Reskilling (%)', 'Reskilling Required (%)']
    reskilling_pct = reskilling_pct.sort_values('Reskilling Required (%)', ascending=False)
    st.dataframe(reskilling_pct, use_container_width=True)
    
    # Overall reskilling percentage
    total_reskill = filtered_df['reskilling_required'].sum()
    total_records = len(filtered_df)
    reskill_pct = (total_reskill / total_records) * 100 if total_records > 0 else 0
    
    st.metric("Overall Reskilling Required", f"{reskill_pct:.1f}%")

st.markdown("---")

# ==================== SECTION 8: PREDICTION 4 METHODS ====================
st.markdown('<div id="prediction" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## 4 Methods Prediction Comparison")

# Select industry for prediction
pred_industries = list(predictions['linear'].keys())
selected_pred_industry = st.selectbox("Select Industry for Prediction", pred_industries, key="pred_industry")

if selected_pred_industry:
    methods = ['linear', 'random_forest', 'gradient_boosting', 'polynomial']
    method_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Polynomial']
    
    # Find best method
    best_method_idx = 0
    best_r2 = -1
    method_r2 = []
    
    for i, method in enumerate(methods):
        pred_data = predictions[method][selected_pred_industry]
        r2 = pred_data['r2_score']
        method_r2.append((i, r2))
        if r2 > best_r2:
            best_r2 = r2
            best_method_idx = i
    
    # Display cards
    cols = st.columns(4)
    for i, (method, name) in enumerate(zip(methods, method_names)):
        pred_data = predictions[method][selected_pred_industry]
        is_best = (i == best_method_idx)
        
        with cols[i]:
            if is_best:
                st.markdown(f"""
                <div class="method-card" style="border: 2px solid #00cc96;">
                    <h4>{name}</h4>
                    <span class="best-method">BEST METHOD</span>
                    <p><strong>R2 Score</strong></p>
                    <p class="metric-value">{pred_data['r2_score']:.4f}</p>
                    <p><strong>MAE</strong><br>{pred_data['mae']:.4f}</p>
                    <p><strong>RMSE</strong><br>{pred_data['rmse']:.4f}</p>
                    <hr>
                    <p><strong>2030 Prediction</strong></p>
                    <p class="prediction-value">{pred_data['predictions'][-1]:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="method-card">
                    <h4>{name}</h4>
                    <p><strong>R2 Score</strong><br>{pred_data['r2_score']:.4f}</p>
                    <p><strong>MAE</strong><br>{pred_data['mae']:.4f}</p>
                    <p><strong>RMSE</strong><br>{pred_data['rmse']:.4f}</p>
                    <hr>
                    <p><strong>2030 Prediction</strong><br><span style="font-size: 1.1rem; font-weight: bold;">{pred_data['predictions'][-1]:.4f}</span></p>
                </div>
                """, unsafe_allow_html=True)
    
    # Historical and prediction chart
    industry_data = df[df['industry'] == selected_pred_industry].copy()
    if 'year' not in industry_data.columns:
        np.random.seed(42)
        industry_data['year'] = np.random.randint(2010, 2026, len(industry_data))
    
    yearly_avg = industry_data.groupby('year')['ai_intensity_score'].mean().reset_index()
    yearly_avg = yearly_avg.sort_values('year')
    
    # Prepare prediction data
    pred_years = [2026, 2027, 2028, 2029, 2030]
    all_predictions_df = pd.DataFrame({'year': pred_years})
    
    for method, name in zip(methods, method_names):
        pred_data = predictions[method][selected_pred_industry]
        all_predictions_df[name] = pred_data['predictions']
    
    # Plot
    fig_pred_line = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, name in enumerate(method_names):
        fig_pred_line.add_trace(go.Scatter(
            x=all_predictions_df['year'],
            y=all_predictions_df[name],
            mode='lines+markers',
            name=name,
            line=dict(width=2, color=colors[i]),
            marker=dict(size=8, color=colors[i])
        ))
    
    fig_pred_line.update_layout(
        title=f'AI Intensity Score Predictions - {selected_pred_industry}',
        xaxis_title='Year',
        yaxis_title='AI Intensity Score',
        height=450,
        template='plotly_dark'
    )
    st.plotly_chart(fig_pred_line, use_container_width=True)
    
    # Prediction table
    st.markdown("### Prediction Table")
    display_table = all_predictions_df.copy()
    for col in method_names:
        display_table[col] = display_table[col].map(lambda x: f"{x:.4f}")
    st.dataframe(display_table, use_container_width=True, hide_index=True)

st.markdown("---")

# ==================== SECTION 9: GLOBAL COMPARISON ====================
st.markdown('<div id="comparison" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## Global Method Ranking")

if ranking_df is not None:
    fig_rank = px.bar(
        ranking_df,
        x='Method',
        y='Avg R2 Score',
        title='Average R2 Score Across All Industries',
        labels={'Avg R2 Score': 'Average R2 Score', 'Method': 'Method'},
        color='Avg R2 Score',
        color_continuous_scale='RdYlGn',
        text='Avg R2 Score',
        template='plotly_dark'
    )
    fig_rank.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_rank.update_layout(height=450)
    st.plotly_chart(fig_rank, use_container_width=True)
    
    st.dataframe(
        ranking_df.style.format({
            'Avg R2 Score': '{:.4f}',
            'Avg MAE': '{:.4f}',
            'Avg RMSE': '{:.4f}'
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Ranking data not available. Please run model training first.")

# ==================== SECTION 10: UNIQUE DATA STORIES ====================
st.markdown('<div id="unique-data" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("""
## 5 Unique Data Stories
Analisis mendalam dataset mengungkap 5 anomali dan kontradiksi unik yang menantang berbagai asumsi umum terkait dampak AI.
""")

story_tab1, story_tab2, story_tab3, story_tab4, story_tab5 = st.tabs([
    "Story 1: Salary-Automation Paradox",
    "Story 2: Seniority Salary Inversion",
    "Story 3: Adoption Stage Contradiction",
    "Story 4: Regional Digital Divide",
    "Story 5: Reskilling Blind Spot"
])

# ==================== STORY 1: SALARY-AUTOMATION PARADOX ====================
with story_tab1:
    st.markdown("""
    ### The Salary-Automation Paradox
    **Temuan:** Pekerjaan bergaji tinggi (kuartil atas ≥Q75) justru memiliki risiko otomatisasi yang **juga sangat tinggi** (≥Q75).  
    Asumsi umum bahwa "gaji tinggi = aman dari AI" ternyata **tidak selalu benar**.
    """)
    
    # Calculate paradox data
    q75_sal = df['salary_usd'].quantile(0.75)
    q75_auto = df['automation_risk_score'].quantile(0.75)
    
    df['paradox_group'] = 'Normal'
    df.loc[(df['salary_usd'] >= q75_sal) & (df['automation_risk_score'] >= q75_auto), 'paradox_group'] = 'Paradox (High Salary + High Risk)'
    df.loc[(df['salary_usd'] >= q75_sal) & (df['automation_risk_score'] < q75_auto), 'paradox_group'] = 'High Salary, Low Risk'
    df.loc[(df['salary_usd'] < q75_sal) & (df['automation_risk_score'] >= q75_auto), 'paradox_group'] = 'Low Salary, High Risk'
    
    paradox_df = df[df['paradox_group'] == 'Paradox (High Salary + High Risk)']
    
    # Metrics
    s1_col1, s1_col2, s1_col3, s1_col4 = st.columns(4)
    with s1_col1:
        st.metric("Data Paradoks", f"{len(paradox_df):,}", f"{len(paradox_df)/len(df)*100:.1f}% dari total")
    with s1_col2:
        st.metric("Avg Salary Paradoks", f"${paradox_df['salary_usd'].mean():,.0f}")
    with s1_col3:
        st.metric("Avg Auto Risk", f"{paradox_df['automation_risk_score'].mean():.3f}")
    with s1_col4:
        st.metric("Threshold Salary", f"${q75_sal:,.0f}", f"Auto Risk ≥ {q75_auto}")
    
    s1_left, s1_right = st.columns(2)
    
    with s1_left:
        # Scatter plot: Salary vs Automation Risk
        fig_s1_scatter = px.scatter(
            df, x='salary_usd', y='automation_risk_score',
            color='paradox_group',
            title='Salary vs Automation Risk Score — Paradox Highlighted',
            labels={'salary_usd': 'Salary (USD)', 'automation_risk_score': 'Automation Risk Score'},
            color_discrete_map={
                'Paradox (High Salary + High Risk)': '#ff4757',
                'High Salary, Low Risk': '#2ed573',
                'Low Salary, High Risk': '#ffa502',
                'Normal': '#747d8c'
            },
            opacity=0.6,
            hover_data=['job_title', 'industry', 'region']
        )
        fig_s1_scatter.add_hline(y=q75_auto, line_dash="dash", line_color="white", annotation_text=f"Auto Risk Q75={q75_auto}")
        fig_s1_scatter.add_vline(x=q75_sal, line_dash="dash", line_color="white", annotation_text=f"Salary Q75=${q75_sal:,.0f}")
        fig_s1_scatter.update_layout(height=500, template='plotly_dark')
        st.plotly_chart(fig_s1_scatter, use_container_width=True)
    
    with s1_right:
        # Bar chart: Paradox by industry
        paradox_by_ind = paradox_df.groupby('industry').size().reset_index(name='count').sort_values('count', ascending=True)
        fig_s1_bar = px.bar(
            paradox_by_ind, x='count', y='industry', orientation='h',
            title='Jumlah Data Paradoks per Industri',
            labels={'count': 'Jumlah', 'industry': 'Industri'},
            color='count',
            color_continuous_scale='RdYlGn_r',
            text='count'
        )
        fig_s1_bar.update_layout(height=500, template='plotly_dark', showlegend=False)
        fig_s1_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_s1_bar, use_container_width=True)
    
    # Paradox by job title
    paradox_by_jt = paradox_df.groupby('job_title').agg(
        count=('job_id', 'count'),
        avg_salary=('salary_usd', 'mean'),
        avg_auto_risk=('automation_risk_score', 'mean')
    ).sort_values('count', ascending=False).round(2)
    paradox_by_jt['avg_salary'] = paradox_by_jt['avg_salary'].map(lambda x: f"${x:,.0f}")
    
    st.markdown("### Distribusi Paradoks per Job Title")
    st.dataframe(paradox_by_jt, use_container_width=True)
    
    # Displacement risk in paradox
    s1b_left, s1b_right = st.columns(2)
    with s1b_left:
        disp_paradox = paradox_df['ai_job_displacement_risk'].value_counts().reset_index()
        disp_paradox.columns = ['risk', 'count']
        fig_s1_pie = px.pie(
            disp_paradox, values='count', names='risk',
            title='Displacement Risk dalam Data Paradoks',
            color='risk',
            color_discrete_map={'Low': '#2ed573', 'Medium': '#ffa502', 'High': '#ff4757'}
        )
        fig_s1_pie.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig_s1_pie, use_container_width=True)
    
    with s1b_right:
        paradox_by_region = paradox_df.groupby('region').size().reset_index(name='count').sort_values('count', ascending=True)
        fig_s1_region = px.bar(
            paradox_by_region, x='count', y='region', orientation='h',
            title='Data Paradoks per Region',
            labels={'count': 'Jumlah', 'region': 'Region'},
            color='count', color_continuous_scale='Reds', text='count'
        )
        fig_s1_region.update_layout(height=400, template='plotly_dark', showlegend=False)
        fig_s1_region.update_traces(textposition='outside')
        st.plotly_chart(fig_s1_region, use_container_width=True)
    
    with st.expander("Lihat Contoh Data Paradoks"):
        st.dataframe(
            paradox_df[['job_title', 'industry', 'region', 'seniority_level', 'salary_usd', 'automation_risk_score', 'ai_job_displacement_risk', 'reskilling_required']].head(20),
            use_container_width=True, hide_index=True
        )
    
    st.info("""
    **Insight:** Pekerjaan bergaji tinggi ($90K+) ternyata juga sangat rentan terhadap otomatisasi AI (risk score 0.84).  
    Agriculture & Government memimpin paradoks ini. Kompensasi tinggi bukan jaminan keamanan di era AI.
    """)

# ==================== STORY 2: SENIORITY SALARY INVERSION ====================
with story_tab2:
    st.markdown("""
    ### The Seniority Salary Inversion
    **Temuan:** Intern/Junior bisa bergaji **top 10%**, sementara Executive/Lead justru ada di **bottom 10%** gaji.  
    Level senioritas **tidak berkorelasi kuat** dengan kompensasi di sektor AI.
    """)
    
    q90_sal = df['salary_usd'].quantile(0.90)
    q10_sal = df['salary_usd'].quantile(0.10)
    
    junior_high = df[(df['seniority_level'].isin(['Intern', 'Junior'])) & (df['salary_usd'] >= q90_sal)]
    exec_low = df[(df['seniority_level'].isin(['Executive', 'Lead'])) & (df['salary_usd'] <= q10_sal)]
    
    s2_col1, s2_col2, s2_col3, s2_col4 = st.columns(4)
    with s2_col1:
        st.metric("Junior/Intern Gaji Top 10%", f"{len(junior_high):,}")
    with s2_col2:
        st.metric("Executive/Lead Gaji Bottom 10%", f"{len(exec_low):,}")
    with s2_col3:
        st.metric("Threshold Top 10%", f"${q90_sal:,.0f}")
    with s2_col4:
        st.metric("Threshold Bottom 10%", f"${q10_sal:,.0f}")
    
    s2_left, s2_right = st.columns(2)
    
    with s2_left:
        # Box plot: Salary by seniority
        fig_s2_box = px.box(
            df, x='seniority_level', y='salary_usd',
            title='Distribusi Gaji per Level Senioritas',
            labels={'salary_usd': 'Salary (USD)', 'seniority_level': 'Seniority Level'},
            color='seniority_level',
            category_orders={'seniority_level': ['Intern', 'Junior', 'Mid', 'Senior', 'Lead', 'Executive']}
        )
        fig_s2_box.update_layout(height=500, template='plotly_dark', showlegend=False)
        st.plotly_chart(fig_s2_box, use_container_width=True)
    
    with s2_right:
        # Create anomaly scatter
        anomaly_data = []
        for _, row in junior_high.iterrows():
            anomaly_data.append({**row.to_dict(), 'anomaly_type': 'Junior/Intern Gaji Tinggi'})
        for _, row in exec_low.iterrows():
            anomaly_data.append({**row.to_dict(), 'anomaly_type': 'Executive/Lead Gaji Rendah'})
        anomaly_df = pd.DataFrame(anomaly_data)
        
        if len(anomaly_df) > 0:
            fig_s2_scatter = px.scatter(
                anomaly_df, x='seniority_level', y='salary_usd',
                color='anomaly_type',
                title='Data Anomali: Inversi Senioritas-Gaji',
                labels={'salary_usd': 'Salary (USD)', 'seniority_level': 'Seniority Level'},
                color_discrete_map={
                    'Junior/Intern Gaji Tinggi': '#2ed573',
                    'Executive/Lead Gaji Rendah': '#ff4757'
                },
                hover_data=['job_title', 'industry', 'region'],
                size='salary_usd', size_max=15
            )
            fig_s2_scatter.update_layout(height=500, template='plotly_dark')
            st.plotly_chart(fig_s2_scatter, use_container_width=True)
    
    # Region breakdown
    st.markdown("### Faktor Region: Pemicu Utama Inversi")
    s2c_left, s2c_right = st.columns(2)
    
    with s2c_left:
        exec_low_region = exec_low.groupby('region').size().reset_index(name='count').sort_values('count', ascending=True)
        fig_s2_exec = px.bar(
            exec_low_region, x='count', y='region', orientation='h',
            title='Executive/Lead Bergaji Rendah per Region',
            color='count', color_continuous_scale='Reds', text='count'
        )
        fig_s2_exec.update_layout(height=400, template='plotly_dark', showlegend=False)
        fig_s2_exec.update_traces(textposition='outside')
        st.plotly_chart(fig_s2_exec, use_container_width=True)
    
    with s2c_right:
        # Mean salary by seniority
        sen_salary = df.groupby('seniority_level')['salary_usd'].agg(['mean', 'median']).round(0).astype(int)
        sen_salary = sen_salary.reindex(['Intern', 'Junior', 'Mid', 'Senior', 'Lead', 'Executive'])
        fig_s2_mean = go.Figure()
        fig_s2_mean.add_trace(go.Bar(x=sen_salary.index, y=sen_salary['mean'], name='Mean Salary', marker_color='#1e90ff'))
        fig_s2_mean.add_trace(go.Bar(x=sen_salary.index, y=sen_salary['median'], name='Median Salary', marker_color='#ffa502'))
        fig_s2_mean.update_layout(
            title='Rata-rata vs Median Gaji per Senioritas',
            barmode='group', height=400, template='plotly_dark',
            yaxis_title='Salary (USD)', xaxis_title='Seniority Level'
        )
        st.plotly_chart(fig_s2_mean, use_container_width=True)
    
    with st.expander("Contoh Data Junior/Intern Bergaji Tinggi"):
        st.dataframe(
            junior_high[['job_title', 'industry', 'region', 'seniority_level', 'salary_usd', 'ai_intensity_score']].sort_values('salary_usd', ascending=False).head(15),
            use_container_width=True, hide_index=True
        )
    
    with st.expander("Contoh Data Executive/Lead Bergaji Rendah"):
        st.dataframe(
            exec_low[['job_title', 'industry', 'region', 'seniority_level', 'salary_usd', 'ai_intensity_score']].sort_values('salary_usd').head(15),
            use_container_width=True, hide_index=True
        )
    
    st.info("""
    **Insight:** Level senioritas TIDAK berkorelasi kuat dengan gaji di sektor AI.  
    Executive/Lead bergaji rendah **99%** berada di Africa dan South Asia.  
    Faktor **region (lokasi geografis)** jauh lebih menentukan kompensasi daripada seniority.
    """)

# ==================== STORY 3: ADOPTION STAGE CONTRADICTION ====================
with story_tab3:
    st.markdown("""
    ### The Adoption Stage Contradiction
    **Temuan:** Industri berlabel **"Mature"** di AI masih punya banyak pekerjaan dengan AI intensity rendah (<0.2),  
    sementara industri **"Emerging"** justru punya pekerjaan dengan AI intensity sangat tinggi (>0.7).
    """)
    
    mature_low = df[(df['industry_ai_adoption_stage'] == 'Mature') & (df['ai_intensity_score'] < 0.2)]
    emerging_high = df[(df['industry_ai_adoption_stage'] == 'Emerging') & (df['ai_intensity_score'] > 0.7)]
    mature_all = df[df['industry_ai_adoption_stage'] == 'Mature']
    emerging_all = df[df['industry_ai_adoption_stage'] == 'Emerging']
    
    s3_col1, s3_col2, s3_col3, s3_col4 = st.columns(4)
    with s3_col1:
        pct_mature_low = len(mature_low) / len(mature_all) * 100 if len(mature_all) > 0 else 0
        st.metric("Mature tapi Low AI", f"{len(mature_low)}", f"{pct_mature_low:.1f}% dari Mature")
    with s3_col2:
        pct_emerging_high = len(emerging_high) / len(emerging_all) * 100 if len(emerging_all) > 0 else 0
        st.metric("Emerging tapi High AI", f"{len(emerging_high)}", f"{pct_emerging_high:.1f}% dari Emerging")
    with s3_col3:
        st.metric("Avg Intensity (Mature Low)", f"{mature_low['ai_intensity_score'].mean():.3f}")
    with s3_col4:
        st.metric("Avg Intensity (Emerging High)", f"{emerging_high['ai_intensity_score'].mean():.3f}")
    
    s3_left, s3_right = st.columns(2)
    
    with s3_left:
        # Violin plot: AI Intensity by Adoption Stage
        fig_s3_violin = px.violin(
            df, x='industry_ai_adoption_stage', y='ai_intensity_score',
            title='Distribusi AI Intensity per Adoption Stage',
            labels={'ai_intensity_score': 'AI Intensity Score', 'industry_ai_adoption_stage': 'Adoption Stage'},
            color='industry_ai_adoption_stage',
            box=True, points='outliers',
            category_orders={'industry_ai_adoption_stage': ['Emerging', 'Growing', 'Mature']},
            color_discrete_map={'Emerging': '#ffa502', 'Growing': '#1e90ff', 'Mature': '#2ed573'}
        )
        fig_s3_violin.update_layout(height=500, template='plotly_dark', showlegend=False)
        st.plotly_chart(fig_s3_violin, use_container_width=True)
    
    with s3_right:
        # Heatmap: Industry x Adoption Stage count
        ct_industry = pd.crosstab(df['industry'], df['industry_ai_adoption_stage'])
        ct_industry = ct_industry.reindex(columns=['Emerging', 'Growing', 'Mature'], fill_value=0)
        
        fig_s3_heat = px.imshow(
            ct_industry.values,
            x=['Emerging', 'Growing', 'Mature'],
            y=ct_industry.index.tolist(),
            title='Heatmap: Industri × Adoption Stage',
            labels={'x': 'Adoption Stage', 'y': 'Industry', 'color': 'Count'},
            color_continuous_scale='YlOrRd',
            text_auto=True,
            aspect='auto'
        )
        fig_s3_heat.update_layout(height=500, template='plotly_dark')
        st.plotly_chart(fig_s3_heat, use_container_width=True)
    
    # Contradiction detail bars
    st.markdown("### Detail Kontradiksi per Industri")
    s3d_left, s3d_right = st.columns(2)
    
    with s3d_left:
        ml_industry = mature_low.groupby('industry').size().reset_index(name='count').sort_values('count', ascending=True)
        fig_s3_ml = px.bar(
            ml_industry, x='count', y='industry', orientation='h',
            title='Mature tapi Low AI Intensity — per Industri',
            color='count', color_continuous_scale='Oranges', text='count'
        )
        fig_s3_ml.update_layout(height=400, template='plotly_dark', showlegend=False)
        fig_s3_ml.update_traces(textposition='outside')
        st.plotly_chart(fig_s3_ml, use_container_width=True)
    
    with s3d_right:
        eh_industry = emerging_high.groupby('industry').size().reset_index(name='count').sort_values('count', ascending=True)
        fig_s3_eh = px.bar(
            eh_industry, x='count', y='industry', orientation='h',
            title='Emerging tapi High AI Intensity — per Industri',
            color='count', color_continuous_scale='Blues', text='count'
        )
        fig_s3_eh.update_layout(height=400, template='plotly_dark', showlegend=False)
        fig_s3_eh.update_traces(textposition='outside')
        st.plotly_chart(fig_s3_eh, use_container_width=True)
    
    with st.expander("Contoh Data: Mature tapi Low AI"):
        st.dataframe(
            mature_low[['job_title', 'industry', 'ai_intensity_score', 'industry_ai_adoption_stage', 'automation_risk_score', 'ai_job_displacement_risk']].head(15),
            use_container_width=True, hide_index=True
        )
    
    with st.expander("Contoh Data: Emerging tapi High AI"):
        st.dataframe(
            emerging_high[['job_title', 'industry', 'ai_intensity_score', 'industry_ai_adoption_stage', 'automation_risk_score', 'ai_job_displacement_risk']].head(15),
            use_container_width=True, hide_index=True
        )
    
    st.info("""
    **Insight:** Hanya **Tech** dan **Finance** yang memiliki status "Mature". Namun 20.5% dari pekerjaan Mature  
    punya AI intensity <0.2. Sebaliknya, industri Emerging punya individu dengan AI intensity 0.83.  
    Label adopsi industri **tidak selalu mencerminkan realitas** di level pekerjaan individual.
    """)

# ==================== STORY 4: REGIONAL DIGITAL DIVIDE ====================
with story_tab4:
    st.markdown("""
    ### The Regional Digital Divide
    **Temuan:** Gap gaji **4x lipat** antara North America dan Africa, meskipun AI intensity,  
    automation risk, dan kebutuhan reskilling **hampir identik** di semua region.
    """)
    
    region_stats = df.groupby('region').agg(
        count=('job_id', 'count'),
        avg_salary=('salary_usd', 'mean'),
        median_salary=('salary_usd', 'median'),
        avg_ai_intensity=('ai_intensity_score', 'mean'),
        avg_auto_risk=('automation_risk_score', 'mean'),
        pct_reskill=('reskilling_required', 'mean')
    ).sort_values('avg_salary', ascending=False).round(2)
    
    top_region = region_stats.index[0]
    bot_region = region_stats.index[-1]
    ratio = region_stats.loc[top_region, 'avg_salary'] / region_stats.loc[bot_region, 'avg_salary']
    
    s4_col1, s4_col2, s4_col3, s4_col4 = st.columns(4)
    with s4_col1:
        st.metric(f"Avg Salary {top_region}", f"${region_stats.loc[top_region, 'avg_salary']:,.0f}")
    with s4_col2:
        st.metric(f"Avg Salary {bot_region}", f"${region_stats.loc[bot_region, 'avg_salary']:,.0f}")
    with s4_col3:
        st.metric("Salary Ratio", f"{ratio:.1f}x")
    with s4_col4:
        st.metric("Jumlah Region", f"{df['region'].nunique()}")
    
    s4_left, s4_right = st.columns(2)
    
    with s4_left:
        # Bar chart: Salary by region
        region_bar = region_stats.reset_index()
        fig_s4_bar = px.bar(
            region_bar, x='region', y='avg_salary',
            title='Rata-rata Gaji per Region',
            labels={'avg_salary': 'Average Salary (USD)', 'region': 'Region'},
            color='avg_salary',
            color_continuous_scale='RdYlGn',
            text=region_bar['avg_salary'].map(lambda x: f'${x:,.0f}')
        )
        fig_s4_bar.update_layout(height=500, template='plotly_dark', showlegend=False)
        fig_s4_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_s4_bar, use_container_width=True)
    
    with s4_right:
        # Radar chart: Region comparison
        # Normalize all metrics to 0-1 for radar
        radar_df = region_stats[['avg_salary', 'avg_ai_intensity', 'avg_auto_risk', 'pct_reskill']].copy()
        for col in radar_df.columns:
            radar_df[col] = (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min())
        
        categories = ['Salary', 'AI Intensity', 'Auto Risk', 'Reskilling']
        
        fig_s4_radar = go.Figure()
        colors_radar = ['#ff4757', '#1e90ff', '#2ed573', '#ffa502', '#a55eea', '#ff6348', '#747d8c', '#2bcbba', '#fd9644']
        
        for i, region in enumerate(radar_df.index[:5]):  # Top 5 regions for clarity
            values = radar_df.loc[region].tolist()
            values.append(values[0])  # Close the polygon
            fig_s4_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=region,
                line=dict(color=colors_radar[i]),
                opacity=0.6
            ))
        
        fig_s4_radar.update_layout(
            title='Radar: Perbandingan Metrik Top 5 Region (Normalized)',
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500, template='plotly_dark'
        )
        st.plotly_chart(fig_s4_radar, use_container_width=True)
    
    # Same job title, different salary
    st.markdown("### Gaji Job Title yang Sama di Region Berbeda")
    selected_jt_s4 = st.selectbox(
        "Pilih Job Title",
        df['job_title'].unique().tolist(),
        key='story4_jt'
    )
    
    jt_region = df[df['job_title'] == selected_jt_s4].groupby('region')['salary_usd'].mean().reset_index()
    jt_region = jt_region.sort_values('salary_usd', ascending=True)
    
    fig_s4_jt = px.bar(
        jt_region, x='salary_usd', y='region', orientation='h',
        title=f'Rata-rata Gaji "{selected_jt_s4}" per Region',
        labels={'salary_usd': 'Average Salary (USD)', 'region': 'Region'},
        color='salary_usd', color_continuous_scale='Viridis',
        text=jt_region['salary_usd'].map(lambda x: f'${x:,.0f}')
    )
    fig_s4_jt.update_layout(height=400, template='plotly_dark', showlegend=False)
    fig_s4_jt.update_traces(textposition='outside')
    st.plotly_chart(fig_s4_jt, use_container_width=True)
    
    # Full region stats table
    with st.expander("Tabel Lengkap Statistik per Region"):
        display_region = region_stats.copy()
        display_region['avg_salary'] = display_region['avg_salary'].map(lambda x: f"${x:,.0f}")
        display_region['median_salary'] = display_region['median_salary'].map(lambda x: f"${x:,.0f}")
        display_region['pct_reskill'] = display_region['pct_reskill'].map(lambda x: f"{x*100:.1f}%")
        st.dataframe(display_region, use_container_width=True)
    
    st.info("""
    **Insight:** AI intensity (~0.29), automation risk (~0.58), dan reskilling rate (~32%) **hampir identik** di semua region.  
    Namun gaji berbeda **4x lipat**. Seorang Data Scientist di Africa menghadapi risiko AI yang SAMA  
    dengan rekannya di North America, tapi dibayar **4x lebih rendah**. Ini adalah kesenjangan digital global yang nyata.
    """)

# ==================== STORY 5: RESKILLING BLIND SPOT ====================
with story_tab5:
    st.markdown("""
    ### The Reskilling Blind Spot
    **Temuan:** **21.1% dari seluruh dataset** (1,056 pekerjaan) memiliki risiko otomatisasi sangat tinggi (>0.8)  
    tetapi **TIDAK mensyaratkan program reskilling sama sekali**. Ini adalah blind spot sistemik yang mengkhawatirkan.
    """)
    
    blind_spot = df[(df['reskilling_required'] == False) & (df['automation_risk_score'] > 0.8)]
    prepared = df[(df['reskilling_required'] == True) & (df['automation_risk_score'] > 0.8)]
    low_risk_reskill = df[(df['reskilling_required'] == True) & (df['automation_risk_score'] <= 0.3)]
    
    s5_col1, s5_col2, s5_col3, s5_col4 = st.columns(4)
    with s5_col1:
        st.metric("Blind Spot (High Risk, No Reskill)", f"{len(blind_spot):,}", f"{len(blind_spot)/len(df)*100:.1f}% dari total")
    with s5_col2:
        st.metric("High Risk WITH Reskill", f"{len(prepared):,}", "0%", delta_color="inverse")
    with s5_col3:
        st.metric("Total High Auto Risk (>0.8)", f"{len(blind_spot) + len(prepared):,}")
    with s5_col4:
        st.metric("Avg Salary (Blind Spot)", f"${blind_spot['salary_usd'].mean():,.0f}")
    
    s5_left, s5_right = st.columns(2)
    
    with s5_left:
        # Pie chart: Blind spot proportion
        pie_data = pd.DataFrame({
            'Category': ['Blind Spot\n(High Risk, No Reskill)', 'Others'],
            'Count': [len(blind_spot), len(df) - len(blind_spot)]
        })
        fig_s5_pie = px.pie(
            pie_data, values='Count', names='Category',
            title='Proporsi Blind Spot dalam Dataset',
            color='Category',
            color_discrete_map={
                'Blind Spot\n(High Risk, No Reskill)': '#ff4757',
                'Others': '#2ed573'
            }
        )
        fig_s5_pie.update_layout(height=450, template='plotly_dark')
        st.plotly_chart(fig_s5_pie, use_container_width=True)
    
    with s5_right:
        # Bar: Blind spot by industry
        bs_industry = blind_spot.groupby('industry').size().reset_index(name='count').sort_values('count', ascending=True)
        fig_s5_bar = px.bar(
            bs_industry, x='count', y='industry', orientation='h',
            title='Blind Spot per Industri',
            labels={'count': 'Jumlah', 'industry': 'Industri'},
            color='count', color_continuous_scale='Reds', text='count'
        )
        fig_s5_bar.update_layout(height=450, template='plotly_dark', showlegend=False)
        fig_s5_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_s5_bar, use_container_width=True)
    
    # Temporal trend
    st.markdown("### Tren Temporal: Blind Spot per Tahun")
    if 'posting_year' in df.columns:
        bs_yearly = blind_spot.groupby('posting_year').size().reset_index(name='blind_spot_count')
        total_yearly = df.groupby('posting_year').size().reset_index(name='total_count')
        yearly_merged = bs_yearly.merge(total_yearly, on='posting_year')
        yearly_merged['pct'] = (yearly_merged['blind_spot_count'] / yearly_merged['total_count'] * 100).round(1)
        yearly_merged = yearly_merged.sort_values('posting_year')
        
        fig_s5_line = go.Figure()
        fig_s5_line.add_trace(go.Scatter(
            x=yearly_merged['posting_year'], y=yearly_merged['blind_spot_count'],
            mode='lines+markers', name='Jumlah Blind Spot',
            line=dict(color='#ff4757', width=3),
            marker=dict(size=10)
        ))
        fig_s5_line.add_trace(go.Bar(
            x=yearly_merged['posting_year'], y=yearly_merged['pct'],
            name='% dari Total per Tahun',
            marker_color='rgba(255, 71, 87, 0.3)',
            yaxis='y2'
        ))
        fig_s5_line.update_layout(
            title='Tren Blind Spot per Tahun (2010-2025)',
            xaxis_title='Tahun',
            yaxis_title='Jumlah Blind Spot',
            yaxis2=dict(title='Persentase (%)', overlaying='y', side='right', range=[0, 50]),
            height=450, template='plotly_dark',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_s5_line, use_container_width=True)
    
    # Reskilling by industry comparison
    st.markdown("### Perbandingan Reskilling Rate per Industri")
    reskill_rate = df.groupby('industry')['reskilling_required'].mean().mul(100).round(1).reset_index()
    reskill_rate.columns = ['industry', 'reskilling_rate']
    reskill_rate = reskill_rate.sort_values('reskilling_rate', ascending=True)
    
    fig_s5_reskill = px.bar(
        reskill_rate, x='reskilling_rate', y='industry', orientation='h',
        title='Reskilling Rate per Industri (% pekerjaan yang mensyaratkan reskilling)',
        labels={'reskilling_rate': 'Reskilling Rate (%)', 'industry': 'Industri'},
        color='reskilling_rate', color_continuous_scale='RdYlGn',
        text=reskill_rate['reskilling_rate'].map(lambda x: f'{x:.1f}%')
    )
    fig_s5_reskill.update_layout(height=400, template='plotly_dark', showlegend=False)
    fig_s5_reskill.update_traces(textposition='outside')
    st.plotly_chart(fig_s5_reskill, use_container_width=True)
    
    with st.expander("Contoh Data Blind Spot"):
        st.dataframe(
            blind_spot[['job_title', 'industry', 'region', 'posting_year', 'salary_usd', 'automation_risk_score', 'ai_job_displacement_risk', 'reskilling_required']].sort_values('automation_risk_score', ascending=False).head(20),
            use_container_width=True, hide_index=True
        )
    
    st.info("""
    **Insight:** Dari 1,056 pekerjaan berisiko otomatisasi tinggi (>0.8), **0%** yang mensyaratkan reskilling.  
    Ini adalah blind spot sistemik. Tren positif: blind spot menurun dari **28%** (2010) ke **~15%** (2025),  
    menunjukkan kesadaran meningkat. Finance (46.3%) dan Tech (44.9%) paling siap, Healthcare (27%) paling rentan.
    """)

# Clean up temporary column
if 'paradox_group' in df.columns:
    df.drop(columns=['paradox_group'], inplace=True)

st.markdown("---")

# Footer
st.markdown("---")
st.caption("Dashboard ini menampilkan analisis dampak AI terhadap pekerjaan dan prediksi adopsi AI menggunakan 4 metode: Linear Regression, Random Forest, Gradient Boosting, dan Polynomial Regression.")