import streamlit as st
import numpy as np
import pickle
import joblib
import warnings
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Fix for numpy._core issue - MUST be at the top
if not hasattr(np, '_core'):
    np._core = np.core

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
        margin: 0.5rem 0 0 0;
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
    .dataset-notice {
        background: linear-gradient(135deg, #1e3a5f 0%, #163050 100%);
        border-left: 4px solid #4da6ff;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.2rem;
    }
    .dataset-notice h4 {
        color: #4da6ff;
        margin: 0 0 0.5rem 0;
    }
    .dataset-notice p {
        color: #c8ddf0;
        margin: 0.3rem 0;
        font-size: 0.92rem;
    }
    .insight-box {
        background-color: #1a2635;
        border-left: 4px solid #00cc96;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.8rem;
    }
    .chart-guide {
        background-color: #1e2535;
        border-left: 4px solid #ffa502;
        padding: 0.8rem 1rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.8rem;
        font-size: 0.88rem;
        color: #d0d8e8;
    }
    .chart-guide strong {
        color: #ffa502;
    }
    .conclusion-note {
        background-color: #1f1f2e;
        border: 1px dashed #666;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #aaa;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Impact on Jobs Dashboard</h1>
    <p>Analisis Dampak <em>Artificial Intelligence</em> terhadap Pasar Kerja &nbsp;|&nbsp; Prediksi Adopsi AI 2026–2030 &nbsp;|&nbsp; 4 Metode Prediksi</p>
</div>
""", unsafe_allow_html=True)

# ==================== DATASET NOTICE (Synthetic Data Disclaimer) ====================
st.markdown("""
<div class="dataset-notice">
    <h4>📋 Catatan Dataset: Data Sintetis (<em>Synthetic Dataset</em>)</h4>
    <p>
        Dataset yang digunakan dalam dashboard ini adalah <strong>data sintetis</strong> (<em>synthetic data</em>)
        yang dihasilkan melalui proses <em>augmentation</em> berbasis distribusi statistik nyata
        dari laporan ketenagakerjaan global (antara lain laporan WEF <em>Future of Jobs</em>, 
        OECD <em>Employment Outlook</em>, dan McKinsey Global Institute).
    </p>
    <p>
        <strong>Proses Pembuatan Data:</strong> Variabel utama (industri, region, seniority, gaji) di-<em>generate</em> 
        menggunakan distribusi probabilistik yang dikalibrasi dari rentang nilai empiris.
        Skor AI — yaitu <em>ai_intensity_score</em> dan <em>automation_risk_score</em> — dihitung 
        melalui formula tertimbang berbasis proporsi tugas manual vs kognitif per kategori pekerjaan.
        Variabel <em>ai_skills_required</em> dirancang agar 67,5% lowongan <strong>tidak</strong> mencantumkan 
        keahlian AI secara eksplisit, namun tetap memiliki <em>ai_intensity_score</em> &gt; 0 karena skor intensitas 
        mencerminkan seberapa besar AI digunakan dalam proses kerja — bukan sekadar apakah lowongan 
        secara eksplisit mensyaratkan keahlian AI.
    </p>
    <p>
        <strong>Implikasi Analisis:</strong> Karena data bersifat sintetis, temuan dalam dashboard ini 
        <strong>tidak mewakili fakta absolut</strong>, melainkan menggambarkan <em>pola</em> dan <em>peluang 
        mendapatkan insight</em> dari data yang representatif secara distribusi.
        Setiap kesimpulan yang disajikan <strong>berlaku berdasarkan data ini</strong> dan perlu divalidasi 
        lebih lanjut dengan data lapangan.
    </p>
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
    st.markdown("## 🗂 Navigasi")
    
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <a href="#data-overview" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: #e0e0e0;">
                📊 Data Overview
            </div>
        </a>
        <a href="#automation-risk" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: #e0e0e0;">
                ⚙️ Automation Risk Analysis
            </div>
        </a>
        <a href="#displacement-risk" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: #e0e0e0;">
                🚨 Displacement Risk by Industry
            </div>
        </a>
        <a href="#ai-intensity" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: #e0e0e0;">
                🔬 AI Intensity Analysis
            </div>
        </a>
        <a href="#adoption-stage" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: #e0e0e0;">
                📈 Adoption Stage
            </div>
        </a>
        <a href="#salary-analysis" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: #e0e0e0;">
                💰 Salary Analysis
            </div>
        </a>
        <a href="#reskilling" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: #e0e0e0;">
                🎓 Reskilling Requirement
            </div>
        </a>
        <a href="#prediction" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: #e0e0e0;">
                🔮 4 Methods Prediction
            </div>
        </a>
        <a href="#comparison" style="text-decoration: none;">
            <div style="background-color: #2d2d3d; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: #e0e0e0;">
                🏆 Global Comparison
            </div>
        </a>
        <a href="#unique-data" style="text-decoration: none;">
            <div style="background: linear-gradient(135deg, #ff6b6b, #ffa500); padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0; text-align: center; color: white; font-weight: bold;">
                💡 Unique Data Stories
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== FILTERS ====================
    st.markdown("## 🔍 Filter Data")
    st.caption("Filter akan diterapkan ke seluruh visualisasi di bawah.")
    
    # Industry filter
    industries = df['industry'].unique().tolist()
    selected_industries = st.multiselect(
        "🏭 Industri",
        industries,
        default=industries[:3],
        help="Pilih satu atau lebih industri untuk dianalisis."
    )
    
    # Job Title filter
    if 'job_title' in df.columns:
        job_titles = df['job_title'].unique().tolist()
        selected_job_titles = st.multiselect(
            "💼 Job Title",
            job_titles,
            default=[],
            help="Kosongkan untuk menampilkan semua job title."
        )
    
    # Company Size filter
    if 'company_size' in df.columns:
        company_sizes = df['company_size'].unique().tolist()
        selected_sizes = st.multiselect(
            "🏢 Company Size",
            company_sizes,
            default=[],
            help="S=Small, M=Medium, L=Large."
        )
    
    # Seniority Level filter
    if 'seniority_level' in df.columns:
        seniority_levels = df['seniority_level'].unique().tolist()
        selected_seniority = st.multiselect(
            "🎖 Seniority Level",
            seniority_levels,
            default=[],
            help="Pilih level senioritas pekerjaan."
        )
    
    # AI Displacement Risk filter
    displacement_risks = df['ai_job_displacement_risk'].unique().tolist()
    selected_risks = st.multiselect(
        "⚠️ Displacement Risk",
        displacement_risks,
        default=displacement_risks,
        help="Low / Medium / High — tingkat risiko pekerjaan tergantikan AI."
    )
    
    # Adoption Stage filter
    adoption_stages = df['industry_ai_adoption_stage'].unique().tolist()
    selected_stages = st.multiselect(
        "🚀 Adoption Stage",
        adoption_stages,
        default=adoption_stages,
        help="Emerging = awal adopsi, Growing = berkembang, Mature = sudah mapan."
    )
    
    st.markdown("---")
    st.markdown("### 🔬 Metode Prediksi")
    st.info("""
    **4 Metode yang digunakan:**
    1. **Linear Regression** — Tren linier sederhana antar waktu
    2. **Random Forest** — Gabungan banyak *decision tree* (ensemble)
    3. **Gradient Boosting** — Pohon keputusan sekuensial yang saling memperbaiki
    4. **Polynomial Regression** — Kurva non-linier derajat 2
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
st.markdown("## 📊 Data Overview")
st.markdown(f"Menampilkan **{len(filtered_df):,}** dari **{len(df):,}** total *records* berdasarkan filter aktif.")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Records", f"{len(df):,}", help="Jumlah total baris data dalam dataset sintetis.")
with col2:
    st.metric("Industri", df['industry'].nunique(), help="Jumlah kategori industri dalam dataset.")
with col3:
    st.metric("Job Title Unik", df['job_title'].nunique() if 'job_title' in df.columns else "N/A",
              help="Jumlah jenis pekerjaan yang berbeda dalam dataset.")
with col4:
    st.metric("Rata-rata Gaji (USD)", f"${df['salary_usd'].mean():,.0f}",
              help="Rata-rata gaji seluruh dataset dalam USD.")
with col5:
    pct_reskill_overall = df['reskilling_required'].mean() * 100
    st.metric("Reskilling Rate", f"{pct_reskill_overall:.1f}%",
              help="Persentase pekerjaan yang mensyaratkan reskilling.")

# KPI baris kedua
col6, col7, col8, col9 = st.columns(4)
with col6:
    avg_auto = df['automation_risk_score'].mean()
    st.metric("Rata-rata Automation Risk", f"{avg_auto:.3f}",
              help="Skor risiko otomatisasi rata-rata (0=rendah, 1=tinggi).")
with col7:
    avg_ai_int = df['ai_intensity_score'].mean()
    st.metric("Rata-rata AI Intensity", f"{avg_ai_int:.3f}",
              help="Skor intensitas penggunaan AI rata-rata per pekerjaan (0–1).")
with col8:
    pct_high_risk = (df['ai_job_displacement_risk'] == 'High').mean() * 100
    st.metric("% Risiko Tinggi (High)", f"{pct_high_risk:.1f}%",
              help="Persentase pekerjaan dengan displacement risk = High.")
with col9:
    pct_no_ai_skills = (df['ai_skills_required'] == False).mean() * 100 if 'ai_skills_required' in df.columns else None
    if pct_no_ai_skills is not None:
        st.metric("Lowongan Tanpa AI Skills", f"{pct_no_ai_skills:.1f}%",
                  help="67,5% lowongan tidak mencantumkan AI skills secara eksplisit, namun tetap memiliki AI intensity > 0.")

# Show filtered data preview
with st.expander("🔍 Lihat Data Terfilter (100 baris pertama)"):
    st.dataframe(filtered_df.head(100), use_container_width=True)

st.markdown('<div class="conclusion-note">⚠️ Kesimpulan di bagian ini berdasarkan data sintetis yang telah dikalibrasi.</div>', unsafe_allow_html=True)
st.markdown("---")

# ==================== SECTION 2: AUTOMATION RISK ====================
st.markdown('<div id="automation-risk" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## ⚙️ Automation Risk Analysis")
st.markdown("""
*Automation risk score* adalah skor (0–1) yang mengukur seberapa besar kemungkinan tugas-tugas 
dalam suatu pekerjaan dapat diotomatiskan oleh AI/mesin. Skor mendekati 1 berarti pekerjaan tersebut 
sangat rentan tergantikan oleh otomatisasi.
""")

st.markdown("""
<div class="chart-guide">
<strong>📖 Cara Membaca Box Plot:</strong> Setiap kotak menunjukkan distribusi skor. 
Garis tengah kotak = nilai median. Tepi bawah kotak = kuartil 1 (Q1, nilai 25%). 
Tepi atas kotak = kuartil 3 (Q3, nilai 75%). Garis panjang (whisker) menunjukkan rentang nilai normal. 
Titik-titik di luar whisker = <em>outlier</em> (nilai ekstrem).
</div>
""", unsafe_allow_html=True)

col_auto1, col_auto2 = st.columns(2)

with col_auto1:
    fig_auto_box = px.box(
        filtered_df, x='industry', y='automation_risk_score',
        title='Distribusi Automation Risk Score per Industri',
        color='industry',
        labels={'automation_risk_score': 'Risk Score (0–1)', 'industry': 'Industri'}
    )
    fig_auto_box.update_layout(
        height=500, template='plotly_dark',
        xaxis_tickangle=-30,
        annotations=[dict(
            text="Semakin tinggi skor → semakin rentan terhadap otomatisasi",
            xref="paper", yref="paper", x=0.5, y=-0.22,
            showarrow=False, font=dict(size=11, color="#aaa"), xanchor="center"
        )]
    )
    st.plotly_chart(fig_auto_box, use_container_width=True)

with col_auto2:
    st.markdown("**📋 Statistik Ringkasan per Industri**")
    st.caption("mean = rata-rata | std = standar deviasi | min/max = nilai terendah/tertinggi")
    auto_stats = filtered_df.groupby('industry')['automation_risk_score'].agg(['mean', 'std', 'min', 'max']).round(3)
    auto_stats.columns = ['Rata-rata', 'Std. Dev', 'Min', 'Maks']
    auto_stats = auto_stats.sort_values('Rata-rata', ascending=False)
    st.dataframe(auto_stats, use_container_width=True)
    
    highest_risk = auto_stats.index[0] if len(auto_stats) > 0 else "N/A"
    lowest_risk = auto_stats.index[-1] if len(auto_stats) > 0 else "N/A"
    st.info(f"**Industri risiko otomatisasi tertinggi:** {highest_risk}  \n**Industri risiko otomatisasi terendah:** {lowest_risk}")
    st.markdown('<div class="conclusion-note">⚠️ Peringkat ini berdasarkan data sintetis.</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== SECTION 3: DISPLACEMENT RISK ====================
st.markdown('<div id="displacement-risk" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## 🚨 AI Job Displacement Risk by Industry")
st.markdown("""
*AI Job Displacement Risk* adalah kategori risiko (Low / Medium / High) yang menggambarkan 
seberapa besar kemungkinan pekerjaan tersebut tergantikan oleh AI dalam jangka menengah. 
Berbeda dengan *automation risk score* (angka kontinu 0–1), variabel ini bersifat kategoris.
""")

displacement_by_industry = filtered_df.groupby(['industry', 'ai_job_displacement_risk']).size().reset_index(name='count')

# Hitung persentase per industri
total_per_industry = filtered_df.groupby('industry').size().reset_index(name='total')
displacement_pct = displacement_by_industry.merge(total_per_industry, on='industry')
displacement_pct['percentage'] = (displacement_pct['count'] / displacement_pct['total'] * 100).round(1)

d_col1, d_col2 = st.columns(2)
with d_col1:
    fig_displacement = px.bar(
        displacement_by_industry,
        x='industry',
        y='count',
        color='ai_job_displacement_risk',
        title='Distribusi Displacement Risk per Industri (Jumlah)',
        labels={'count': 'Jumlah Pekerjaan', 'industry': 'Industri', 'ai_job_displacement_risk': 'Displacement Risk'},
        barmode='stack',
        color_discrete_map={'Low': '#2ed573', 'Medium': '#ffa502', 'High': '#ff4757'},
        category_orders={'ai_job_displacement_risk': ['Low', 'Medium', 'High']}
    )
    fig_displacement.update_layout(height=500, template='plotly_dark', xaxis_tickangle=-30)
    st.plotly_chart(fig_displacement, use_container_width=True)

with d_col2:
    fig_displacement_pct = px.bar(
        displacement_pct,
        x='industry',
        y='percentage',
        color='ai_job_displacement_risk',
        title='Distribusi Displacement Risk per Industri (%)',
        labels={'percentage': 'Persentase (%)', 'industry': 'Industri', 'ai_job_displacement_risk': 'Displacement Risk'},
        barmode='stack',
        color_discrete_map={'Low': '#2ed573', 'Medium': '#ffa502', 'High': '#ff4757'},
        category_orders={'ai_job_displacement_risk': ['Low', 'Medium', 'High']}
    )
    fig_displacement_pct.update_layout(height=500, template='plotly_dark', xaxis_tickangle=-30, yaxis_range=[0, 105])
    st.plotly_chart(fig_displacement_pct, use_container_width=True)

# Summary tabel
with st.expander("📋 Tabel Persentase Displacement Risk per Industri"):
    pivot_disp = displacement_pct.pivot_table(index='industry', columns='ai_job_displacement_risk', values='percentage', fill_value=0).round(1)
    st.dataframe(pivot_disp, use_container_width=True)

st.markdown('<div class="conclusion-note">⚠️ Distribusi risiko ini berdasarkan data sintetis dan mencerminkan pola distribusi, bukan fakta lapangan.</div>', unsafe_allow_html=True)
st.markdown("---")

# ==================== SECTION 4: AI INTENSITY ====================
st.markdown('<div id="ai-intensity" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## 🔬 AI Intensity Analysis")
st.markdown("""
*AI Intensity Score* (0–1) mengukur seberapa intensif AI digunakan dalam proses kerja suatu pekerjaan. 
Skor ini **tidak hanya didasarkan pada apakah lowongan mencantumkan AI skills**, melainkan pada proporsi 
tugas yang melibatkan AI secara langsung maupun tidak langsung. Inilah mengapa 67,5% lowongan tanpa 
AI skills eksplisit tetap bisa memiliki *intensity score* > 0.
""")

st.markdown("""
<div class="chart-guide">
<strong>📖 Cara Membaca Box Plot:</strong> Lihat penjelasan di bagian Automation Risk di atas. 
Untuk AI Intensity, skor tinggi (≥0.7) berarti sebagian besar aktivitas kerja harian sudah melibatkan AI.
</div>
""", unsafe_allow_html=True)

col_int1, col_int2 = st.columns(2)

with col_int1:
    fig_intensity_box = px.box(
        filtered_df, x='industry', y='ai_intensity_score',
        title='Distribusi AI Intensity Score per Industri',
        color='industry',
        labels={'ai_intensity_score': 'AI Intensity (0–1)', 'industry': 'Industri'}
    )
    fig_intensity_box.update_layout(height=500, template='plotly_dark', xaxis_tickangle=-30)
    st.plotly_chart(fig_intensity_box, use_container_width=True)

with col_int2:
    st.markdown("**📋 Statistik Ringkasan per Industri**")
    st.caption("mean = rata-rata | std = standar deviasi | min/max = nilai terendah/tertinggi")
    intensity_stats = filtered_df.groupby('industry')['ai_intensity_score'].agg(['mean', 'std', 'min', 'max']).round(3)
    intensity_stats.columns = ['Rata-rata', 'Std. Dev', 'Min', 'Maks']
    intensity_stats = intensity_stats.sort_values('Rata-rata', ascending=False)
    st.dataframe(intensity_stats, use_container_width=True)
    
    highest_intensity = intensity_stats.index[0] if len(intensity_stats) > 0 else "N/A"
    lowest_intensity = intensity_stats.index[-1] if len(intensity_stats) > 0 else "N/A"
    st.success(f"**Intensitas AI tertinggi:** {highest_intensity}  \n**Intensitas AI terendah:** {lowest_intensity}")
    st.markdown('<div class="conclusion-note">⚠️ Kesimpulan ini berdasarkan data sintetis.</div>', unsafe_allow_html=True)

# Scatter: AI Intensity vs Automation Risk
st.markdown("### Hubungan AI Intensity vs Automation Risk")
st.markdown("Apakah pekerjaan dengan AI tinggi juga berisiko otomatisasi tinggi?")
fig_scatter_int = px.scatter(
    filtered_df, x='ai_intensity_score', y='automation_risk_score',
    color='industry', opacity=0.5,
    title='Scatter Plot: AI Intensity vs Automation Risk per Industri',
    labels={'ai_intensity_score': 'AI Intensity Score', 'automation_risk_score': 'Automation Risk Score'},
    trendline='ols'
)
fig_scatter_int.update_layout(height=450, template='plotly_dark')
st.plotly_chart(fig_scatter_int, use_container_width=True)
st.caption("Garis tren (OLS) menunjukkan arah korelasi linear antara intensitas AI dan risiko otomatisasi per industri.")

st.markdown("---")

# ==================== SECTION 5: ADOPTION STAGE ====================
st.markdown('<div id="adoption-stage" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## 📈 AI Adoption Stage per Industri")
st.markdown("""
*AI Adoption Stage* menggambarkan sejauh mana suatu industri telah mengadopsi AI secara keseluruhan:
- **Emerging** — Adopsi AI masih awal, sebagian besar bersifat eksperimental  
- **Growing** — Adopsi AI sedang berkembang, mulai terintegrasi ke proses bisnis utama  
- **Mature** — Adopsi AI sudah mapan dan menjadi bagian inti dari operasional industri
""")

adoption_by_industry = filtered_df.groupby(['industry', 'industry_ai_adoption_stage']).size().reset_index(name='count')

fig_adoption = px.bar(
    adoption_by_industry,
    x='industry', y='count',
    color='industry_ai_adoption_stage',
    title='Distribusi AI Adoption Stage per Industri',
    labels={'count': 'Jumlah Pekerjaan', 'industry': 'Industri', 'industry_ai_adoption_stage': 'Adoption Stage'},
    barmode='stack',
    category_orders={'industry_ai_adoption_stage': ['Emerging', 'Growing', 'Mature']},
    color_discrete_map={'Emerging': '#ffa502', 'Growing': '#1e90ff', 'Mature': '#2ed573'}
)
fig_adoption.update_layout(height=500, template='plotly_dark', xaxis_tickangle=-30)
st.plotly_chart(fig_adoption, use_container_width=True)

# Pie dan bar berdampingan
adp_left, adp_right = st.columns(2)
with adp_left:
    adoption_overall = filtered_df['industry_ai_adoption_stage'].value_counts().reset_index()
    adoption_overall.columns = ['stage', 'count']
    adoption_overall['pct'] = (adoption_overall['count'] / adoption_overall['count'].sum() * 100).round(1)

    fig_adoption_pie = px.pie(
        adoption_overall, values='count', names='stage',
        title='Proporsi Keseluruhan AI Adoption Stage',
        color_discrete_sequence=['#ffa502', '#1e90ff', '#2ed573'],
        hole=0.35
    )
    fig_adoption_pie.update_traces(texttemplate='%{label}<br>%{percent}')
    fig_adoption_pie.update_layout(height=400, template='plotly_dark')
    st.plotly_chart(fig_adoption_pie, use_container_width=True)

with adp_right:
    st.markdown("**📋 Ringkasan Adoption Stage**")
    st.dataframe(adoption_overall.rename(columns={'stage': 'Stage', 'count': 'Jumlah', 'pct': 'Persentase (%)'}),
                 use_container_width=True, hide_index=True)
    st.info("**Catatan:** Status 'Mature' tidak menjamin semua pekerjaan di industri tersebut memiliki AI intensity tinggi (lihat Story 3: Adoption Stage Contradiction).")

st.markdown('<div class="conclusion-note">⚠️ Distribusi adoption stage ini berdasarkan data sintetis.</div>', unsafe_allow_html=True)
st.markdown("---")

# ==================== SECTION 6: SALARY ANALYSIS ====================
st.markdown('<div id="salary-analysis" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## 💰 Salary Analysis")
st.markdown("""
Analisis distribusi gaji dalam satuan USD, dipecah per industri, level risiko, dan region. 
Perlu diingat bahwa variabel GDP per kapita merupakan salah satu faktor sosio-ekonomi yang paling signifikan 
mempengaruhi gaji — sehingga perbandingan lintas region perlu mempertimbangkan konteks ekonomi masing-masing.
""")

st.markdown("""
<div class="chart-guide">
<strong>📖 Cara Membaca Box Plot Gaji:</strong> Kotak menunjukkan rentang gaji yang paling umum (Q1–Q3). 
Garis tengah = median gaji. Nilai di atas whisker atas = pekerjaan bergaji sangat tinggi (<em>outlier</em>). 
Semakin lebar kotak → semakin besar variasi gaji dalam industri tersebut.
</div>
""", unsafe_allow_html=True)

col_sal1, col_sal2 = st.columns(2)

with col_sal1:
    fig_salary_box = px.box(
        filtered_df, x='industry', y='salary_usd',
        title='Distribusi Gaji per Industri (USD)',
        color='industry',
        labels={'salary_usd': 'Gaji (USD)', 'industry': 'Industri'}
    )
    fig_salary_box.update_layout(height=500, template='plotly_dark', xaxis_tickangle=-30)
    st.plotly_chart(fig_salary_box, use_container_width=True)

with col_sal2:
    st.markdown("**📋 Statistik Gaji per Industri**")
    salary_stats = filtered_df.groupby('industry')['salary_usd'].agg(['mean', 'median', 'min', 'max']).round(0).astype(int)
    salary_stats.columns = ['Rata-rata', 'Median', 'Min', 'Maks']
    salary_stats = salary_stats.sort_values('Rata-rata', ascending=False)
    salary_stats_display = salary_stats.copy()
    for col in salary_stats_display.columns:
        salary_stats_display[col] = salary_stats_display[col].map(lambda x: f"${x:,}")
    st.dataframe(salary_stats_display, use_container_width=True)
    
    highest_salary = salary_stats.index[0] if len(salary_stats) > 0 else "N/A"
    lowest_salary = salary_stats.index[-1] if len(salary_stats) > 0 else "N/A"
    st.success(f"**Industri gaji tertinggi:** {highest_salary}  \n**Industri gaji terendah:** {lowest_salary}")
    st.markdown('<div class="conclusion-note">⚠️ Berdasarkan data sintetis.</div>', unsafe_allow_html=True)

# Salary by displacement risk
st.markdown("### Gaji vs Displacement Risk")
st.markdown("Apakah pekerjaan berisiko tinggi cenderung bergaji lebih rendah atau lebih tinggi?")

fig_salary_risk = px.box(
    filtered_df, x='ai_job_displacement_risk', y='salary_usd',
    title='Distribusi Gaji berdasarkan Tingkat Displacement Risk',
    color='ai_job_displacement_risk',
    labels={'salary_usd': 'Gaji (USD)', 'ai_job_displacement_risk': 'Displacement Risk'},
    color_discrete_map={'Low': '#2ed573', 'Medium': '#ffa502', 'High': '#ff4757'},
    category_orders={'ai_job_displacement_risk': ['Low', 'Medium', 'High']}
)
fig_salary_risk.update_layout(height=450, template='plotly_dark')
st.plotly_chart(fig_salary_risk, use_container_width=True)

# Salary by region (if available)
if 'region' in filtered_df.columns:
    st.markdown("### Gaji per Region")
    st.markdown("Faktor **GDP per kapita** merupakan pendorong utama perbedaan gaji antar region.")
    region_salary_box = filtered_df.groupby('region')['salary_usd'].mean().reset_index().sort_values('salary_usd', ascending=True)
    fig_region_sal = px.bar(
        region_salary_box, x='salary_usd', y='region', orientation='h',
        title='Rata-rata Gaji per Region (USD)',
        labels={'salary_usd': 'Rata-rata Gaji (USD)', 'region': 'Region'},
        color='salary_usd', color_continuous_scale='RdYlGn',
        text=region_salary_box['salary_usd'].map(lambda x: f'${x:,.0f}')
    )
    fig_region_sal.update_layout(height=450, template='plotly_dark', showlegend=False)
    fig_region_sal.update_traces(textposition='outside')
    st.plotly_chart(fig_region_sal, use_container_width=True)

st.markdown("---")

# ==================== SECTION 7: RESKILLING ====================
st.markdown('<div id="reskilling" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## 🎓 Reskilling Requirement")
st.markdown("""
*Reskilling* mengacu pada proses pelatihan ulang tenaga kerja untuk menguasai keterampilan baru 
yang relevan di era AI. Variabel `reskilling_required` bernilai True/False: apakah lowongan 
mensyaratkan kandidat sudah atau siap menjalani program *reskilling*.
""")

col_res1, col_res2 = st.columns(2)

with col_res1:
    reskilling_by_industry = filtered_df.groupby(['industry', 'reskilling_required']).size().reset_index(name='count')
    reskilling_by_industry['label'] = reskilling_by_industry['reskilling_required'].map({True: 'Reskilling Diperlukan', False: 'Tidak Diperlukan'})
    
    fig_reskilling = px.bar(
        reskilling_by_industry,
        x='industry', y='count',
        color='label',
        title='Kebutuhan Reskilling per Industri',
        labels={'count': 'Jumlah Pekerjaan', 'industry': 'Industri', 'label': 'Status Reskilling'},
        barmode='stack',
        color_discrete_map={'Reskilling Diperlukan': '#ff6b6b', 'Tidak Diperlukan': '#00cc96'}
    )
    fig_reskilling.update_layout(height=500, template='plotly_dark', xaxis_tickangle=-30)
    st.plotly_chart(fig_reskilling, use_container_width=True)

with col_res2:
    reskilling_pct = pd.crosstab(
        filtered_df['industry'], 
        filtered_df['reskilling_required'], 
        normalize='index'
    ).mul(100).round(1)
    reskilling_pct.columns = ['Tidak Diperlukan (%)', 'Diperlukan (%)']
    reskilling_pct = reskilling_pct.sort_values('Diperlukan (%)', ascending=False)
    st.markdown("**📋 Reskilling Rate per Industri (%)**")
    st.dataframe(reskilling_pct, use_container_width=True)
    
    total_reskill = filtered_df['reskilling_required'].sum()
    total_records = len(filtered_df)
    reskill_pct = (total_reskill / total_records) * 100 if total_records > 0 else 0
    
    st.metric("Overall Reskilling Rate (data terfilter)", f"{reskill_pct:.1f}%",
              help="Persentase pekerjaan dalam filter aktif yang mensyaratkan reskilling.")

st.markdown('<div class="conclusion-note">⚠️ Berdasarkan data sintetis.</div>', unsafe_allow_html=True)
st.markdown("---")

# ==================== SECTION 8: PREDICTION 4 METHODS ====================
st.markdown('<div id="prediction" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## 🔮 Perbandingan 4 Metode Prediksi")
st.markdown("""
Empat metode *machine learning* dan statistik digunakan untuk memprediksi **AI Intensity Score** 
per industri pada periode 2026–2030. Setiap metode memiliki kekuatan dan kelemahan berbeda:

| Metode | Keunggulan | Keterbatasan |
|---|---|---|
| **Linear Regression** | Sederhana, mudah diinterpretasi | Asumsi tren linier — kurang fleksibel |
| **Random Forest** | Robust terhadap *noise*, akurat | Sulit diinterpretasi (*black box*) |
| **Gradient Boosting** | Akurasi tinggi, menangani *non-linearity* | Lebih lambat, sensitif *hyperparameter* |
| **Polynomial Regression** | Menangkap kurva non-linier | Rentan *overfitting* pada derajat tinggi |

**Cara membaca kartu metode:** R² mendekati 1 = model sangat baik menjelaskan variasi data. 
MAE dan RMSE = rata-rata kesalahan prediksi (semakin kecil semakin baik).
""")

# Select industry for prediction
pred_industries = list(predictions['linear'].keys())
selected_pred_industry = st.selectbox(
    "🏭 Pilih Industri untuk Prediksi",
    pred_industries,
    key="pred_industry",
    help="Pilih industri untuk melihat prediksi AI Intensity Score 2026–2030."
)

if selected_pred_industry:
    methods = ['linear', 'random_forest', 'gradient_boosting', 'polynomial']
    method_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Polynomial']
    method_desc = {
        'Linear Regression': 'Tren linier sederhana',
        'Random Forest': 'Ensemble 100+ decision tree',
        'Gradient Boosting': 'Boosting sekuensial adaptif',
        'Polynomial': 'Kurva non-linier derajat 2'
    }
    
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
    
    # Display cards — menggunakan komponen Streamlit native agar tidak rusak
    cols = st.columns(4)
    for i, (method, name) in enumerate(zip(methods, method_names)):
        pred_data = predictions[method][selected_pred_industry]
        is_best = (i == best_method_idx)
        
        with cols[i]:
            # Border hijau untuk best method via CSS container
            if is_best:
                st.markdown(
                    f'<div style="border:2px solid #00cc96; border-radius:10px; padding:0.8rem; '
                    f'background:#262730; margin-bottom:0.5rem;">'
                    f'<span style="background:#00cc96;color:#1a1a2e;padding:2px 8px;border-radius:4px;'
                    f'font-size:0.75rem;font-weight:bold;">✅ BEST METHOD</span>'
                    f'<h4 style="color:#fff;margin:0.5rem 0 0.2rem 0;">{name}</h4>'
                    f'<p style="color:#aaa;font-size:0.8rem;margin:0 0 0.8rem 0;">{method_desc[name]}</p>'
                    f'<p style="color:#ccc;margin:0.2rem 0;font-size:0.85rem;"><strong>R² Score</strong></p>'
                    f'<p style="font-size:1.3rem;font-weight:bold;color:#00cc96;margin:0 0 0.5rem 0;">'
                    f'{pred_data["r2_score"]:.4f}</p>'
                    f'<p style="color:#ccc;font-size:0.85rem;margin:0.2rem 0;">'
                    f'<strong>MAE</strong> <span style="color:#aaa;font-size:0.75rem;">(rata-rata error)</span><br>'
                    f'<span style="font-size:1rem;">{pred_data["mae"]:.4f}</span></p>'
                    f'<p style="color:#ccc;font-size:0.85rem;margin:0.2rem 0;">'
                    f'<strong>RMSE</strong> <span style="color:#aaa;font-size:0.75rem;">(root mean sq error)</span><br>'
                    f'<span style="font-size:1rem;">{pred_data["rmse"]:.4f}</span></p>'
                    f'<hr style="border-color:#4a4a5a;margin:0.8rem 0;">'
                    f'<p style="color:#ccc;font-size:0.85rem;margin:0.2rem 0;"><strong>Prediksi 2030</strong></p>'
                    f'<p style="font-size:1.4rem;font-weight:bold;color:#ff6b6b;margin:0;">'
                    f'{pred_data["predictions"][-1]:.4f}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="border:1px solid #4a4a5a; border-radius:10px; padding:0.8rem; '
                    f'background:#262730; margin-bottom:0.5rem;">'
                    f'<h4 style="color:#fff;margin:0 0 0.2rem 0;">{name}</h4>'
                    f'<p style="color:#aaa;font-size:0.8rem;margin:0 0 0.8rem 0;">{method_desc[name]}</p>'
                    f'<p style="color:#ccc;margin:0.2rem 0;font-size:0.85rem;"><strong>R² Score</strong></p>'
                    f'<p style="font-size:1.3rem;font-weight:bold;color:#00cc96;margin:0 0 0.5rem 0;">'
                    f'{pred_data["r2_score"]:.4f}</p>'
                    f'<p style="color:#ccc;font-size:0.85rem;margin:0.2rem 0;">'
                    f'<strong>MAE</strong> <span style="color:#aaa;font-size:0.75rem;">(rata-rata error)</span><br>'
                    f'<span style="font-size:1rem;">{pred_data["mae"]:.4f}</span></p>'
                    f'<p style="color:#ccc;font-size:0.85rem;margin:0.2rem 0;">'
                    f'<strong>RMSE</strong> <span style="color:#aaa;font-size:0.75rem;">(root mean sq error)</span><br>'
                    f'<span style="font-size:1rem;">{pred_data["rmse"]:.4f}</span></p>'
                    f'<hr style="border-color:#4a4a5a;margin:0.8rem 0;">'
                    f'<p style="color:#ccc;font-size:0.85rem;margin:0.2rem 0;"><strong>Prediksi 2030</strong></p>'
                    f'<p style="font-size:1.4rem;font-weight:bold;color:#ff6b6b;margin:0;">'
                    f'{pred_data["predictions"][-1]:.4f}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    # Historical and prediction chart
    industry_data = df[df["industry"] == selected_pred_industry].copy()
    # Gunakan posting_year jika tersedia, fallback ke year, lalu generate acak
    if "posting_year" in industry_data.columns:
        industry_data["_year"] = industry_data["posting_year"]
    elif "year" in industry_data.columns:
        industry_data["_year"] = industry_data["year"]
    else:
        np.random.seed(42)
        industry_data["_year"] = np.random.randint(2010, 2026, len(industry_data))
    yearly_avg = industry_data.groupby("_year")["ai_intensity_score"].mean().reset_index()
    yearly_avg = yearly_avg.rename(columns={"_year": "year"})
    yearly_avg = yearly_avg.sort_values("year")
    
    pred_years = [2026, 2027, 2028, 2029, 2030]
    all_predictions_df = pd.DataFrame({'year': pred_years})
    
    for method, name in zip(methods, method_names):
        pred_data = predictions[method][selected_pred_industry]
        all_predictions_df[name] = pred_data['predictions']
    
    fig_pred_line = go.Figure()

    # ── Hitung statistik per tahun untuk menampilkan distribusi historis ──
    hist_stats = industry_data.groupby("_year")["ai_intensity_score"].agg(
        mean='mean', std='std', min='min', max='max',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index().rename(columns={"_year": "year"}).sort_values("year")
    hist_stats['std'] = hist_stats['std'].fillna(0)

    # Shaded band: min–max (sangat transparan)
    fig_pred_line.add_trace(go.Scatter(
        x=pd.concat([hist_stats['year'], hist_stats['year'][::-1]]),
        y=pd.concat([hist_stats['max'], hist_stats['min'][::-1]]),
        fill='toself',
        fillcolor='rgba(200,200,255,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip',
        name='Rentang Min–Maks'
    ))

    # Shaded band: Q25–Q75 (lebih solid = IQR / rentang tengah 50%)
    fig_pred_line.add_trace(go.Scatter(
        x=pd.concat([hist_stats['year'], hist_stats['year'][::-1]]),
        y=pd.concat([hist_stats['q75'], hist_stats['q25'][::-1]]),
        fill='toself',
        fillcolor='rgba(120,180,255,0.18)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=True,
        name='IQR Historis (Q25–Q75)',
        hoverinfo='skip',
    ))

    # Scatter titik individual per tahun (jitter kecil agar tidak tumpuk)
    np.random.seed(7)
    jitter = np.random.uniform(-0.15, 0.15, len(industry_data))
    fig_pred_line.add_trace(go.Scatter(
        x=industry_data["_year"] + jitter,
        y=industry_data["ai_intensity_score"],
        mode='markers',
        name='Data Aktual (tiap pekerjaan)',
        marker=dict(
            size=4,
            color='rgba(160,200,255,0.35)',
            line=dict(width=0)
        ),
        hovertemplate='Tahun: %{x:.0f}<br>Score: %{y:.4f}<extra></extra>'
    ))

    # Garis mean historis — tebal, cerah, mudah dilihat
    fig_pred_line.add_trace(go.Scatter(
        x=hist_stats['year'],
        y=hist_stats['mean'],
        mode='lines+markers',
        name='Rata-rata Historis (Aktual)',
        line=dict(width=3.5, color='#7ecfff'),
        marker=dict(size=9, color='#7ecfff', symbol='circle',
                    line=dict(width=2, color='#ffffff')),
        hovertemplate='Tahun: %{x}<br>Rata-rata: %{y:.4f}<extra></extra>'
    ))

    # ── Garis prediksi 4 metode ──
    pred_colors = ['#ffd166', '#ef476f', '#06d6a0', '#a855f7']
    pred_symbols = ['square', 'diamond', 'triangle-up', 'cross']
    for i, name in enumerate(method_names):
        fig_pred_line.add_trace(go.Scatter(
            x=all_predictions_df['year'],
            y=all_predictions_df[name],
            mode='lines+markers',
            name=f'{name}',
            line=dict(width=2.5, color=pred_colors[i], dash='dot'),
            marker=dict(size=10, color=pred_colors[i], symbol=pred_symbols[i],
                        line=dict(width=1.5, color='#ffffff')),
            hovertemplate=f'{name}<br>Tahun: %{{x}}<br>Prediksi: %{{y:.4f}}<extra></extra>'
        ))

    # Garis pemisah historis vs prediksi
    fig_pred_line.add_vline(
        x=2025.5, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1.5,
        annotation_text="◀ Historis   Prediksi ▶",
        annotation_position="top",
        annotation_font=dict(color="rgba(255,255,255,0.7)", size=12)
    )

    # Shaded area prediksi (background ringan)
    fig_pred_line.add_vrect(
        x0=2025.5, x1=2030.5,
        fillcolor="rgba(255,200,100,0.04)",
        layer="below", line_width=0,
    )

    y_all = pd.concat([
        hist_stats['min'], hist_stats['max'],
        all_predictions_df[method_names].min(axis=1),
        all_predictions_df[method_names].max(axis=1)
    ])
    y_margin = (y_all.max() - y_all.min()) * 0.15
    y_min = max(0, y_all.min() - y_margin)
    y_max = min(1, y_all.max() + y_margin)

    fig_pred_line.update_layout(
        title=dict(
            text=f'<b>AI Intensity Score</b> — Historis & Prediksi 2026–2030 | Industri: <b>{selected_pred_industry}</b>',
            font=dict(size=15)
        ),
        xaxis=dict(
            title='Tahun',
            tickmode='linear', dtick=1,
            tickangle=-45,
            gridcolor='rgba(255,255,255,0.07)'
        ),
        yaxis=dict(
            title='AI Intensity Score (0–1)',
            range=[y_min, y_max],
            gridcolor='rgba(255,255,255,0.07)'
        ),
        height=530,
        template='plotly_dark',
        legend=dict(
            orientation='h',
            yanchor='bottom', y=-0.42,
            xanchor='center', x=0.5,
            font=dict(size=11),
            bgcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='rgba(22,22,40,1)',
        paper_bgcolor='rgba(22,22,40,0)',
        hovermode='x unified',
        margin=dict(b=130)
    )
    st.plotly_chart(fig_pred_line, use_container_width=True)

    # Keterangan legend
    st.markdown("""
    <div style="font-size:0.83rem; color:#aaa; line-height:1.7; padding:0.5rem 0;">
    🔵 <b style="color:#7ecfff">Garis biru tebal</b> = rata-rata <em>AI Intensity Score</em> per tahun (data historis aktual) &nbsp;|&nbsp;
    🟦 <b style="color:#78b4ff">Area biru muda</b> = rentang IQR (Q25–Q75, tengah 50% data tiap tahun) &nbsp;|&nbsp;
    ⚪ Titik-titik transparan = skor tiap pekerjaan individual &nbsp;|&nbsp;
    <b>Garis warna putus-putus</b> = proyeksi 4 metode prediksi (2026–2030)
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction table
    st.markdown("### 📋 Tabel Prediksi (2026–2030)")
    display_table = all_predictions_df.copy()
    display_table['year'] = display_table['year'].astype(int)
    for col in method_names:
        display_table[col] = display_table[col].map(lambda x: f"{x:.4f}")
    st.dataframe(display_table.rename(columns={'year': 'Tahun'}), use_container_width=True, hide_index=True)
    st.markdown('<div class="conclusion-note">⚠️ Prediksi ini dilatih pada data sintetis — gunakan sebagai gambaran tren, bukan fakta absolut.</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== SECTION 9: GLOBAL COMPARISON ====================
st.markdown('<div id="comparison" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("## 🏆 Perbandingan Global: Ranking Metode Prediksi")
st.markdown("""
Metrik **R² Score** (*coefficient of determination*) menunjukkan seberapa baik model menjelaskan variasi 
dalam data. Nilai R² = 1.0 berarti model sempurna; nilai mendekati 0 berarti model tidak lebih baik dari 
sekadar menggunakan nilai rata-rata. MAE (*Mean Absolute Error*) dan RMSE (*Root Mean Squared Error*) 
mengukur rata-rata kesalahan prediksi — semakin kecil semakin baik.
""")

if ranking_df is not None:
    r2_col, table_col = st.columns([3, 2])
    with r2_col:
        fig_rank = px.bar(
            ranking_df,
            x='Method', y='Avg R2 Score',
            title='Rata-rata R² Score — Semua Industri',
            labels={'Avg R2 Score': 'Avg R² Score', 'Method': 'Metode'},
            color='Avg R2 Score',
            color_continuous_scale='RdYlGn',
            text='Avg R2 Score',
            template='plotly_dark'
        )
        fig_rank.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_rank.update_layout(height=420, yaxis_range=[0, 1.1])
        st.plotly_chart(fig_rank, use_container_width=True)

    with table_col:
        st.markdown("**📋 Tabel Ranking Metode**")
        st.caption("Semakin tinggi R² dan semakin kecil MAE/RMSE → metode semakin baik.")
        st.dataframe(
            ranking_df.style.format({
                'Avg R2 Score': '{:.4f}',
                'Avg MAE': '{:.4f}',
                'Avg RMSE': '{:.4f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        st.markdown('<div class="conclusion-note">⚠️ Ranking ini berlaku untuk dataset sintetis ini.</div>', unsafe_allow_html=True)
else:
    st.info("Data ranking tidak tersedia. Jalankan model training terlebih dahulu.")

# ==================== SECTION 10: UNIQUE DATA STORIES ====================
st.markdown('<div id="unique-data" class="section-title"></div>', unsafe_allow_html=True)
st.markdown("""
## 💡 5 Unique Data Stories
Analisis mendalam terhadap dataset mengungkap 5 anomali dan kontradiksi unik yang menantang asumsi umum 
terkait dampak AI pada pasar kerja. **Semua temuan berikut berlaku berdasarkan dataset sintetis ini.**
""")

story_tab1, story_tab2, story_tab3, story_tab4, story_tab5 = st.tabs([
    "1: Salary-Automation Paradox",
    "2: Seniority Salary Inversion",
    "3: Adoption Stage Contradiction",
    "4: Regional Digital Divide",
    "5: Reskilling Blind Spot"
])

# ==================== STORY 1: SALARY-AUTOMATION PARADOX ====================
with story_tab1:
    st.markdown("""
    ### 💸 The Salary-Automation Paradox
    **Temuan:** Pekerjaan bergaji tinggi (kuartil atas ≥Q75) justru memiliki risiko otomatisasi yang 
    **juga sangat tinggi** (≥Q75). Asumsi umum bahwa "gaji tinggi = aman dari AI" ternyata **tidak selalu benar** 
    berdasarkan data ini.
    """)
    
    q75_sal = df['salary_usd'].quantile(0.75)
    q75_auto = df['automation_risk_score'].quantile(0.75)
    
    df['paradox_group'] = 'Normal'
    df.loc[(df['salary_usd'] >= q75_sal) & (df['automation_risk_score'] >= q75_auto), 'paradox_group'] = 'Paradox (High Salary + High Risk)'
    df.loc[(df['salary_usd'] >= q75_sal) & (df['automation_risk_score'] < q75_auto), 'paradox_group'] = 'High Salary, Low Risk'
    df.loc[(df['salary_usd'] < q75_sal) & (df['automation_risk_score'] >= q75_auto), 'paradox_group'] = 'Low Salary, High Risk'
    
    paradox_df = df[df['paradox_group'] == 'Paradox (High Salary + High Risk)']
    
    s1_col1, s1_col2, s1_col3, s1_col4 = st.columns(4)
    with s1_col1:
        st.metric("Data Paradoks", f"{len(paradox_df):,}", f"{len(paradox_df)/len(df)*100:.1f}% dari total")
    with s1_col2:
        st.metric("Avg Salary Paradoks", f"${paradox_df['salary_usd'].mean():,.0f}")
    with s1_col3:
        st.metric("Avg Auto Risk", f"{paradox_df['automation_risk_score'].mean():.3f}")
    with s1_col4:
        st.metric("Threshold Salary Q75", f"${q75_sal:,.0f}")

    st.markdown("""
    <div class="chart-guide">
    <strong>📖 Cara Membaca Scatter Plot:</strong> Setiap titik mewakili satu pekerjaan. 
    Sumbu X = gaji (USD), sumbu Y = skor risiko otomatisasi (0–1). 
    Garis putus-putus membagi plot menjadi 4 kuadran. 
    Kuadran kanan-atas (merah) = Paradox — gaji tinggi <em>dan</em> risiko otomatisasi tinggi.
    </div>
    """, unsafe_allow_html=True)
    
    s1_left, s1_right = st.columns(2)
    
    with s1_left:
        fig_s1_scatter = px.scatter(
            df, x='salary_usd', y='automation_risk_score',
            color='paradox_group',
            title='Salary vs Automation Risk — Paradox Highlighted',
            labels={'salary_usd': 'Gaji (USD)', 'automation_risk_score': 'Automation Risk Score'},
            color_discrete_map={
                'Paradox (High Salary + High Risk)': '#ff4757',
                'High Salary, Low Risk': '#2ed573',
                'Low Salary, High Risk': '#ffa502',
                'Normal': '#747d8c'
            },
            opacity=0.6,
            hover_data=['job_title', 'industry', 'region']
        )
        fig_s1_scatter.add_hline(y=q75_auto, line_dash="dash", line_color="white",
                                  annotation_text=f"Auto Risk Q75={q75_auto:.2f}")
        fig_s1_scatter.add_vline(x=q75_sal, line_dash="dash", line_color="white",
                                  annotation_text=f"Salary Q75=${q75_sal:,.0f}")
        fig_s1_scatter.update_layout(height=500, template='plotly_dark',
                                      legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig_s1_scatter, use_container_width=True)
    
    with s1_right:
        paradox_by_ind = paradox_df.groupby('industry').size().reset_index(name='count').sort_values('count', ascending=True)
        fig_s1_bar = px.bar(
            paradox_by_ind, x='count', y='industry', orientation='h',
            title='Jumlah Data Paradoks per Industri',
            labels={'count': 'Jumlah', 'industry': 'Industri'},
            color='count', color_continuous_scale='RdYlGn_r', text='count'
        )
        fig_s1_bar.update_layout(height=500, template='plotly_dark', showlegend=False)
        fig_s1_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_s1_bar, use_container_width=True)
    
    paradox_by_jt = paradox_df.groupby('job_title').agg(
        count=('job_id', 'count'),
        avg_salary=('salary_usd', 'mean'),
        avg_auto_risk=('automation_risk_score', 'mean')
    ).sort_values('count', ascending=False).round(2)
    paradox_by_jt['avg_salary'] = paradox_by_jt['avg_salary'].map(lambda x: f"${x:,.0f}")
    
    st.markdown("### Distribusi Paradoks per Job Title")
    st.dataframe(paradox_by_jt.rename(columns={'count':'Jumlah','avg_salary':'Rata-rata Gaji','avg_auto_risk':'Avg Auto Risk'}),
                 use_container_width=True)
    
    s1b_left, s1b_right = st.columns(2)
    with s1b_left:
        disp_paradox = paradox_df['ai_job_displacement_risk'].value_counts().reset_index()
        disp_paradox.columns = ['risk', 'count']
        fig_s1_pie = px.pie(
            disp_paradox, values='count', names='risk',
            title='Displacement Risk dalam Data Paradoks',
            color='risk',
            color_discrete_map={'Low': '#2ed573', 'Medium': '#ffa502', 'High': '#ff4757'},
            hole=0.35
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
            paradox_df[['job_title','industry','region','seniority_level','salary_usd','automation_risk_score','ai_job_displacement_risk','reskilling_required']].head(20),
            use_container_width=True, hide_index=True
        )
    
    st.info("""
    **Insight (berdasarkan data sintetis):** Pekerjaan bergaji tinggi ($90K+) ternyata juga sangat rentan terhadap 
    otomatisasi AI (*risk score* 0.84). Agriculture & Government memimpin paradoks ini. 
    Kompensasi tinggi bukan jaminan keamanan di era AI.
    """)

# ==================== STORY 2: SENIORITY SALARY INVERSION ====================
with story_tab2:
    st.markdown("""
    ### 🔄 The Seniority Salary Inversion
    **Temuan:** *Intern*/Junior bisa bergaji **top 10%**, sementara Executive/Lead justru ada di **bottom 10%** gaji.  
    Level senioritas **tidak berkorelasi kuat** dengan kompensasi di sektor AI berdasarkan data ini.
    """)
    
    q90_sal = df['salary_usd'].quantile(0.90)
    q10_sal = df['salary_usd'].quantile(0.10)
    
    junior_high = df[(df['seniority_level'].isin(['Intern', 'Junior'])) & (df['salary_usd'] >= q90_sal)]
    exec_low = df[(df['seniority_level'].isin(['Executive', 'Lead'])) & (df['salary_usd'] <= q10_sal)]
    
    s2_col1, s2_col2, s2_col3, s2_col4 = st.columns(4)
    with s2_col1:
        st.metric("Junior/Intern Gaji Top 10%", f"{len(junior_high):,}",
                  help="Jumlah Intern/Junior dengan gaji ≥ Q90")
    with s2_col2:
        st.metric("Executive/Lead Gaji Bottom 10%", f"{len(exec_low):,}",
                  help="Jumlah Executive/Lead dengan gaji ≤ Q10")
    with s2_col3:
        st.metric("Threshold Top 10% (Q90)", f"${q90_sal:,.0f}")
    with s2_col4:
        st.metric("Threshold Bottom 10% (Q10)", f"${q10_sal:,.0f}")
    
    st.markdown("""
    <div class="chart-guide">
    <strong>📖 Cara Membaca Violin Plot:</strong> Bentuk "biola" menunjukkan distribusi gaji. 
    Bagian lebih lebar = lebih banyak pekerjaan pada rentang gaji tersebut. 
    Kotak kecil di tengah = statistik <em>box plot</em> (Q1, median, Q3).
    </div>
    """, unsafe_allow_html=True)
    
    s2_left, s2_right = st.columns(2)
    
    with s2_left:
        fig_s2_box = px.box(
            df, x='seniority_level', y='salary_usd',
            title='Distribusi Gaji per Level Senioritas',
            labels={'salary_usd': 'Gaji (USD)', 'seniority_level': 'Seniority Level'},
            color='seniority_level',
            category_orders={'seniority_level': ['Intern', 'Junior', 'Mid', 'Senior', 'Lead', 'Executive']}
        )
        fig_s2_box.update_layout(height=500, template='plotly_dark', showlegend=False)
        st.plotly_chart(fig_s2_box, use_container_width=True)
    
    with s2_right:
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
                labels={'salary_usd': 'Gaji (USD)', 'seniority_level': 'Seniority Level'},
                color_discrete_map={
                    'Junior/Intern Gaji Tinggi': '#2ed573',
                    'Executive/Lead Gaji Rendah': '#ff4757'
                },
                hover_data=['job_title', 'industry', 'region'],
                size='salary_usd', size_max=15
            )
            fig_s2_scatter.update_layout(height=500, template='plotly_dark',
                                          legend=dict(orientation='h', y=-0.2))
            st.plotly_chart(fig_s2_scatter, use_container_width=True)
    
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
        sen_salary = df.groupby('seniority_level')['salary_usd'].agg(['mean', 'median']).round(0).astype(int)
        sen_salary = sen_salary.reindex(['Intern', 'Junior', 'Mid', 'Senior', 'Lead', 'Executive'])
        fig_s2_mean = go.Figure()
        fig_s2_mean.add_trace(go.Bar(x=sen_salary.index, y=sen_salary['mean'], name='Mean Salary', marker_color='#1e90ff'))
        fig_s2_mean.add_trace(go.Bar(x=sen_salary.index, y=sen_salary['median'], name='Median Salary', marker_color='#ffa502'))
        fig_s2_mean.update_layout(
            title='Rata-rata vs Median Gaji per Senioritas',
            barmode='group', height=400, template='plotly_dark',
            yaxis_title='Gaji (USD)', xaxis_title='Seniority Level'
        )
        st.plotly_chart(fig_s2_mean, use_container_width=True)
    
    with st.expander("Contoh Data Junior/Intern Bergaji Tinggi"):
        st.dataframe(
            junior_high[['job_title','industry','region','seniority_level','salary_usd','ai_intensity_score']].sort_values('salary_usd', ascending=False).head(15),
            use_container_width=True, hide_index=True
        )
    with st.expander("Contoh Data Executive/Lead Bergaji Rendah"):
        st.dataframe(
            exec_low[['job_title','industry','region','seniority_level','salary_usd','ai_intensity_score']].sort_values('salary_usd').head(15),
            use_container_width=True, hide_index=True
        )
    
    st.info("""
    **Insight (berdasarkan data sintetis):** Level senioritas TIDAK berkorelasi kuat dengan gaji di sektor AI.  
    Executive/Lead bergaji rendah (~99%) berada di Africa dan South Asia — 
    faktor **region** (lokasi geografis dan GDP per kapita) jauh lebih menentukan kompensasi 
    daripada seniority.
    """)

# ==================== STORY 3: ADOPTION STAGE CONTRADICTION ====================
with story_tab3:
    st.markdown("""
    ### ⚡ The Adoption Stage Contradiction
    **Temuan:** Industri berlabel **"Mature"** di AI masih memiliki banyak pekerjaan dengan 
    *AI intensity* rendah (<0.2), sementara industri **"Emerging"** justru punya pekerjaan dengan 
    *AI intensity* sangat tinggi (>0.7).
    """)
    
    mature_low = df[(df['industry_ai_adoption_stage'] == 'Mature') & (df['ai_intensity_score'] < 0.2)]
    emerging_high = df[(df['industry_ai_adoption_stage'] == 'Emerging') & (df['ai_intensity_score'] > 0.7)]
    mature_all = df[df['industry_ai_adoption_stage'] == 'Mature']
    emerging_all = df[df['industry_ai_adoption_stage'] == 'Emerging']
    
    s3_col1, s3_col2, s3_col3, s3_col4 = st.columns(4)
    with s3_col1:
        pct_mature_low = len(mature_low) / len(mature_all) * 100 if len(mature_all) > 0 else 0
        st.metric("Mature tapi Low AI", f"{len(mature_low):,}", f"{pct_mature_low:.1f}% dari seluruh Mature")
    with s3_col2:
        pct_emerging_high = len(emerging_high) / len(emerging_all) * 100 if len(emerging_all) > 0 else 0
        st.metric("Emerging tapi High AI", f"{len(emerging_high):,}", f"{pct_emerging_high:.1f}% dari seluruh Emerging")
    with s3_col3:
        st.metric("Avg Intensity (Mature Low)", f"{mature_low['ai_intensity_score'].mean():.3f}")
    with s3_col4:
        st.metric("Avg Intensity (Emerging High)", f"{emerging_high['ai_intensity_score'].mean():.3f}")
    
    st.markdown("""
    <div class="chart-guide">
    <strong>📖 Cara Membaca Violin Plot:</strong> Bentuk simetris yang melebar di atas = banyak pekerjaan 
    dengan <em>AI intensity</em> tinggi. Titik-titik di tepi = <em>outlier</em>. 
    Perhatikan bagaimana "Emerging" dan "Mature" memiliki distribusi yang tumpang-tindih — 
    ini yang disebut "kontradiksi".
    </div>
    """, unsafe_allow_html=True)
    
    s3_left, s3_right = st.columns(2)
    
    with s3_left:
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
        st.markdown("""
        <div class="chart-guide">
        <strong>📖 Cara Membaca Heatmap:</strong> Warna lebih gelap/merah = jumlah pekerjaan lebih banyak. 
        Angka di dalam sel = jumlah pekerjaan (jobs) pada kombinasi industri × adoption stage tersebut.
        </div>
        """, unsafe_allow_html=True)
        ct_industry = pd.crosstab(df['industry'], df['industry_ai_adoption_stage'])
        ct_industry = ct_industry.reindex(columns=['Emerging', 'Growing', 'Mature'], fill_value=0)
        
        fig_s3_heat = px.imshow(
            ct_industry.values,
            x=['Emerging', 'Growing', 'Mature'],
            y=ct_industry.index.tolist(),
            title='Heatmap: Industri × Adoption Stage (Jumlah Pekerjaan)',
            labels={'x': 'Adoption Stage', 'y': 'Industri', 'color': 'Jumlah'},
            color_continuous_scale='YlOrRd',
            text_auto=True, aspect='auto'
        )
        fig_s3_heat.update_layout(height=500, template='plotly_dark')
        st.plotly_chart(fig_s3_heat, use_container_width=True)
    
    st.markdown("### Detail Kontradiksi per Industri")
    s3d_left, s3d_right = st.columns(2)
    
    with s3d_left:
        ml_industry = mature_low.groupby('industry').size().reset_index(name='count').sort_values('count', ascending=True)
        fig_s3_ml = px.bar(
            ml_industry, x='count', y='industry', orientation='h',
            title='Mature tapi AI Intensity Rendah — per Industri',
            color='count', color_continuous_scale='Oranges', text='count'
        )
        fig_s3_ml.update_layout(height=400, template='plotly_dark', showlegend=False)
        fig_s3_ml.update_traces(textposition='outside')
        st.plotly_chart(fig_s3_ml, use_container_width=True)
    
    with s3d_right:
        eh_industry = emerging_high.groupby('industry').size().reset_index(name='count').sort_values('count', ascending=True)
        fig_s3_eh = px.bar(
            eh_industry, x='count', y='industry', orientation='h',
            title='Emerging tapi AI Intensity Tinggi — per Industri',
            color='count', color_continuous_scale='Blues', text='count'
        )
        fig_s3_eh.update_layout(height=400, template='plotly_dark', showlegend=False)
        fig_s3_eh.update_traces(textposition='outside')
        st.plotly_chart(fig_s3_eh, use_container_width=True)
    
    with st.expander("Contoh Data: Mature tapi Low AI"):
        st.dataframe(
            mature_low[['job_title','industry','ai_intensity_score','industry_ai_adoption_stage','automation_risk_score','ai_job_displacement_risk']].head(15),
            use_container_width=True, hide_index=True
        )
    with st.expander("Contoh Data: Emerging tapi High AI"):
        st.dataframe(
            emerging_high[['job_title','industry','ai_intensity_score','industry_ai_adoption_stage','automation_risk_score','ai_job_displacement_risk']].head(15),
            use_container_width=True, hide_index=True
        )
    
    st.info("""
    **Insight (berdasarkan data sintetis):** Hanya Tech dan Finance yang memiliki status "Mature". 
    Namun 20,5% dari pekerjaan Mature punya *AI intensity* <0.2. 
    Sebaliknya, industri Emerging punya individu dengan *AI intensity* 0.83.  
    Label adopsi industri **tidak selalu mencerminkan realitas** di level pekerjaan individual — 
    ini mencerminkan heterogenitas dalam transformasi digital yang sebenarnya.
    """)

# ==================== STORY 4: REGIONAL DIGITAL DIVIDE ====================
with story_tab4:
    st.markdown("""
    ### 🌍 The Regional Digital Divide
    **Temuan:** Gap gaji **4x lipat** antara *North America* dan *Africa*, meskipun *AI intensity*, 
    *automation risk*, dan kebutuhan *reskilling* **hampir identik** di semua region. 
    Variabel sosio-ekonomi — terutama GDP per kapita — menjadi faktor paling signifikan.
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
        st.metric(f"Avg Salary {top_region}", f"${region_stats.loc[top_region, 'avg_salary']:,.0f}",
                  help="Region dengan gaji tertinggi")
    with s4_col2:
        st.metric(f"Avg Salary {bot_region}", f"${region_stats.loc[bot_region, 'avg_salary']:,.0f}",
                  help="Region dengan gaji terendah")
    with s4_col3:
        st.metric("Rasio Gaji (Tertinggi/Terendah)", f"{ratio:.1f}×",
                  help="Seberapa besar perbedaan gaji antar region ekstrem")
    with s4_col4:
        st.metric("Jumlah Region", f"{df['region'].nunique()}")
    
    s4_left, s4_right = st.columns(2)
    
    with s4_left:
        region_bar = region_stats.reset_index()
        fig_s4_bar = px.bar(
            region_bar, x='region', y='avg_salary',
            title='Rata-rata Gaji per Region (USD)',
            labels={'avg_salary': 'Rata-rata Gaji (USD)', 'region': 'Region'},
            color='avg_salary', color_continuous_scale='RdYlGn',
            text=region_bar['avg_salary'].map(lambda x: f'${x:,.0f}')
        )
        fig_s4_bar.update_layout(height=500, template='plotly_dark', showlegend=False, xaxis_tickangle=-30)
        fig_s4_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_s4_bar, use_container_width=True)
    
    with s4_right:
        st.markdown("""
        <div class="chart-guide">
        <strong>📖 Cara Membaca Radar Chart:</strong> Setiap sudut (vertex) mewakili satu metrik yang sudah 
        dinormalisasi (0–1). Semakin luas area yang dibentuk oleh garis suatu region → semakin tinggi nilainya 
        secara relatif di semua dimensi. Perhatikan bahwa luas area serupa antar region menunjukkan 
        bahwa AI intensity dan risiko otomatisasi hampir sama, tetapi gaji berbeda drastis.
        </div>
        """, unsafe_allow_html=True)
        
        radar_df = region_stats[['avg_salary', 'avg_ai_intensity', 'avg_auto_risk', 'pct_reskill']].copy()
        for col in radar_df.columns:
            col_range = radar_df[col].max() - radar_df[col].min()
            radar_df[col] = (radar_df[col] - radar_df[col].min()) / col_range if col_range > 0 else 0
        
        categories = ['Rata-rata Gaji', 'AI Intensity', 'Auto Risk', 'Reskilling Rate']
        
        fig_s4_radar = go.Figure()
        colors_radar = ['#ff4757', '#1e90ff', '#2ed573', '#ffa502', '#a55eea']
        
        for i, region in enumerate(radar_df.index[:5]):
            values = radar_df.loc[region].tolist()
            values.append(values[0])
            fig_s4_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself', name=region,
                line=dict(color=colors_radar[i]), opacity=0.6
            ))
        
        fig_s4_radar.update_layout(
            title='Radar: Perbandingan Metrik Top 5 Region (Dinormalisasi)',
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500, template='plotly_dark'
        )
        st.plotly_chart(fig_s4_radar, use_container_width=True)
    
    st.markdown("### Gaji Job Title yang Sama di Region Berbeda")
    selected_jt_s4 = st.selectbox(
        "Pilih Job Title",
        df['job_title'].unique().tolist(),
        key='story4_jt',
        help="Pilih jabatan untuk melihat seberapa besar perbedaan gajinya antar region."
    )
    
    jt_region = df[df['job_title'] == selected_jt_s4].groupby('region')['salary_usd'].mean().reset_index()
    jt_region = jt_region.sort_values('salary_usd', ascending=True)
    
    fig_s4_jt = px.bar(
        jt_region, x='salary_usd', y='region', orientation='h',
        title=f'Rata-rata Gaji "{selected_jt_s4}" per Region',
        labels={'salary_usd': 'Rata-rata Gaji (USD)', 'region': 'Region'},
        color='salary_usd', color_continuous_scale='Viridis',
        text=jt_region['salary_usd'].map(lambda x: f'${x:,.0f}')
    )
    fig_s4_jt.update_layout(height=400, template='plotly_dark', showlegend=False)
    fig_s4_jt.update_traces(textposition='outside')
    st.plotly_chart(fig_s4_jt, use_container_width=True)
    
    with st.expander("Tabel Lengkap Statistik per Region"):
        display_region = region_stats.copy()
        display_region['avg_salary'] = display_region['avg_salary'].map(lambda x: f"${x:,.0f}")
        display_region['median_salary'] = display_region['median_salary'].map(lambda x: f"${x:,.0f}")
        display_region['pct_reskill'] = display_region['pct_reskill'].map(lambda x: f"{x*100:.1f}%")
        display_region.columns = ['Jumlah', 'Avg Salary', 'Median Salary', 'Avg AI Intensity', 'Avg Auto Risk', 'Reskill Rate']
        st.dataframe(display_region, use_container_width=True)
    
    st.info("""
    **Insight (berdasarkan data sintetis):** *AI intensity* (~0.29), *automation risk* (~0.58), 
    dan *reskilling rate* (~32%) **hampir identik** di semua region.  
    Namun gaji berbeda **4× lipat**. Seorang *Data Scientist* di Afrika menghadapi risiko AI yang SAMA  
    dengan rekannya di North America, tetapi dibayar **4× lebih rendah**. 
    Ini mencerminkan kesenjangan digital global yang perlu diperhatikan.
    """)

# ==================== STORY 5: RESKILLING BLIND SPOT ====================
with story_tab5:
    st.markdown("""
    ### 🚫 The Reskilling Blind Spot
    **Temuan:** **21,1% dari seluruh dataset** (1.056 pekerjaan) memiliki risiko otomatisasi sangat tinggi (>0.8)  
    tetapi **tidak mensyaratkan program *reskilling* sama sekali**. Ini adalah *blind spot* sistemik yang 
    mengkhawatirkan dalam perencanaan ketenagakerjaan.
    """)
    
    blind_spot = df[(df['reskilling_required'] == False) & (df['automation_risk_score'] > 0.8)]
    prepared = df[(df['reskilling_required'] == True) & (df['automation_risk_score'] > 0.8)]
    
    s5_col1, s5_col2, s5_col3, s5_col4 = st.columns(4)
    with s5_col1:
        st.metric("Blind Spot (High Risk, No Reskill)", f"{len(blind_spot):,}",
                  f"{len(blind_spot)/len(df)*100:.1f}% dari total dataset")
    with s5_col2:
        st.metric("High Risk WITH Reskill", f"{len(prepared):,}",
                  help="Pekerjaan berisiko tinggi yang sudah mensyaratkan reskilling")
    with s5_col3:
        st.metric("Total High Auto Risk (>0.8)", f"{len(blind_spot) + len(prepared):,}")
    with s5_col4:
        st.metric("Avg Salary (Blind Spot)", f"${blind_spot['salary_usd'].mean():,.0f}")
    
    s5_left, s5_right = st.columns(2)
    
    with s5_left:
        pie_data = pd.DataFrame({
            'Kategori': ['Blind Spot\n(High Risk, No Reskill)', 'Lainnya'],
            'Jumlah': [len(blind_spot), len(df) - len(blind_spot)]
        })
        fig_s5_pie = px.pie(
            pie_data, values='Jumlah', names='Kategori',
            title='Proporsi Blind Spot dalam Dataset',
            color='Kategori',
            color_discrete_map={'Blind Spot\n(High Risk, No Reskill)': '#ff4757', 'Lainnya': '#2ed573'},
            hole=0.35
        )
        fig_s5_pie.update_layout(height=450, template='plotly_dark')
        st.plotly_chart(fig_s5_pie, use_container_width=True)
    
    with s5_right:
        bs_industry = blind_spot.groupby('industry').size().reset_index(name='count').sort_values('count', ascending=True)
        fig_s5_bar = px.bar(
            bs_industry, x='count', y='industry', orientation='h',
            title='Blind Spot per Industri (Pekerjaan Berisiko Tanpa Reskilling)',
            labels={'count': 'Jumlah', 'industry': 'Industri'},
            color='count', color_continuous_scale='Reds', text='count'
        )
        fig_s5_bar.update_layout(height=450, template='plotly_dark', showlegend=False)
        fig_s5_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_s5_bar, use_container_width=True)
    
    st.markdown("### Tren Temporal: Blind Spot per Tahun")
    if 'posting_year' in df.columns:
        bs_yearly = blind_spot.groupby('posting_year').size().reset_index(name='blind_spot_count')
        total_yearly = df.groupby('posting_year').size().reset_index(name='total_count')
        yearly_merged = bs_yearly.merge(total_yearly, on='posting_year')
        yearly_merged['pct'] = (yearly_merged['blind_spot_count'] / yearly_merged['total_count'] * 100).round(1)
        yearly_merged = yearly_merged.sort_values('posting_year')
        
        st.markdown("""
        <div class="chart-guide">
        <strong>📖 Cara Membaca Dual-Axis Chart:</strong> Garis merah (sumbu kiri) = jumlah absolut 
        <em>blind spot</em> per tahun. Batang semi-transparan (sumbu kanan) = persentase <em>blind spot</em> 
        dari total pekerjaan tahun tersebut. Tren menurun = semakin banyak organisasi menyadari pentingnya 
        <em>reskilling</em>.
        </div>
        """, unsafe_allow_html=True)
        
        fig_s5_line = go.Figure()
        fig_s5_line.add_trace(go.Scatter(
            x=yearly_merged['posting_year'], y=yearly_merged['blind_spot_count'],
            mode='lines+markers', name='Jumlah Blind Spot',
            line=dict(color='#ff4757', width=3), marker=dict(size=10)
        ))
        fig_s5_line.add_trace(go.Bar(
            x=yearly_merged['posting_year'], y=yearly_merged['pct'],
            name='% dari Total per Tahun',
            marker_color='rgba(255, 71, 87, 0.3)', yaxis='y2'
        ))
        fig_s5_line.update_layout(
            title='Tren Blind Spot per Tahun (2010–2025)',
            xaxis_title='Tahun',
            yaxis_title='Jumlah Blind Spot',
            yaxis2=dict(title='Persentase (%)', overlaying='y', side='right', range=[0, 50]),
            height=450, template='plotly_dark',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_s5_line, use_container_width=True)
    
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
            blind_spot[['job_title','industry','region','posting_year','salary_usd','automation_risk_score','ai_job_displacement_risk','reskilling_required']].sort_values('automation_risk_score', ascending=False).head(20),
            use_container_width=True, hide_index=True
        )
    
    st.info("""
    **Insight (berdasarkan data sintetis):** Dari 1.056 pekerjaan berisiko otomatisasi tinggi (>0.8), 
    mayoritas tidak mensyaratkan *reskilling*. Tren positif: *blind spot* menurun dari **~28%** (2010) 
    ke **~15%** (2025), menunjukkan kesadaran yang meningkat. Finance (46,3%) dan Tech (44,9%) 
    paling siap; Healthcare (27%) paling rentan berdasarkan data ini.
    """)

# Clean up temporary column
if 'paradox_group' in df.columns:
    df.drop(columns=['paradox_group'], inplace=True)

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align:center; color:#888; padding: 1rem 0;">
    <p>Dashboard <em>AI Impact on Jobs</em> — dibangun dengan Streamlit & Plotly</p>
    <p style="font-size:0.8rem;">Analisis menggunakan 4 metode: <em>Linear Regression</em>, <em>Random Forest</em>, 
    <em>Gradient Boosting</em>, dan <em>Polynomial Regression</em>.<br>
    <strong>Semua temuan dan kesimpulan berlaku berdasarkan dataset sintetis ini.</strong></p>
</div>
""", unsafe_allow_html=True)
