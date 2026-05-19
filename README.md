# AI Adoption Predictor 2030

## Deskripsi Proyek
Dashboard prediksi tingkat adopsi AI (AI Intensity Score) per industri untuk tahun 2026-2030 berdasarkan data historis 2010-2025 menggunakan pendekatan regresi linear.

## Fitur
- Prediksi per industri (9 sektor)
- Dashboard interaktif dengan Streamlit
- Visualisasi tren historis dan prediksi
- Komparasi antar industri
- Ekspor data ke CSV

## Teknologi
- Python 3.8+
- Pandas, NumPy (Data processing)
- Scikit-learn (Linear Regression)
- Plotly (Visualisasi interaktif)
- Streamlit (Dashboard framework)

## Struktur Proyek
📁 AI-Adoption-Prediction/
├── 📁 data/
│ └── ai_impact_jobs_2010_2025.csv
├── 📁 models/
│ ├── linear_models.pkl
│ └── predictions.pkl
├── 📁 pages/
│ ├── 1_Prediction.py
│ ├── 2_Comparison.py
│ └── 3_Data_Table.py
├── app.py
├── model_training.py
├── requirements.txt
└── README.md