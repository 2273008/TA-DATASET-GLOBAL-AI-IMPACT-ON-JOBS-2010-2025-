import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI PATH ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

print("="*70)
print("4 METHODS AI ADOPTION PREDICTION MODEL TRAINING")
print("="*70)
print(f"Base Directory: {BASE_DIR}")
print(f"Models Directory: {MODELS_DIR}")
print("="*70)

# Load data
file_path = os.path.join(DATA_DIR, 'ai_impact_jobs_2010_2025.csv')
df = pd.read_csv(file_path)

print(f"\nData loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Tambahkan kolom year jika tidak ada
if 'year' not in df.columns:
    print("\nKolom 'year' tidak ditemukan, membuat data tahun...")
    np.random.seed(42)
    years = list(range(2010, 2026))
    n_years = len(years)
    probabilities = [1/n_years] * n_years
    
    industries = df['industry'].unique()
    for industry in industries:
        industry_count = len(df[df['industry'] == industry])
        assigned_years = np.random.choice(years, size=industry_count, replace=True, p=probabilities)
        df.loc[df['industry'] == industry, 'year'] = assigned_years
    
    print("Kolom 'year' berhasil ditambahkan")

# Group data per tahun
yearly_avg = df.groupby(['year', 'industry'])['ai_intensity_score'].mean().reset_index()
industries = yearly_avg['industry'].unique()

# Dictionary untuk menyimpan semua model dan prediksi
all_models = {
    'linear': {},
    'random_forest': {},
    'gradient_boosting': {},
    'polynomial': {}
}

all_predictions = {
    'linear': {},
    'random_forest': {},
    'gradient_boosting': {},
    'polynomial': {}
}

all_results = []

print("\n" + "="*70)
print("TRAINING 4 METHODS PER INDUSTRY")
print("="*70)

for industry in industries:
    industry_data = yearly_avg[yearly_avg['industry'] == industry].copy()
    industry_data = industry_data.sort_values('year')
    
    if len(industry_data) < 3:
        print(f"\nWarning: {industry} - Data tidak cukup ({len(industry_data)} tahun), skip")
        continue
    
    X = industry_data[['year']].values
    y = industry_data['ai_intensity_score'].values
    future_years = np.array([[2026], [2027], [2028], [2029], [2030]])
    
    print(f"\n{'='*50}")
    print(f"Industry: {industry}")
    print(f"{'='*50}")
    
    # ========== METHOD 1: LINEAR REGRESSION ==========
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    lr_pred = lr_model.predict(future_years)
    lr_y_pred = lr_model.predict(X)
    
    lr_r2 = r2_score(y, lr_y_pred)
    lr_mae = mean_absolute_error(y, lr_y_pred)
    lr_rmse = np.sqrt(mean_squared_error(y, lr_y_pred))
    
    all_models['linear'][industry] = lr_model
    all_predictions['linear'][industry] = {
        'years': future_years.flatten(),
        'predictions': lr_pred,
        'r2_score': lr_r2,
        'mae': lr_mae,
        'rmse': lr_rmse,
        'slope': lr_model.coef_[0],
        'intercept': lr_model.intercept_
    }
    
    print(f"\n[Method 1] Linear Regression:")
    print(f"   R2 Score: {lr_r2:.4f}")
    print(f"   MAE: {lr_mae:.4f}")
    print(f"   RMSE: {lr_rmse:.4f}")
    print(f"   Slope: {lr_model.coef_[0]:.4f}")
    print(f"   Prediction 2030: {lr_pred[-1]:.4f}")
    
    # ========== METHOD 2: RANDOM FOREST ==========
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    rf_pred = rf_model.predict(future_years)
    rf_y_pred = rf_model.predict(X)
    
    rf_r2 = r2_score(y, rf_y_pred)
    rf_mae = mean_absolute_error(y, rf_y_pred)
    rf_rmse = np.sqrt(mean_squared_error(y, rf_y_pred))
    
    all_models['random_forest'][industry] = rf_model
    all_predictions['random_forest'][industry] = {
        'years': future_years.flatten(),
        'predictions': rf_pred,
        'r2_score': rf_r2,
        'mae': rf_mae,
        'rmse': rf_rmse
    }
    
    print(f"\n[Method 2] Random Forest:")
    print(f"   R2 Score: {rf_r2:.4f}")
    print(f"   MAE: {rf_mae:.4f}")
    print(f"   RMSE: {rf_rmse:.4f}")
    print(f"   Prediction 2030: {rf_pred[-1]:.4f}")
    
    # ========== METHOD 3: GRADIENT BOOSTING ==========
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X, y)
    gb_pred = gb_model.predict(future_years)
    gb_y_pred = gb_model.predict(X)
    
    gb_r2 = r2_score(y, gb_y_pred)
    gb_mae = mean_absolute_error(y, gb_y_pred)
    gb_rmse = np.sqrt(mean_squared_error(y, gb_y_pred))
    
    all_models['gradient_boosting'][industry] = gb_model
    all_predictions['gradient_boosting'][industry] = {
        'years': future_years.flatten(),
        'predictions': gb_pred,
        'r2_score': gb_r2,
        'mae': gb_mae,
        'rmse': gb_rmse
    }
    
    print(f"\n[Method 3] Gradient Boosting:")
    print(f"   R2 Score: {gb_r2:.4f}")
    print(f"   MAE: {gb_mae:.4f}")
    print(f"   RMSE: {gb_rmse:.4f}")
    print(f"   Prediction 2030: {gb_pred[-1]:.4f}")
    
    # ========== METHOD 4: POLYNOMIAL REGRESSION (Degree 2) ==========
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)
    poly_pred = poly_model.predict(future_years)
    poly_y_pred = poly_model.predict(X)
    
    poly_r2 = r2_score(y, poly_y_pred)
    poly_mae = mean_absolute_error(y, poly_y_pred)
    poly_rmse = np.sqrt(mean_squared_error(y, poly_y_pred))
    
    all_models['polynomial'][industry] = poly_model
    all_predictions['polynomial'][industry] = {
        'years': future_years.flatten(),
        'predictions': poly_pred,
        'r2_score': poly_r2,
        'mae': poly_mae,
        'rmse': poly_rmse
    }
    
    print(f"\n[Method 4] Polynomial Regression (Degree 2):")
    print(f"   R2 Score: {poly_r2:.4f}")
    print(f"   MAE: {poly_mae:.4f}")
    print(f"   RMSE: {poly_rmse:.4f}")
    print(f"   Prediction 2030: {poly_pred[-1]:.4f}")
    
    # Simpan hasil ringkasan
    all_results.append({
        'industry': industry,
        'linear_r2': lr_r2,
        'linear_mae': lr_mae,
        'linear_rmse': lr_rmse,
        'linear_pred_2030': lr_pred[-1],
        'rf_r2': rf_r2,
        'rf_mae': rf_mae,
        'rf_rmse': rf_rmse,
        'rf_pred_2030': rf_pred[-1],
        'gb_r2': gb_r2,
        'gb_mae': gb_mae,
        'gb_rmse': gb_rmse,
        'gb_pred_2030': gb_pred[-1],
        'poly_r2': poly_r2,
        'poly_mae': poly_mae,
        'poly_rmse': poly_rmse,
        'poly_pred_2030': poly_pred[-1]
    })

# Simpan semua model dan prediksi
joblib.dump(all_models, os.path.join(MODELS_DIR, 'all_models.pkl'))
joblib.dump(all_predictions, os.path.join(MODELS_DIR, 'all_predictions.pkl'))

print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

results_df = pd.DataFrame(all_results)

# Tampilkan perbandingan R2 Score
print("\nComparison of R2 Score per Industry (higher is better):")
print("-" * 90)
print(results_df[['industry', 'linear_r2', 'rf_r2', 'gb_r2', 'poly_r2']].to_string(index=False))

# Tampilkan perbandingan prediksi 2030
print("\nComparison of 2030 Prediction per Industry:")
print("-" * 90)
print(results_df[['industry', 'linear_pred_2030', 'rf_pred_2030', 'gb_pred_2030', 'poly_pred_2030']].to_string(index=False))

# Hitung rata-rata performa per metode
avg_performance = pd.DataFrame({
    'Method': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Polynomial Regression'],
    'Avg R2 Score': [
        results_df['linear_r2'].mean(),
        results_df['rf_r2'].mean(),
        results_df['gb_r2'].mean(),
        results_df['poly_r2'].mean()
    ],
    'Avg MAE': [
        results_df['linear_mae'].mean(),
        results_df['rf_mae'].mean(),
        results_df['gb_mae'].mean(),
        results_df['poly_mae'].mean()
    ],
    'Avg RMSE': [
        results_df['linear_rmse'].mean(),
        results_df['rf_rmse'].mean(),
        results_df['gb_rmse'].mean(),
        results_df['poly_rmse'].mean()
    ]
}).sort_values('Avg R2 Score', ascending=False)

print("\n" + "="*70)
print("PERFORMANCE RANKING (Average across all industries)")
print("="*70)
print(avg_performance.to_string(index=False))

# Tentukan metode terbaik
best_method = avg_performance.iloc[0]['Method']
print(f"\nBest method based on average R2 Score: {best_method}")

# Simpan hasil komparasi
results_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'), index=False)
avg_performance.to_csv(os.path.join(MODELS_DIR, 'method_ranking.csv'), index=False)

print(f"\nComparison results saved to: {os.path.join(MODELS_DIR, 'model_comparison.csv')}")
print(f"Method ranking saved to: {os.path.join(MODELS_DIR, 'method_ranking.csv')}")