import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI PATH ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
file_path = os.path.join(DATA_DIR, 'ai_impact_jobs_2010_2025.csv')

# Load data
df = pd.read_csv(file_path)
print(f"Original data shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

# ==================== FUNGSI REMOVE OUTLIER ====================

def remove_outliers_iqr(df, column, group_columns=None, multiplier=1.5):
    """
    Menghapus outlier menggunakan metode IQR (Interquartile Range)
    
    Parameters:
    - df: DataFrame
    - column: nama kolom yang akan dibersihkan
    - group_columns: list kolom untuk grouping (misal ['industry', 'seniority_level'])
    - multiplier: faktor IQR (default 1.5)
    
    Returns:
    - df_clean: DataFrame tanpa outlier
    - outliers_df: DataFrame yang berisi outlier
    """
    
    if group_columns is None:
        group_columns = []
    
    df_clean = df.copy()
    outlier_indices = []
    
    if len(group_columns) > 0:
        # Group by specified columns
        grouped = df.groupby(group_columns)
        
        for group_keys, group_data in grouped:
            if len(group_data) > 0:
                Q1 = group_data[column].quantile(0.25)
                Q3 = group_data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                # Find outliers in this group
                group_outliers = group_data[(group_data[column] < lower_bound) | (group_data[column] > upper_bound)].index.tolist()
                outlier_indices.extend(group_outliers)
    else:
        # No grouping, apply to entire column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
    
    outliers_df = df.loc[outlier_indices]
    df_clean = df.drop(index=outlier_indices)
    
    return df_clean, outliers_df


def remove_outliers_zscore(df, column, group_columns=None, threshold=3):
    """
    Menghapus outlier menggunakan metode Z-Score
    
    Parameters:
    - df: DataFrame
    - column: nama kolom yang akan dibersihkan
    - group_columns: list kolom untuk grouping
    - threshold: batas Z-Score (default 3)
    
    Returns:
    - df_clean: DataFrame tanpa outlier
    - outliers_df: DataFrame yang berisi outlier
    """
    
    if group_columns is None:
        group_columns = []
    
    df_clean = df.copy()
    outlier_indices = []
    
    if len(group_columns) > 0:
        grouped = df.groupby(group_columns)
        
        for group_keys, group_data in grouped:
            if len(group_data) > 1:
                z_scores = np.abs(stats.zscore(group_data[column].dropna()))
                group_outliers = group_data[z_scores > threshold].index.tolist()
                outlier_indices.extend(group_outliers)
    else:
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outlier_indices = df[z_scores > threshold].index.tolist()
    
    outliers_df = df.loc[outlier_indices]
    df_clean = df.drop(index=outlier_indices)
    
    return df_clean, outliers_df


def remove_outliers_percentile(df, column, group_columns=None, lower_percentile=1, upper_percentile=99):
    """
    Menghapus outlier menggunakan metode Percentile
    
    Parameters:
    - df: DataFrame
    - column: nama kolom yang akan dibersihkan
    - group_columns: list kolom untuk grouping
    - lower_percentile: persentil bawah (default 1)
    - upper_percentile: persentil atas (default 99)
    
    Returns:
    - df_clean: DataFrame tanpa outlier
    - outliers_df: DataFrame yang berisi outlier
    """
    
    if group_columns is None:
        group_columns = []
    
    df_clean = df.copy()
    outlier_indices = []
    
    if len(group_columns) > 0:
        grouped = df.groupby(group_columns)
        
        for group_keys, group_data in grouped:
            if len(group_data) > 0:
                lower_bound = group_data[column].quantile(lower_percentile / 100)
                upper_bound = group_data[column].quantile(upper_percentile / 100)
                
                group_outliers = group_data[(group_data[column] < lower_bound) | (group_data[column] > upper_bound)].index.tolist()
                outlier_indices.extend(group_outliers)
    else:
        lower_bound = df[column].quantile(lower_percentile / 100)
        upper_bound = df[column].quantile(upper_percentile / 100)
        outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
    
    outliers_df = df.loc[outlier_indices]
    df_clean = df.drop(index=outlier_indices)
    
    return df_clean, outliers_df


# ==================== TERAPKAN REMOVE OUTLIER ====================

print("\n" + "="*60)
print("REMOVING OUTLIERS FROM DATA")
print("="*60)

# Buat copy data asli
df_clean = df.copy()

# ==================== 1. REMOVE OUTLIER UNTUK AI INTENSITY SCORE ====================
print("\n1. Cleaning AI Intensity Score...")

# Remove outliers for ai_intensity_score by industry and seniority_level
if 'seniority_level' in df.columns:
    df_clean, outliers_intensity = remove_outliers_iqr(
        df_clean, 
        'ai_intensity_score', 
        group_columns=['industry', 'seniority_level'],
        multiplier=1.5
    )
    print(f"   Removed {len(outliers_intensity)} outliers from ai_intensity_score")
else:
    df_clean, outliers_intensity = remove_outliers_iqr(
        df_clean, 
        'ai_intensity_score', 
        group_columns=['industry'],
        multiplier=1.5
    )
    print(f"   Removed {len(outliers_intensity)} outliers from ai_intensity_score")

# ==================== 2. REMOVE OUTLIER UNTUK SALARY ====================
print("\n2. Cleaning Salary...")

if 'seniority_level' in df.columns:
    df_clean, outliers_salary = remove_outliers_iqr(
        df_clean, 
        'salary_usd', 
        group_columns=['industry', 'seniority_level'],
        multiplier=1.5
    )
    print(f"   Removed {len(outliers_salary)} outliers from salary_usd")
else:
    df_clean, outliers_salary = remove_outliers_iqr(
        df_clean, 
        'salary_usd', 
        group_columns=['industry'],
        multiplier=1.5
    )
    print(f"   Removed {len(outliers_salary)} outliers from salary_usd")

# ==================== 3. REMOVE OUTLIER UNTUK AUTOMATION RISK SCORE ====================
print("\n3. Cleaning Automation Risk Score...")

if 'seniority_level' in df.columns:
    df_clean, outliers_auto = remove_outliers_iqr(
        df_clean, 
        'automation_risk_score', 
        group_columns=['industry', 'seniority_level'],
        multiplier=1.5
    )
    print(f"   Removed {len(outliers_auto)} outliers from automation_risk_score")
else:
    df_clean, outliers_auto = remove_outliers_iqr(
        df_clean, 
        'automation_risk_score', 
        group_columns=['industry'],
        multiplier=1.5
    )
    print(f"   Removed {len(outliers_auto)} outliers from automation_risk_score")

# ==================== 4. KHUSUS REMOVE OUTLIER UNTUK SENIORITY LEVEL LEAD ====================
print("\n4. Specific cleaning for Seniority Level = Lead...")

if 'seniority_level' in df.columns:
    # Focus on Lead level in Manufacturing and Healthcare
    lead_data = df_clean[df_clean['seniority_level'] == 'Lead']
    
    if len(lead_data) > 0:
        # Remove outliers for Lead in Manufacturing
        manufacturing_lead = df_clean[(df_clean['industry'] == 'Manufacturing') & (df_clean['seniority_level'] == 'Lead')]
        if len(manufacturing_lead) > 0:
            manuf_lead_clean, manuf_lead_outliers = remove_outliers_iqr(
                manufacturing_lead,
                'ai_intensity_score',
                group_columns=['industry', 'seniority_level'],
                multiplier=1.2
            )
            print(f"   Manufacturing Lead: removed {len(manuf_lead_outliers)} outliers")
            df_clean = df_clean.drop(index=manuf_lead_outliers.index)
        
        # Remove outliers for Lead in Healthcare
        healthcare_lead = df_clean[(df_clean['industry'] == 'Healthcare') & (df_clean['seniority_level'] == 'Lead')]
        if len(healthcare_lead) > 0:
            health_lead_clean, health_lead_outliers = remove_outliers_iqr(
                healthcare_lead,
                'ai_intensity_score',
                group_columns=['industry', 'seniority_level'],
                multiplier=1.2
            )
            print(f"   Healthcare Lead: removed {len(health_lead_outliers)} outliers")
            df_clean = df_clean.drop(index=health_lead_outliers.index)

# ==================== 5. REMOVE OUTLIER UNTUK AI JOB DISPLACEMENT RISK (optional) ====================
print("\n5. Cleaning AI Job Displacement Risk (categorical - no outlier removal needed)")
print("   Skipping categorical column")

# ==================== 6. REMOVE OUTLIER UNTUK RESKILLING REQUIRED (optional) ====================
print("\n6. Cleaning Reskilling Required (boolean - no outlier removal needed)")
print("   Skipping boolean column")

# ==================== HASIL AKHIR ====================
print("\n" + "="*60)
print("CLEANING RESULTS")
print("="*60)
print(f"Original data shape: {df.shape}")
print(f"Cleaned data shape: {df_clean.shape}")
print(f"Total outliers removed: {df.shape[0] - df_clean.shape[0]}")
print(f"Percentage of data retained: {(df_clean.shape[0] / df.shape[0] * 100):.2f}%")

# ==================== VISUALISASI PERBANDINGAN ====================

# Function to create comparison plots
def create_comparison_plot(df_original, df_cleaned, column, title):
    fig = go.Figure()
    
    # Add original data box plot
    fig.add_trace(go.Box(
        y=df_original[column],
        name='Original',
        boxmean='sd',
        marker_color='lightblue'
    ))
    
    # Add cleaned data box plot
    fig.add_trace(go.Box(
        y=df_cleaned[column],
        name='After Outlier Removal',
        boxmean='sd',
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title=f'{title} - Before vs After Outlier Removal',
        yaxis_title=column,
        template='plotly_white',
        height=500
    )
    
    return fig

# Create comparison plots
if 'ai_intensity_score' in df.columns:
    fig1 = create_comparison_plot(df, df_clean, 'ai_intensity_score', 'AI Intensity Score')
    fig1.show()

if 'salary_usd' in df.columns:
    fig2 = create_comparison_plot(df, df_clean, 'salary_usd', 'Salary')
    fig2.show()

if 'automation_risk_score' in df.columns:
    fig3 = create_comparison_plot(df, df_clean, 'automation_risk_score', 'Automation Risk Score')
    fig3.show()

# ==================== SAVE CLEANED DATA ====================
output_path = os.path.join(DATA_DIR, 'ai_impact_jobs_cleaned.csv')
df_clean.to_csv(output_path, index=False)
print(f"\nCleaned data saved to: {output_path}")

# Save outliers for analysis
outliers_path = os.path.join(DATA_DIR, 'outliers_detected.csv')
outliers_combined = pd.concat([outliers_intensity, outliers_salary, outliers_auto]).drop_duplicates()
outliers_combined.to_csv(outliers_path, index=False)
print(f"Outliers data saved to: {outliers_path}")

# ==================== STATISTIK SETELAH CLEANING ====================
print("\n" + "="*60)
print("STATISTICS AFTER CLEANING")
print("="*60)

print("\nAI Intensity Score by Industry and Seniority Level:")
if 'seniority_level' in df_clean.columns:
    stats_intensity = df_clean.groupby(['industry', 'seniority_level'])['ai_intensity_score'].agg(['mean', 'std', 'count']).round(3)
    print(stats_intensity.head(20))
else:
    stats_intensity = df_clean.groupby('industry')['ai_intensity_score'].agg(['mean', 'std', 'count']).round(3)
    print(stats_intensity)

print("\nSalary by Industry and Seniority Level:")
if 'seniority_level' in df_clean.columns:
    stats_salary = df_clean.groupby(['industry', 'seniority_level'])['salary_usd'].agg(['mean', 'std', 'count']).round(0).astype(int)
    print(stats_salary.head(20))
else:
    stats_salary = df_clean.groupby('industry')['salary_usd'].agg(['mean', 'std', 'count']).round(0).astype(int)
    print(stats_salary)

print("\nAutomation Risk Score by Industry and Seniority Level:")
if 'seniority_level' in df_clean.columns:
    stats_auto = df_clean.groupby(['industry', 'seniority_level'])['automation_risk_score'].agg(['mean', 'std', 'count']).round(3)
    print(stats_auto.head(20))
else:
    stats_auto = df_clean.groupby('industry')['automation_risk_score'].agg(['mean', 'std', 'count']).round(3)
    print(stats_auto)