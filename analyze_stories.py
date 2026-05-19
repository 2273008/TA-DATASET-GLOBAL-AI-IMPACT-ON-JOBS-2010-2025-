import pandas as pd
import numpy as np

df = pd.read_csv('data/ai_impact_jobs_2010_2025.csv')

# ========== STORY 1 ==========
print("=" * 70)
print("STORY 1: THE SALARY-AUTOMATION PARADOX")
print("=" * 70)
q75_sal = df['salary_usd'].quantile(0.75)
q75_auto = df['automation_risk_score'].quantile(0.75)
paradox = df[(df['salary_usd'] >= q75_sal) & (df['automation_risk_score'] >= q75_auto)]
normal = df[~((df['salary_usd'] >= q75_sal) & (df['automation_risk_score'] >= q75_auto))]

print(f"Threshold salary Q75: ${q75_sal:,.0f}")
print(f"Threshold automation Q75: {q75_auto}")
print(f"Paradox count: {len(paradox)} ({len(paradox)/len(df)*100:.1f}%)")

print("\nParadox by industry:")
ind_stats = paradox.groupby('industry').agg(
    count=('job_id', 'count'),
    avg_salary=('salary_usd', 'mean'),
    avg_auto=('automation_risk_score', 'mean')
).sort_values('count', ascending=False).round(0)
print(ind_stats)

print("\nParadox by job_title:")
jt_stats = paradox.groupby('job_title').agg(
    count=('job_id', 'count'),
    avg_salary=('salary_usd', 'mean'),
    avg_auto=('automation_risk_score', 'mean'),
    pct_reskilling=('reskilling_required', 'mean')
).sort_values('count', ascending=False).round(3)
print(jt_stats)

print("\nDisplacement risk distribution in paradox:")
print(paradox['ai_job_displacement_risk'].value_counts())

print("\nParadox by region:")
print(paradox.groupby('region').size().sort_values(ascending=False))

print("\nParadox by seniority:")
print(paradox.groupby('seniority_level').size().sort_values(ascending=False))

reskill_paradox = paradox['reskilling_required'].mean() * 100
reskill_normal = normal['reskilling_required'].mean() * 100
print(f"\nReskilling in paradox: {reskill_paradox:.1f}%")
print(f"Reskilling in normal: {reskill_normal:.1f}%")

print("\n5 Example records:")
cols = ['job_title', 'industry', 'region', 'salary_usd', 'automation_risk_score', 'ai_job_displacement_risk', 'reskilling_required', 'seniority_level']
print(paradox[cols].head(5).to_string())

# ========== STORY 2 ==========
print("\n" + "=" * 70)
print("STORY 2: THE SENIORITY SALARY INVERSION")
print("=" * 70)
q90_sal = df['salary_usd'].quantile(0.90)
q10_sal = df['salary_usd'].quantile(0.10)
print(f"Q90 salary: ${q90_sal:,.0f}")
print(f"Q10 salary: ${q10_sal:,.0f}")

junior_high = df[(df['seniority_level'].isin(['Intern', 'Junior'])) & (df['salary_usd'] >= q90_sal)]
exec_low = df[(df['seniority_level'].isin(['Executive', 'Lead'])) & (df['salary_usd'] <= q10_sal)]

print(f"\nJunior/Intern with top 10% salary: {len(junior_high)} records")
print(f"  Avg salary: ${junior_high['salary_usd'].mean():,.0f}")
print("  By industry:")
print(junior_high.groupby('industry').size().sort_values(ascending=False))
print("  By region:")
print(junior_high.groupby('region').size().sort_values(ascending=False))
print("  By job title:")
print(junior_high.groupby('job_title').size().sort_values(ascending=False))
print(f"  AI intensity avg: {junior_high['ai_intensity_score'].mean():.3f}")

print(f"\nExecutive/Lead with bottom 10% salary: {len(exec_low)} records")
print(f"  Avg salary: ${exec_low['salary_usd'].mean():,.0f}")
print("  By industry:")
print(exec_low.groupby('industry').size().sort_values(ascending=False))
print("  By region:")
print(exec_low.groupby('region').size().sort_values(ascending=False))

print("\nOverall Seniority vs Salary:")
print(df.groupby('seniority_level')['salary_usd'].agg(['mean', 'median', 'std']).sort_values('mean', ascending=False).round(0))

# ========== STORY 3 ==========
print("\n" + "=" * 70)
print("STORY 3: THE ADOPTION STAGE CONTRADICTION")
print("=" * 70)
mature_all = df[df['industry_ai_adoption_stage'] == 'Mature']
emerging_all = df[df['industry_ai_adoption_stage'] == 'Emerging']
growing_all = df[df['industry_ai_adoption_stage'] == 'Growing']

mature_low = df[(df['industry_ai_adoption_stage'] == 'Mature') & (df['ai_intensity_score'] < 0.2)]
emerging_high = df[(df['industry_ai_adoption_stage'] == 'Emerging') & (df['ai_intensity_score'] > 0.7)]

print(f"Total Mature: {len(mature_all)}, Emerging: {len(emerging_all)}, Growing: {len(growing_all)}")
print(f"\nMature but LOW intensity (<0.2): {len(mature_low)} ({len(mature_low)/len(mature_all)*100:.1f}% of Mature)")
print("  By industry:")
print(mature_low.groupby('industry').size().sort_values(ascending=False))
print("  By job title:")
print(mature_low.groupby('job_title').size().sort_values(ascending=False))
print(f"  Avg AI intensity: {mature_low['ai_intensity_score'].mean():.3f}")
print("  Displacement risk:")
print(mature_low['ai_job_displacement_risk'].value_counts())

print(f"\nEmerging but HIGH intensity (>0.7): {len(emerging_high)} ({len(emerging_high)/len(emerging_all)*100:.1f}% of Emerging)")
print("  By industry:")
print(emerging_high.groupby('industry').size().sort_values(ascending=False))
print("  By job title:")
print(emerging_high.groupby('job_title').size().sort_values(ascending=False))
print(f"  Avg AI intensity: {emerging_high['ai_intensity_score'].mean():.3f}")

print("\nOverall Adoption Stage vs AI Intensity:")
print(df.groupby('industry_ai_adoption_stage')['ai_intensity_score'].agg(['mean', 'median', 'std', 'min', 'max']).round(3))

print("\nIndustry x Adoption Stage:")
ct = pd.crosstab(df['industry'], df['industry_ai_adoption_stage'])
print(ct)

# ========== STORY 4 ==========
print("\n" + "=" * 70)
print("STORY 4: THE REGIONAL DIGITAL DIVIDE")
print("=" * 70)
region_stats = df.groupby('region').agg(
    count=('job_id', 'count'),
    avg_salary=('salary_usd', 'mean'),
    median_salary=('salary_usd', 'median'),
    std_salary=('salary_usd', 'std'),
    avg_ai_intensity=('ai_intensity_score', 'mean'),
    avg_auto_risk=('automation_risk_score', 'mean'),
    pct_reskill=('reskilling_required', 'mean')
).sort_values('avg_salary', ascending=False).round(2)
print(region_stats)

top_region = region_stats.index[0]
bot_region = region_stats.index[-1]
ratio = region_stats.loc[top_region, 'avg_salary'] / region_stats.loc[bot_region, 'avg_salary']
print(f"\nSalary ratio {top_region} vs {bot_region}: {ratio:.1f}x")

print("\nRegion x Displacement Risk:")
ct2 = pd.crosstab(df['region'], df['ai_job_displacement_risk'], normalize='index').mul(100).round(1)
print(ct2)

print("\nRegion x Adoption Stage:")
ct3 = pd.crosstab(df['region'], df['industry_ai_adoption_stage'], normalize='index').mul(100).round(1)
print(ct3)

# Same job title, different region salary
print("\nSame Job Title salary by region:")
for jt in ['Data Scientist', 'ML Engineer', 'Software Engineer']:
    sub = df[df['job_title'] == jt]
    rs = sub.groupby('region')['salary_usd'].mean().sort_values(ascending=False).round(0)
    print(f"\n  {jt}:")
    print(f"  {rs.to_dict()}")

# ========== STORY 5 ==========
print("\n" + "=" * 70)
print("STORY 5: THE RESKILLING BLIND SPOT")
print("=" * 70)
blind_spot = df[(df['reskilling_required'] == False) & (df['automation_risk_score'] > 0.8)]
prepared = df[(df['reskilling_required'] == True) & (df['automation_risk_score'] > 0.8)]

print(f"High auto risk (>0.8) but NO reskilling: {len(blind_spot)} ({len(blind_spot)/len(df)*100:.1f}% of total)")
print(f"High auto risk (>0.8) WITH reskilling: {len(prepared)} ({len(prepared)/len(df)*100:.1f}% of total)")
print(f"Total high auto risk: {len(blind_spot)+len(prepared)}")

print("\nBlind spot by industry:")
print(blind_spot.groupby('industry').size().sort_values(ascending=False))

print("\nBlind spot by job title:")
print(blind_spot.groupby('job_title').size().sort_values(ascending=False))

print("\nBlind spot by region:")
print(blind_spot.groupby('region').size().sort_values(ascending=False))

print("\nBlind spot displacement risk:")
print(blind_spot['ai_job_displacement_risk'].value_counts())

print("\nBlind spot by seniority:")
print(blind_spot.groupby('seniority_level').size().sort_values(ascending=False))

print("\nBlind spot avg salary vs prepared:")
print(f"  Blind spot avg salary: ${blind_spot['salary_usd'].mean():,.0f}")
print(f"  Prepared avg salary: ${prepared['salary_usd'].mean():,.0f}")

print("\nBlind spot by posting year:")
print(blind_spot.groupby('posting_year').size().sort_values(ascending=False).head(10))

# Reskilling overall by industry
print("\nReskilling rate by industry (all data):")
rr = df.groupby('industry')['reskilling_required'].mean().mul(100).round(1).sort_values(ascending=False)
print(rr)
