# -*- coding: utf-8 -*-
"""
MÓDULO 4 · FASE 4: Análisis Exploratorio de Datos (EDA) Inferencial y Multidimensional
Calcula pruebas de hipótesis bivariadas (Chi2, Odds Ratios con IC 95%), correlaciones de Spearman,
distribuciones poblacionales por país y sintetiza los patrones preliminares de vulnerabilidad.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

def run_phase4_statistical_eda(df=None, data_dir="data", out_dir="artifacts"):
    """
    Ejecuta Fase 4: Análisis Exploratorio de Datos (EDA) Estadístico.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("[FASE 4] Iniciando Análisis Exploratorio de Datos (EDA) Inferencial...")
    
    if df is None:
        feat_csv = os.path.join(data_dir, "gshs_features_engineered.csv")
        if os.path.exists(feat_csv):
            df = pd.read_csv(feat_csv)
        else:
            from phase3_feature_engineering import run_phase3_feature_engineering
            df = run_phase3_feature_engineering(data_dir=data_dir)
            
    report = {
        "sample_size": len(df),
        "y1_prevalence": float(df["y1_distress"].mean()),
        "countries_count": int(df["country"].nunique()),
        "prevalence_by_country": df.groupby("country")["y1_distress"].mean().round(4).to_dict(),
        "prevalence_by_sex": {
            "Femenino (0)": float(df[df.sex == 0]["y1_distress"].mean()),
            "Masculino (1)": float(df[df.sex == 1]["y1_distress"].mean())
        },
        "prevalence_by_age": df.groupby("age_band")["y1_distress"].mean().round(4).to_dict(),
        "bivariate_associations": [],
        "spearman_correlations": {},
        "descriptive_univariate": {}
    }
    
    # 1. Asociaciones bivariadas para determinantes categóricos y ordinales
    test_vars = [
        ("bullied_past_month", "Acoso escolar (últimos 30d)", "binaria"),
        ("food_insecurity", "Inseguridad alimentaria (≥2)", "binaria_d"),
        ("parents_support", "Bajo apoyo parental (≤2)", "binaria_d"),
        ("school_connectedness", "Baja conexión escolar (≤2)", "binaria_d"),
        ("truancy_past_month", "Absentismo escolar injustificado", "binaria"),
        ("alcohol_use", "Consumo reciente de alcohol", "binaria"),
        ("drug_use", "Consumo de drogas ilícitas", "binaria"),
        ("physical_fights", "Peleas físicas (≥1)", "binaria_d"),
        ("anxiety_sleep", "Ansiedad que interrumpe el sueño (≥4)", "binaria_d"),
        ("loneliness", "Soledad grave frecuente (≥4)", "binaria_d"),
        ("sex", "Sexo femenino (referencia masc.)", "binaria_d")
    ]
    
    for var, label, vtype in test_vars:
        if vtype == "binaria":
            binary_exposure = (df[var] == 1).astype(int)
        elif vtype == "binaria_d":
            if var == "food_insecurity":
                binary_exposure = (df[var] >= 2).astype(int)
            elif var in ("parents_support", "school_connectedness"):
                binary_exposure = (df[var] <= 2).astype(int)
            elif var in ("anxiety_sleep", "loneliness"):
                binary_exposure = (df[var] >= 4).astype(int)
            elif var == "physical_fights":
                binary_exposure = (df[var] >= 1).astype(int)
            elif var == "sex":
                binary_exposure = (df["sex"] == 0).astype(int)
                
        ct = pd.crosstab(binary_exposure, df["y1_distress"])
        if ct.shape == (2, 2):
            chi2, p_val, _, _ = stats.chi2_contingency(ct)
            a = ct.loc[1, 1] if (1 in ct.index and 1 in ct.columns) else 0
            b = ct.loc[1, 0] if (1 in ct.index and 0 in ct.columns) else 0
            c = ct.loc[0, 1] if (0 in ct.index and 1 in ct.columns) else 0
            d = ct.loc[0, 0] if (0 in ct.index and 0 in ct.columns) else 0
            
            if b * c > 0:
                odds_ratio = (a * d) / (b * c)
                log_or = np.log(odds_ratio)
                se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
                ci_low = np.exp(log_or - 1.96 * se_log_or)
                ci_high = np.exp(log_or + 1.96 * se_log_or)
            else:
                odds_ratio, ci_low, ci_high = np.nan, np.nan, np.nan
                
            report["bivariate_associations"].append({
                "determinante": label,
                "variable": var,
                "chi2_stat": round(float(chi2), 2),
                "p_value": float(p_val),
                "odds_ratio": round(float(odds_ratio), 2) if not np.isnan(odds_ratio) else None,
                "ci_95": f"[{ci_low:.2f} - {ci_high:.2f}]" if not np.isnan(ci_low) else "N/A",
                "significativo": bool(p_val < 0.05)
            })
            
    # 2. Correlaciones de Spearman
    corr_vars = [
        "loneliness", "anxiety_sleep", "bullied_past_month", "bullied_type_count",
        "parents_support", "family_connectedness", "school_connectedness",
        "food_insecurity", "alcohol_use", "drug_use", "gini_index", "health_exp_pc_usd"
    ]
    for cv in corr_vars:
        if cv in df.columns:
            rho, pval = stats.spearmanr(df[cv].fillna(df[cv].median()), df["y1_distress"])
            report["spearman_correlations"][cv] = {
                "rho": round(float(rho), 3),
                "p_value": float(pval)
            }
            
    # 3. Guardar reporte de EDA
    eda_json_path = os.path.join(out_dir, "eda_statistical_summary.json")
    with open(eda_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"  ✓ Asociaciones bivariadas evaluadas: {len(report['bivariate_associations'])} determinantes.")
    print(f"  ✓ Reporte de EDA estadístico guardado en: {eda_json_path}")
    return df, report

if __name__ == "__main__":
    run_phase4_statistical_eda()

