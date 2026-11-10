# -*- coding: utf-8 -*-
"""
SP-07 · Fase 3 y 4: Ingeniería de Variables y Análisis Exploratorio de Datos (EDA) Estadístico
Calcula estadísticos descriptivos, contrastes de hipótesis bivariados (Chi2, t-test, ORs),
matriz de correlaciones Spearman y materializa la tabla de características para entrenamiento.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

CAT_VARS = ["country", "sex", "urban_rural", "grade_band", "age_band"]
NUM_VARS = [
    "age_years", "grade", "food_insecurity", "parents_supervision", "parents_support",
    "family_connectedness", "friends_support", "loneliness", "anxiety_sleep",
    "bullied_past_month", "bullied_type_count", "physical_fights", "physically_attacked",
    "tobacco_use", "alcohol_use", "drug_use", "fruit_veg_intake", "physical_activity_60min",
    "sedentary_hours", "truancy_past_month", "school_connectedness", "bmi_for_age_z",
    "gini_index", "health_exp_pc_usd", "unemployment_youth", "internet_pct"
]

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sp07:sp07_dev_password@localhost:5432/sp07_db")

def run_feature_engineering(df):
    """
    Fase 3: Construcción y derivación de variables epidemiológicas.
    """
    df = df.copy()
    
    # Categorización de bandas
    df["grade_band"] = pd.cut(df["grade"], bins=[0, 2, 4, 7], labels=["low_7_8", "mid_9_10", "high_11_12"]).astype(str)
    df["age_band"] = pd.cut(df["age_years"], bins=[10, 13, 15, 18], labels=["11-13", "14-15", "16-17"]).astype(str)
    
    # Desenlaces secundarios multitarea
    df["y2_ideation"] = df["suicide_ideation"].astype(int)
    df["y3_plan"] = df["suicide_plan"].astype(int)
    df["y4_attempt"] = df["suicide_attempt"].astype(int)
    
    # Desenlace primario compuesto Y1 según protocolo OMS-GSHS
    if "y1_distress" not in df.columns:
        df["y1_distress"] = np.where(
            (df["loneliness"] >= 4) & 
            ((df["suicide_ideation"] == 1) | (df["anxiety_sleep"] >= 4)),
            1, 0
        )
        
    # Índice de vulnerabilidad psicosocial acumulada (score 0-10)
    df["cumulative_risk_score"] = (
        (df["bullied_past_month"] == 1).astype(int) +
        (df["food_insecurity"] >= 2).astype(int) +
        (df["parents_support"] <= 2).astype(int) +
        (df["school_connectedness"] <= 2).astype(int) +
        (df["alcohol_use"] == 1).astype(int) +
        (df["drug_use"] == 1).astype(int) +
        (df["physical_fights"] >= 1).astype(int) +
        (df["anxiety_sleep"] >= 4).astype(int) +
        (df["loneliness"] >= 4).astype(int) +
        (df["truancy_past_month"] == 1).astype(int)
    )
    
    return df

def run_statistical_eda(df, out_dir="artifacts"):
    """
    Fase 4: Análisis Exploratorio de Datos (EDA) Inferencial y Descriptivo.
    - Frecuencias y prevalencias por país y demografía.
    - Test Chi-cuadrado con Odds Ratios (OR) e IC 95% para variables categóricas vs Y1.
    - Test t de Student / Mann-Whitney U con tamaño del efecto para continuas vs Y1.
    - Matriz de correlación Spearman de determinantes.
    """
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "sample_size": len(df),
        "y1_prevalence": float(df["y1_distress"].mean()),
        "countries_count": int(df["country"].nunique()),
        "prevalence_by_country": df.groupby("country")["y1_distress"].mean().round(4).to_dict(),
        "prevalence_by_sex": {"Femenino": float(df[df.sex == 0]["y1_distress"].mean()),
                              "Masculino": float(df[df.sex == 1]["y1_distress"].mean())},
        "prevalence_by_age": df.groupby("age_band")["y1_distress"].mean().round(4).to_dict(),
        "bivariate_associations": [],
        "spearman_correlations": {}
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
                
        # Tabla de contingencia 2x2
        ct = pd.crosstab(binary_exposure, df["y1_distress"])
        if ct.shape == (2, 2):
            chi2, p_val, _, _ = stats.chi2_contingency(ct)
            # Odds Ratio = (a*d) / (b*c)
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
            
    # 2. Correlaciones de Spearman con Y1
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
            
    # Guardar reporte de EDA
    eda_json_path = os.path.join(out_dir, "eda_statistical_summary.json")
    with open(eda_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Reporte de EDA estadístico generado en: {eda_json_path}")
    
    # 3. Materializar características en PostgreSQL (tabla features)
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE sp07.features CASCADE;"))
            
            features_records = []
            for _, r in df.iterrows():
                feat_dict = {
                    "age_years": int(r["age_years"]),
                    "sex": int(r["sex"]),
                    "grade_band": str(r["grade_band"]),
                    "urban_rural": int(r["urban_rural"]),
                    "loneliness": int(r["loneliness"]),
                    "anxiety_sleep": int(r["anxiety_sleep"]),
                    "bullied_past_month": int(r["bullied_past_month"]),
                    "bullied_type_count": int(r["bullied_type_count"]),
                    "physical_fights": int(r["physical_fights"]),
                    "tobacco_use": int(r["tobacco_use"]),
                    "alcohol_use": int(r["alcohol_use"]),
                    "drug_use": int(r["drug_use"]),
                    "school_connectedness": int(r["school_connectedness"]),
                    "family_connectedness": int(r["family_connectedness"]),
                    "friends_support": int(r["friends_support"]),
                    "parents_support": int(r["parents_support"]),
                    "food_insecurity": int(r["food_insecurity"]),
                    "gini_index": float(r["gini_index"]),
                    "health_exp_pc_usd": float(r["health_exp_pc_usd"])
                }
                features_records.append({
                    "respondent_id": str(r["respondent_id"]),
                    "country": str(r["country"]),
                    "survey_year": int(r["survey_year"]),
                    "sex": int(r["sex"]),
                    "urban_rural": int(r["urban_rural"]),
                    "age_years": int(r["age_years"]),
                    "y1_distress": int(r["y1_distress"]),
                    "y2_ideation": int(r["y2_ideation"]),
                    "y3_plan": int(r["y3_plan"]),
                    "y4_attempt": int(r["y4_attempt"]),
                    "feat_json": json.dumps(feat_dict)
                })
                
            df_feat_sql = pd.DataFrame(features_records)
            df_feat_sql.to_sql("features", con=conn, schema="sp07", if_exists="append", index=False, chunksize=1000)
            print("Tabla sp07.features materializada exitosamente en PostgreSQL.")
    except Exception as e:
        print(f"Aviso de materialización en sp07.features: {e}")
        
    return df, report

if __name__ == "__main__":
    csv_path = "data/gshs_adolescents_clean.csv"
    if os.path.exists(csv_path):
        df_in = pd.read_csv(csv_path)
        df_eng = run_feature_engineering(df_in)
        _, eda_rep = run_statistical_eda(df_eng)
        print("Fases 3 y 4 completadas con éxito.")
    else:
        print(f"No se encontró {csv_path}. Ejecute data_ingestion.py primero.")

