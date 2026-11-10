# -*- coding: utf-8 -*-
"""
MÓDULO 3 · FASE 3: Ingeniería de Variables y Derivación de Desenlaces Epidemiológicos
Construye el desenlace primario compuesto Y1 (Distress), los desenlaces multitarea Y2..Y4,
las bandas demográficas, el score de riesgo acumulado y materializa la tabla sp07.features.
"""

import os
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sp07:sp07_dev_password@localhost:5432/sp07_db")

def run_phase3_feature_engineering(df=None, data_dir="data"):
    """
    Ejecuta Fase 3: Ingeniería de características y desenlaces clínicos.
    """
    os.makedirs(data_dir, exist_ok=True)
    print("[FASE 3] Iniciando ingeniería de características y desenlaces epidemiológicos...")
    
    if df is None:
        clean_csv = os.path.join(data_dir, "gshs_adolescents_clean.csv")
        if os.path.exists(clean_csv):
            df = pd.read_csv(clean_csv)
        else:
            from phase2_quality_audit import run_phase2_quality_audit
            df, _ = run_phase2_quality_audit(data_dir=data_dir)
            
    df = df.copy()
    
    # 1. Bandas demográficas
    df["grade_band"] = pd.cut(df["grade"], bins=[0, 2, 4, 7], labels=["low_7_8", "mid_9_10", "high_11_12"]).astype(str)
    df["age_band"] = pd.cut(df["age_years"], bins=[10, 13, 15, 18], labels=["11-13", "14-15", "16-17"]).astype(str)
    
    # 2. Desenlaces secundarios multitarea
    df["y2_ideation"] = df["suicide_ideation"].astype(int)
    df["y3_plan"] = df["suicide_plan"].astype(int)
    df["y4_attempt"] = df["suicide_attempt"].astype(int)
    
    # 3. Desenlace primario compuesto Y1 (OMS-GSHS)
    # Soledad severa frecuente (>=4) Y (Ideación suicida == 1 O Ansiedad de sueño >= 4)
    df["y1_distress"] = np.where(
        (df["loneliness"] >= 4) & 
        ((df["suicide_ideation"] == 1) | (df["anxiety_sleep"] >= 4)),
        1, 0
    )
    
    # 4. Score de vulnerabilidad psicosocial acumulada (0-10)
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
    
    out_csv = os.path.join(data_dir, "gshs_features_engineered.csv")
    df.to_csv(out_csv, index=False)
    
    # 5. Materialización en PostgreSQL tabla sp07.features
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
                
            df_sql = pd.DataFrame(features_records)
            df_sql.to_sql("features", con=conn, schema="sp07", if_exists="append", index=False, chunksize=1000)
            print("  ✓ Tabla sp07.features materializada exitosamente en PostgreSQL.")
    except Exception as e:
        print(f"  Aviso PostgreSQL en Fase 3: {e}")
        
    print(f"  ✓ Fase 3 completada: Y1_distress prevalencia = {df['y1_distress'].mean():.2%}, N = {len(df)}.")
    return df

if __name__ == "__main__":
    run_phase3_feature_engineering()

