# -*- coding: utf-8 -*-
"""
MÓDULO 3 · FASE 3: Ingeniería de Variables y Derivación de Desenlaces Epidemiológicos
Construye los desenlaces formales según PROTOCOLO_SP07_CORREGIDO.md (v2.0):
  - Y1 (Primario): Ideación suicida autorreportada en últimos 12 meses (suicide_ideation == 1).
  - Y2 (Secundario): Plan suicida (suicide_plan == 1).
  - Y3 (Secundario): Intento suicida (suicide_attempt == 1).
  - Y4 (Secundario): Soledad severa frecuente (loneliness >= 4).
  - Y5 (Secundario): Preocupación que interfiere con el sueño (anxiety_sleep >= 4).
  - Y1_distress (Sensibilidad/secundario): Indicador compuesto previo retenido para comparación.
Construye bandas demográficas, score acumulativo SIN FUGA (excluye desenlaces)
y materializa la tabla sp07.features.
"""

import os
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sp07:sp07_dev_password@localhost:5432/sp07_db")

def derive_outcomes_and_features(df):
    """
    Deriva desenlaces e ingeniería de variables respetando el Protocolo v2.0.
    """
    df = df.copy()
    
    # 1. Bandas demográficas
    df["grade_band"] = pd.cut(df["grade"], bins=[0, 2, 4, 7], labels=["low_7_8", "mid_9_10", "high_11_12"]).astype(str)
    df["age_band"] = pd.cut(df["age_years"], bins=[10, 13, 15, 18], labels=["11-13", "14-15", "16-17"]).astype(str)
    
    # 2. Desenlaces primario y secundarios según Protocolo v2.0
    df["y1_ideation"] = (df["suicide_ideation"] == 1).astype(int)
    df["y2_plan"] = (df["suicide_plan"] == 1).astype(int)
    df["y3_attempt"] = (df["suicide_attempt"] == 1).astype(int)
    df["y4_loneliness_frequent"] = (df["loneliness"] >= 4).astype(int)
    df["y5_worry_sleep_frequent"] = (df["anxiety_sleep"] >= 4).astype(int)
    
    # Indicador compuesto retenido exclusivamente para sensibilidad
    df["y1_distress"] = np.where(
        (df["loneliness"] >= 4) & 
        ((df["suicide_ideation"] == 1) | (df["anxiety_sleep"] >= 4)),
        1, 0
    )
    
    # 3. Score de vulnerabilidad psicosocial acumulada (0-10)
    # REGLA ESTRICTA CONTRA FUGA: Excluye soledad, insomnio/ansiedad e ideación suicida.
    df["cumulative_risk_score"] = (
        (df["bullied_past_month"] == 1).astype(int) +
        (df["food_insecurity"] >= 2).astype(int) +
        (df["parents_support"] <= 2).astype(int) +
        (df["school_connectedness"] <= 2).astype(int) +
        (df["alcohol_use"] == 1).astype(int) +
        (df["drug_use"] == 1).astype(int) +
        (df["physical_fights"] >= 1).astype(int) +
        (df["physically_attacked"] == 1).astype(int) +
        (df["tobacco_use"] == 1).astype(int) +
        (df["truancy_past_month"] == 1).astype(int)
    )
    
    return df

def run_phase3_feature_engineering(df=None, data_dir="data"):
    """
    Ejecuta Fase 3: Ingeniería de características y desenlaces epidemiológicos.
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
            
    df = derive_outcomes_and_features(df)
    
    out_csv = os.path.join(data_dir, "gshs_features_engineered.csv")
    df.to_csv(out_csv, index=False)
    
    # Materialización en PostgreSQL tabla sp07.features
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            # Asegurar columnas en PostgreSQL sp07.features
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sp07.features (
                    respondent_id     UUID PRIMARY KEY,
                    country           CHAR(3)  NOT NULL,
                    survey_year       SMALLINT NOT NULL,
                    sex               SMALLINT,
                    urban_rural       SMALLINT,
                    age_years         SMALLINT,
                    y1_ideation       SMALLINT NOT NULL,
                    y2_plan           SMALLINT,
                    y3_attempt        SMALLINT,
                    y4_loneliness_frequent SMALLINT,
                    y5_worry_sleep_frequent SMALLINT,
                    y1_distress       SMALLINT,
                    feat_json         JSONB NOT NULL
                );
            """))
            conn.execute(text("TRUNCATE TABLE sp07.features CASCADE;"))
            
            features_records = []
            for _, r in df.iterrows():
                # Feat dict contiene predictores SIN ETIQUETAS
                feat_dict = {
                    "age_years": int(r["age_years"]),
                    "sex": int(r["sex"]),
                    "grade_band": str(r["grade_band"]),
                    "urban_rural": int(r["urban_rural"]),
                    "bullied_past_month": int(r["bullied_past_month"]),
                    "bullied_type_count": int(r["bullied_type_count"]),
                    "physical_fights": int(r["physical_fights"]),
                    "physically_attacked": int(r["physically_attacked"]),
                    "tobacco_use": int(r["tobacco_use"]),
                    "alcohol_use": int(r["alcohol_use"]),
                    "drug_use": int(r["drug_use"]),
                    "fruit_veg_intake": int(r["fruit_veg_intake"]),
                    "physical_activity_60min": int(r["physical_activity_60min"]),
                    "sedentary_hours": int(r["sedentary_hours"]),
                    "truancy_past_month": int(r["truancy_past_month"]),
                    "school_connectedness": int(r["school_connectedness"]),
                    "family_connectedness": int(r["family_connectedness"]),
                    "friends_support": int(r["friends_support"]),
                    "parents_support": int(r["parents_support"]),
                    "parents_supervision": int(r["parents_supervision"]),
                    "food_insecurity": int(r["food_insecurity"]),
                    "cumulative_risk_score": int(r["cumulative_risk_score"]),
                    "gini_index": float(r["gini_index"]),
                    "health_exp_pc_usd": float(r["health_exp_pc_usd"]),
                    "unemployment_youth": float(r["unemployment_youth"]),
                    "internet_pct": float(r["internet_pct"])
                }
                features_records.append({
                    "respondent_id": str(r["respondent_id"]),
                    "country": str(r["country"]),
                    "survey_year": int(r["survey_year"]),
                    "sex": int(r["sex"]),
                    "urban_rural": int(r["urban_rural"]),
                    "age_years": int(r["age_years"]),
                    "y1_ideation": int(r["y1_ideation"]),
                    "y2_plan": int(r["y2_plan"]),
                    "y3_attempt": int(r["y3_attempt"]),
                    "y4_loneliness_frequent": int(r["y4_loneliness_frequent"]),
                    "y5_worry_sleep_frequent": int(r["y5_worry_sleep_frequent"]),
                    "y1_distress": int(r["y1_distress"]),
                    "feat_json": json.dumps(feat_dict)
                })
                
            ins_sql = text("""
                INSERT INTO sp07.features (respondent_id, country, survey_year, sex, urban_rural, age_years,
                                           y1_ideation, y2_plan, y3_attempt, y4_loneliness_frequent,
                                           y5_worry_sleep_frequent, y1_distress, feat_json)
                VALUES (:respondent_id, :country, :survey_year, :sex, :urban_rural, :age_years,
                        :y1_ideation, :y2_plan, :y3_attempt, :y4_loneliness_frequent,
                        :y5_worry_sleep_frequent, :y1_distress, :feat_json)
            """)
            conn.execute(ins_sql, features_records)
            
        print("  ✓ Tabla sp07.features materializada exitosamente en PostgreSQL.")
    except Exception as e:
        print(f"  Aviso PostgreSQL en Fase 3: {e}")
        
    prev_y1 = df["y1_ideation"].mean()
    print(f"  ✓ Fase 3 completada: Y1_ideacion prevalencia = {prev_y1:.2%}, N = {len(df)}.")
    return df

if __name__ == "__main__":
    run_phase3_feature_engineering()
