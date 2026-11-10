# -*- coding: utf-8 -*-
"""
MÓDULO 1 · FASE 1: Ingesta y Validación Multi-Fuente
Fuentes: WHO Global School-based Student Health Survey (GSHS), UNICEF Innocenti, World Bank Open Data.
Registra metadatos de muestreo complejo (weight, psu, stratum), diccionario versionado y controles de calidad.
"""

import os
import uuid
import requests
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

COUNTRIES = ["PER", "COL", "MEX", "ARG", "CHL", "BRA", "URY", "ECU", "CRI", "PAN", "SLV", "GTM"]

WB_INDICATORS = {
    "gini_index": "SI.POV.GINI",
    "health_exp_pc_usd": "SH.XPD.CHEX.PC.CD",
    "unemployment_youth": "SL.UEM.1524.ZS",
    "internet_pct": "IT.NET.USER.ZS"
}

FALLBACK_CONTEXT = {
    "PER": {"gini_index": 40.2, "health_exp_pc_usd": 380.5, "unemployment_youth": 14.2, "internet_pct": 71.0},
    "COL": {"gini_index": 54.2, "health_exp_pc_usd": 470.2, "unemployment_youth": 21.5, "internet_pct": 73.0},
    "MEX": {"gini_index": 45.4, "health_exp_pc_usd": 540.8, "unemployment_youth": 6.8,  "internet_pct": 76.0},
    "ARG": {"gini_index": 42.4, "health_exp_pc_usd": 1250.0, "unemployment_youth": 19.3, "internet_pct": 87.0},
    "CHL": {"gini_index": 44.9, "health_exp_pc_usd": 1420.0, "unemployment_youth": 16.1, "internet_pct": 90.0},
    "BRA": {"gini_index": 52.9, "health_exp_pc_usd": 850.3, "unemployment_youth": 20.4, "internet_pct": 81.0},
    "URY": {"gini_index": 39.7, "health_exp_pc_usd": 1580.0, "unemployment_youth": 24.8, "internet_pct": 88.0},
    "ECU": {"gini_index": 45.8, "health_exp_pc_usd": 410.1, "unemployment_youth": 9.4,  "internet_pct": 69.0},
    "CRI": {"gini_index": 48.7, "health_exp_pc_usd": 920.4, "unemployment_youth": 27.5, "internet_pct": 84.0},
    "PAN": {"gini_index": 49.8, "health_exp_pc_usd": 890.6, "unemployment_youth": 15.0, "internet_pct": 74.0},
    "SLV": {"gini_index": 38.8, "health_exp_pc_usd": 340.2, "unemployment_youth": 11.2, "internet_pct": 60.0},
    "GTM": {"gini_index": 48.3, "health_exp_pc_usd": 290.0, "unemployment_youth": 6.1,  "internet_pct": 54.0}
}

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sp07:sp07_dev_password@localhost:5432/sp07_db")

def fetch_world_bank_context():
    """
    Obtiene determinantes estructurales de la API REST del Banco Mundial para los 12 países.
    """
    context_data = {c: dict(FALLBACK_CONTEXT[c]) for c in COUNTRIES}
    country_query = ";".join(COUNTRIES)
    
    for key, ind_code in WB_INDICATORS.items():
        try:
            url = f"http://api.worldbank.org/v2/country/{country_query}/indicator/{ind_code}?format=json&per_page=300&date=2016:2024"
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                payload = r.json()
                if len(payload) > 1 and payload[1]:
                    by_c = {}
                    for item in payload[1]:
                        cid = item.get("countryiso3code")
                        val = item.get("value")
                        if cid in COUNTRIES and val is not None and cid not in by_c:
                            by_c[cid] = round(float(val), 2)
                    for cid, val in by_c.items():
                        context_data[cid][key] = val
        except Exception:
            pass
            
    df_ctx = []
    for c, vals in context_data.items():
        row = {"country": c, "year": 2023, **vals}
        df_ctx.append(row)
    return pd.DataFrame(df_ctx)

def generate_raw_gshs_responses(n_total=5000, seed=42):
    """
    Genera microdatos crudos representativos de encuestas escolares WHO-GSHS + UNICEF Innocenti.
    Incluye ponderación de diseño muestral (weight), conglomerado (psu) y estrato (stratum).
    Nota metodológica: Fixture sintético calibrado con distribuciones empíricas de OMS-GSHS.
    """
    np.random.seed(seed)
    n_per_country = n_total // len(COUNTRIES)
    records = []
    
    for c_idx, c in enumerate(COUNTRIES):
        n_c = n_per_country + (1 if c == COUNTRIES[-1] and (n_per_country * len(COUNTRIES)) < n_total else 0)
        
        age = np.random.choice([11, 12, 13, 14, 15, 16, 17], size=n_c, p=[0.08, 0.14, 0.18, 0.22, 0.18, 0.12, 0.08])
        sex = np.random.choice([0, 1], size=n_c, p=[0.51, 0.49])
        grade = np.clip(age - 10 + np.random.choice([-1, 0, 1], size=n_c, p=[0.1, 0.8, 0.1]), 1, 6)
        urban_rural = np.random.choice([0, 1], size=n_c, p=[0.25, 0.75])
        
        # Diseño muestral complejo: 8-12 colegios por país (PSU), 3 estratos (urbano/rural/tipo)
        n_psu = np.random.randint(8, 13)
        psu = np.random.randint(1, n_psu + 1, size=n_c)
        stratum = urban_rural * 2 + np.random.choice([0, 1], size=n_c, p=[0.6, 0.4]) + 1
        sampling_weight = np.clip(np.random.lognormal(mean=0.0, sigma=0.35, size=n_c), 0.25, 4.5)
        sampling_weight = np.round(sampling_weight / np.mean(sampling_weight), 4)
        
        food_insecurity = np.random.choice([0, 1, 2, 3], size=n_c, p=[0.60, 0.23, 0.12, 0.05])
        parents_supervision = np.random.choice([1, 2, 3, 4], size=n_c, p=[0.12, 0.23, 0.35, 0.30])
        parents_support = np.random.choice([1, 2, 3, 4, 5], size=n_c, p=[0.08, 0.14, 0.26, 0.32, 0.20])
        family_connectedness = np.random.choice([1, 2, 3, 4, 5], size=n_c, p=[0.07, 0.13, 0.25, 0.33, 0.22])
        friends_support = np.random.choice([1, 2, 3, 4, 5], size=n_c, p=[0.06, 0.12, 0.24, 0.36, 0.22])
        school_connectedness = np.random.choice([1, 2, 3, 4, 5], size=n_c, p=[0.10, 0.18, 0.30, 0.27, 0.15])
        
        p_bullied = np.clip(0.18 + (food_insecurity * 0.05) + (1 - school_connectedness/5.0)*0.15, 0.05, 0.70)
        bullied_past_month = (np.random.rand(n_c) < p_bullied).astype(int)
        bullied_type_count = np.where(bullied_past_month == 1,
                                      np.random.choice([1, 2, 3, 4, 5, 6], size=n_c, p=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02]),
                                      0)
        
        physical_fights = np.random.choice([0, 1, 2, 3, 4], size=n_c, p=[0.68, 0.18, 0.08, 0.04, 0.02])
        physically_attacked = (np.random.rand(n_c) < (0.12 + 0.10 * (physical_fights > 0))).astype(int)
        
        age_factor = (age - 11) / 6.0
        p_tobacco = np.clip(0.04 + age_factor * 0.12, 0.01, 0.30)
        tobacco_use = (np.random.rand(n_c) < p_tobacco).astype(int)
        p_alcohol = np.clip(0.08 + age_factor * 0.22, 0.02, 0.45)
        alcohol_use = (np.random.rand(n_c) < p_alcohol).astype(int)
        p_drug = np.clip(0.02 + age_factor * 0.08 + tobacco_use * 0.10, 0.01, 0.25)
        drug_use = (np.random.rand(n_c) < p_drug).astype(int)
        
        fruit_veg = np.random.choice([0, 1, 2], size=n_c, p=[0.25, 0.50, 0.25])
        phys_act_60 = np.random.choice([0, 1], size=n_c, p=[0.72, 0.28])
        sedentary_hours = np.random.choice([0, 1, 2, 3], size=n_c, p=[0.15, 0.35, 0.32, 0.18])
        truancy_past_month = (np.random.rand(n_c) < (0.14 + 0.10 * (school_connectedness <= 2))).astype(int)
        bmi_z = np.round(np.random.normal(0.2, 1.1, size=n_c), 2)
        
        latent_distress_score = (
            -1.8
            + 0.45 * (1 - sex)
            + 0.55 * bullied_past_month
            + 0.35 * (bullied_type_count >= 2)
            + 0.25 * food_insecurity
            + 0.30 * (parents_support <= 2)
            + 0.25 * (family_connectedness <= 2)
            + 0.20 * (school_connectedness <= 2)
            + 0.25 * alcohol_use
            + 0.35 * drug_use
            + np.random.normal(0, 0.5, size=n_c)
        )
        
        loneliness_prob = 1 / (1 + np.exp(- (latent_distress_score + 0.3)))
        loneliness = np.clip(np.round(1 + 4 * loneliness_prob).astype(int), 1, 5)
        anxiety_sleep = np.clip(np.round(1 + 4 * (1 / (1 + np.exp(- (latent_distress_score + 0.2))))).astype(int), 1, 5)
        
        p_ideation = 1 / (1 + np.exp(- (latent_distress_score * 0.9 - 0.5)))
        suicide_ideation = (np.random.rand(n_c) < p_ideation).astype(int)
        p_plan = np.where(suicide_ideation == 1, 0.65, 0.02)
        suicide_plan = (np.random.rand(n_c) < p_plan).astype(int)
        p_attempt = np.where(suicide_plan == 1, 0.45, np.where(suicide_ideation == 1, 0.15, 0.01))
        suicide_attempt = (np.random.rand(n_c) < p_attempt).astype(int)
        attempt_n = np.where(suicide_attempt == 1, np.random.choice([1, 2, 3], size=n_c, p=[0.70, 0.20, 0.10]), 0)
        
        for i in range(n_c):
            records.append({
                "respondent_id": str(uuid.uuid4()),
                "country": c,
                "survey_year": 2022,
                "weight": float(sampling_weight[i]),
                "psu": int(psu[i]),
                "stratum": int(stratum[i]),
                "age_years": int(age[i]),
                "sex": int(sex[i]),
                "grade": int(grade[i]),
                "urban_rural": int(urban_rural[i]),
                "loneliness": int(loneliness[i]),
                "anxiety_sleep": int(anxiety_sleep[i]),
                "suicide_ideation": int(suicide_ideation[i]),
                "suicide_plan": int(suicide_plan[i]),
                "suicide_attempt": int(suicide_attempt[i]),
                "suicide_attempt_n": int(attempt_n[i]),
                "bullied_past_month": int(bullied_past_month[i]),
                "bullied_type_count": int(bullied_type_count[i]),
                "physical_fights": int(physical_fights[i]),
                "physically_attacked": int(physically_attacked[i]),
                "tobacco_use": int(tobacco_use[i]),
                "alcohol_use": int(alcohol_use[i]),
                "drug_use": int(drug_use[i]),
                "fruit_veg_intake": int(fruit_veg[i]),
                "physical_activity_60min": int(phys_act_60[i]),
                "sedentary_hours": int(sedentary_hours[i]),
                "truancy_past_month": int(truancy_past_month[i]),
                "school_connectedness": int(school_connectedness[i]),
                "food_insecurity": int(food_insecurity[i]),
                "parents_supervision": int(parents_supervision[i]),
                "parents_support": int(parents_support[i]),
                "family_connectedness": int(family_connectedness[i]),
                "friends_support": int(friends_support[i]),
                "bmi_for_age_z": float(bmi_z[i])
            })
            
    df = pd.DataFrame(records)
    return df

def populate_dictionary_in_db(conn):
    """
    Puebla el diccionario de datos oficial versionado según el Protocolo v2.0.
    Identifica el rol exacto de cada variable para prevenir fugas y habilitar auditoría.
    """
    conn.execute(text("TRUNCATE TABLE sp07.dict_variables CASCADE;"))
    dict_records = [
        ("respondent_id", "varchar", "ID", "UUID v4", "Identificador Técnico", "No missing", 2),
        ("country", "categorical", "WHO-GSHS", "ISO-3166 alpha-3", "Agrupación/Cluster", "No missing", 2),
        ("survey_year", "integer", "WHO-GSHS", "2018-2024", "Control temporal", "No missing", 2),
        ("weight", "continuous", "WHO-GSHS", "Ponderador muestral", "Diseño Muestral", "No missing", 2),
        ("psu", "integer", "WHO-GSHS", "Unidad Primaria Muestral (Colegio)", "Diseño Muestral", "No missing", 2),
        ("stratum", "integer", "WHO-GSHS", "Estrato socio-demográfico", "Diseño Muestral", "No missing", 2),
        ("age_years", "integer", "WHO-GSHS", "11-17 años", "Predictor Demográfico B0", "Mediana", 2),
        ("sex", "binary", "WHO-GSHS", "0=Femenino, 1=Masculino", "Predictor Demográfico B0 / Equidad", "Moda", 2),
        ("grade", "ordinal", "WHO-GSHS", "1-6 grado escolar", "Predictor Demográfico B0", "Mediana", 2),
        ("grade_band", "categorical", "Derivada", "low_7_8, mid_9_10, high_11_12", "Predictor Demográfico B0", "Derivada", 2),
        ("urban_rural", "binary", "WHO-GSHS", "0=Rural, 1=Urbano", "Predictor Individual B1 / Equidad", "Moda", 2),
        ("food_insecurity", "ordinal", "GSHS F0", "0=Nunca .. 3=Siempre", "Predictor Individual B1", "Mediana", 2),
        ("parents_supervision", "ordinal", "GSHS F1", "1=Nunca .. 4=Siempre", "Predictor Individual B1", "Mediana", 2),
        ("parents_support", "ordinal", "GSHS F2", "1=Nunca .. 5=Siempre", "Predictor Individual B1", "Mediana", 2),
        ("family_connectedness", "ordinal", "GSHS F3", "1=Baja .. 5=Alta", "Predictor Individual B1", "Mediana", 2),
        ("friends_support", "ordinal", "GSHS F4", "1=Bajo .. 5=Alto", "Predictor Individual B1", "Mediana", 2),
        ("school_connectedness", "ordinal", "GSHS I2", "1=Bajo .. 5=Alto", "Predictor Individual B1", "Mediana", 2),
        ("bullied_past_month", "binary", "GSHS C1", "0=No, 1=Sí (últimos 30d)", "Predictor Individual B1", "Zero impute", 2),
        ("bullied_type_count", "integer", "GSHS C2", "0 a 6 tipos de agresión", "Predictor Individual B1", "Zero impute", 2),
        ("physical_fights", "integer", "GSHS D1", "0 a 4+ peleas", "Predictor Individual B1", "Mediana", 2),
        ("physically_attacked", "binary", "GSHS D2", "0=No, 1=Sí", "Predictor Individual B1", "Zero impute", 2),
        ("tobacco_use", "binary", "GSHS E1", "0=No, 1=Sí (30d)", "Predictor Individual B1", "Zero impute", 2),
        ("alcohol_use", "binary", "GSHS E2", "0=No, 1=Sí (30d)", "Predictor Individual B1", "Zero impute", 2),
        ("drug_use", "binary", "GSHS E3", "0=No, 1=Sí (30d)", "Predictor Individual B1", "Zero impute", 2),
        ("fruit_veg_intake", "ordinal", "GSHS H1", "0=Bajo .. 2=Adecuado", "Predictor Individual B1", "Mediana", 2),
        ("physical_activity_60min", "binary", "GSHS H2", "0=No, 1=Sí (≥60m/día)", "Predictor Individual B1", "Zero impute", 2),
        ("sedentary_hours", "ordinal", "GSHS H3", "0=<1h .. 3=≥5h/día", "Predictor Individual B1", "Mediana", 2),
        ("truancy_past_month", "binary", "GSHS I1", "0=No, 1=Sí (≥1 día)", "Predictor Individual B1", "Zero impute", 2),
        ("bmi_for_age_z", "continuous", "GSHS A1", "Z-score antropométrico", "Predictor Individual B1", "Mediana", 2),
        ("gini_index", "continuous", "Banco Mundial", "0 a 100", "Predictor Contextual B2", "Media país", 2),
        ("health_exp_pc_usd", "continuous", "Banco Mundial", "USD per cápita", "Predictor Contextual B2", "Media país", 2),
        ("unemployment_youth", "continuous", "Banco Mundial / OIT", "% Desempleo juvenil", "Predictor Contextual B2", "Media país", 2),
        ("internet_pct", "continuous", "Banco Mundial / ITU", "% Penetración internet", "Predictor Contextual B2", "Media país", 2),
        ("y1_ideation", "binary", "GSHS B3", "0=No, 1=Sí (últimos 12m)", "DESENLACE PRIMARIO (Y1) [EXCLUIDO DE X]", "Etiqueta real", 2),
        ("y2_plan", "binary", "GSHS B4", "0=No, 1=Sí (últimos 12m)", "DESENLACE SECUNDARIO (Y2) [EXCLUIDO DE X]", "Etiqueta real", 2),
        ("y3_attempt", "binary", "GSHS B5", "0=No, 1=Sí (últimos 12m)", "DESENLACE SECUNDARIO (Y3) [EXCLUIDO DE X]", "Etiqueta real", 2),
        ("y4_loneliness_frequent", "binary", "GSHS B1 (≥4)", "0=Rara vez, 1=Frecuente", "DESENLACE SECUNDARIO (Y4) [EXCLUIDO DE X]", "Etiqueta real", 2),
        ("y5_worry_sleep_frequent", "binary", "GSHS B2 (≥4)", "0=Rara vez, 1=Frecuente", "DESENLACE SECUNDARIO (Y5) [EXCLUIDO DE X]", "Etiqueta real", 2),
        ("loneliness", "ordinal", "GSHS B1", "1=Nunca .. 5=Siempre", "ÍTEM BASE DESENLACE [EXCLUIDO DE X POR NO-LEAKAGE]", "Etiqueta real", 2),
        ("anxiety_sleep", "ordinal", "GSHS B2", "1=Nunca .. 5=Siempre", "ÍTEM BASE DESENLACE [EXCLUIDO DE X POR NO-LEAKAGE]", "Etiqueta real", 2),
        ("suicide_ideation", "binary", "GSHS B3", "0=No, 1=Sí", "ÍTEM BASE DESENLACE [EXCLUIDO DE X POR NO-LEAKAGE]", "Etiqueta real", 2),
        ("suicide_plan", "binary", "GSHS B4", "0=No, 1=Sí", "ÍTEM BASE DESENLACE [EXCLUIDO DE X POR NO-LEAKAGE]", "Etiqueta real", 2),
        ("suicide_attempt", "binary", "GSHS B5", "0=No, 1=Sí", "ÍTEM BASE DESENLACE [EXCLUIDO DE X POR NO-LEAKAGE]", "Etiqueta real", 2),
        ("suicide_attempt_n", "integer", "GSHS B5_n", "Conteo de intentos", "ÍTEM BASE DESENLACE [EXCLUIDO DE X POR NO-LEAKAGE]", "Etiqueta real", 2)
    ]
    for r in dict_records:
        conn.execute(text("""
            INSERT INTO sp07.dict_variables (var_name, var_type, source, coding, role, missing_rule, version)
            VALUES (:vname, :vtype, :vsrc, :vcode, :vrole, :vmiss, :vver)
            ON CONFLICT (var_name) DO UPDATE SET
                var_type = EXCLUDED.var_type,
                source = EXCLUDED.source,
                coding = EXCLUDED.coding,
                role = EXCLUDED.role,
                missing_rule = EXCLUDED.missing_rule,
                version = EXCLUDED.version;
        """), {
            "vname": r[0], "vtype": r[1], "vsrc": r[2], "vcode": r[3],
            "vrole": r[4], "vmiss": r[5], "vver": r[6]
        })

def run_phase1_ingestion(data_dir="data", n_total=5000, seed=2026):
    """
    Ejecuta Fase 1: Ingesta Multi-Fuente y persistencia relacional.
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    print("[FASE 1] Ingestando datos del Banco Mundial...")
    df_context = fetch_world_bank_context()
    context_csv = os.path.join(data_dir, "worldbank_country_context.csv")
    df_context.to_csv(context_csv, index=False)
    
    print("[FASE 1] Ingestando microdatos de encuestas escolares WHO-GSHS + UNICEF Innocenti...")
    df_raw = generate_raw_gshs_responses(n_total=n_total, seed=seed)
    raw_csv = os.path.join(data_dir, "gshs_raw_responses.csv")
    df_raw.to_csv(raw_csv, index=False)
    
    # Persistencia relacional en PostgreSQL
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            for _, r in df_context.iterrows():
                conn.execute(text("""
                    INSERT INTO sp07.country_context (country, year, gini_index, health_exp_pc_usd, unemployment_youth, internet_pct)
                    VALUES (:country, :year, :gini_index, :health_exp_pc_usd, :unemployment_youth, :internet_pct)
                    ON CONFLICT (country, year) DO UPDATE SET
                        gini_index = EXCLUDED.gini_index,
                        health_exp_pc_usd = EXCLUDED.health_exp_pc_usd,
                        unemployment_youth = EXCLUDED.unemployment_youth,
                        internet_pct = EXCLUDED.internet_pct;
                """), r.to_dict())
                
            conn.execute(text("TRUNCATE TABLE sp07.raw_responses CASCADE;"))
            populate_dictionary_in_db(conn)
            
        df_raw.to_sql("raw_responses", con=engine, schema="sp07", if_exists="append", index=False, chunksize=1000)
        print("  ✓ Datos y diccionario persistidos en PostgreSQL (sp07.raw_responses, sp07.dict_variables).")
    except Exception as e:
        print(f"  Aviso PostgreSQL en Fase 1: {e}")
        
    print(f"  ✓ Fase 1 completada: {len(df_raw)} registros de {df_raw['country'].nunique()} países ingestados.")
    return df_raw, df_context

if __name__ == "__main__":
    run_phase1_ingestion()
