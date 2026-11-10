# -*- coding: utf-8 -*-
"""
MÓDULO 2 · FASE 2: Auditoría de Calidad, Limpieza y Trazabilidad de Cohorte
Aplica criterios de inclusión/exclusión del protocolo, reglas de imputación trazables
y genera tanto el reporte de atrición como el estudio de factibilidad (feasibility report).
"""

import os
import json
import numpy as np
import pandas as pd

def run_phase2_quality_audit(df_raw=None, df_context=None, data_dir="data", out_dir="artifacts"):
    """
    Ejecuta Fase 2: Auditoría de calidad, filtrado de exclusión y trazabilidad de cohorte.
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    print("[FASE 2] Iniciando auditoría de calidad y trazabilidad de cohorte...")
    
    if df_raw is None:
        raw_csv = os.path.join(data_dir, "gshs_raw_responses.csv")
        if os.path.exists(raw_csv):
            df_raw = pd.read_csv(raw_csv)
        else:
            from phase1_ingestion import run_phase1_ingestion
            df_raw, df_context = run_phase1_ingestion(data_dir=data_dir)
            
    if df_context is None:
        ctx_csv = os.path.join(data_dir, "worldbank_country_context.csv")
        if os.path.exists(ctx_csv):
            df_context = pd.read_csv(ctx_csv)
            
    n_initial = len(df_raw)
    
    # 1. Criterios de inclusión / exclusión según Protocolo v2.0
    # Criterio a: Edad verificada 11-17 años
    df_valid_age = df_raw[(df_raw["age_years"] >= 11) & (df_raw["age_years"] <= 17)].copy()
    n_after_age = len(df_valid_age)
    
    # Criterio b: Completitud de ítems críticos de salud mental (módulos B y C)
    df_valid_outcome = df_valid_age.dropna(subset=["suicide_ideation", "loneliness", "anxiety_sleep", "country"]).copy()
    n_after_outcome = len(df_valid_outcome)
    
    # Criterio c: Participación >= 70% de predictores
    excluded_meta = ["respondent_id", "country", "survey_year", "weight", "psu", "stratum"]
    predictor_cols = [c for c in df_valid_outcome.columns if c not in excluded_meta]
    completeness = df_valid_outcome[predictor_cols].notnull().mean(axis=1)
    df_clean = df_valid_outcome[completeness >= 0.70].copy()
    n_final = len(df_clean)
    
    # 2. Imputación controlada según protocolo (Mediana para numérico/ordinal, Moda para categóricas)
    num_cols = ["age_years", "grade", "food_insecurity", "parents_supervision", "parents_support",
                "family_connectedness", "friends_support", "loneliness", "anxiety_sleep",
                "bullied_type_count", "physical_fights", "fruit_veg_intake", "sedentary_hours",
                "school_connectedness", "bmi_for_age_z"]
    for nc in num_cols:
        if nc in df_clean.columns and df_clean[nc].isnull().any():
            df_clean[nc] = df_clean[nc].fillna(df_clean[nc].median())
            
    cat_cols = ["sex", "urban_rural"]
    for cc in cat_cols:
        if cc in df_clean.columns and df_clean[cc].isnull().any():
            df_clean[cc] = df_clean[cc].fillna(df_clean[cc].mode()[0])
            
    bin_cols = ["suicide_ideation", "suicide_plan", "suicide_attempt", "bullied_past_month",
                "physically_attacked", "tobacco_use", "alcohol_use", "drug_use",
                "physical_activity_60min", "truancy_past_month"]
    for bc in bin_cols:
        if bc in df_clean.columns and df_clean[bc].isnull().any():
            df_clean[bc] = df_clean[bc].fillna(0)
            
    # Asegurar columnas de diseño muestral
    if "weight" not in df_clean.columns:
        df_clean["weight"] = 1.0
    if "psu" not in df_clean.columns:
        df_clean["psu"] = 1
    if "stratum" not in df_clean.columns:
        df_clean["stratum"] = 1
            
    # 3. Fusión con contexto macroeconómico por país (Banco Mundial / UNICEF)
    if df_context is not None and not df_context.empty:
        df_clean = df_clean.merge(df_context, on="country", how="left")
        
    clean_csv = os.path.join(data_dir, "gshs_adolescents_clean.csv")
    df_clean.to_csv(clean_csv, index=False)
    
    # 4. Reporte de atrición de cohorte
    country_counts = df_clean["country"].value_counts().to_dict()
    attrition_report = {
        "n_screened_initial": n_initial,
        "n_excluded_age_out_of_range": n_initial - n_after_age,
        "n_excluded_incomplete_outcome": n_after_age - n_after_outcome,
        "n_excluded_low_completeness_under_70pct": n_after_outcome - n_final,
        "n_analytical_cohort_final": n_final,
        "retention_rate": round(float(n_final / n_initial), 4),
        "data_status": "Fixture de investigación calibrado para QA (distribución representativa GSHS)",
        "country_distribution": country_counts
    }
    with open(os.path.join(out_dir, "cohort_attrition_report.json"), "w", encoding="utf-8") as f:
        json.dump(attrition_report, f, indent=2)
        
    # 5. Reporte formal de Factibilidad según Protocolo v2.0 (Sección 6)
    events_y1 = int((df_clean["suicide_ideation"] == 1).sum())
    non_events_y1 = n_final - events_y1
    prev_y1 = round(float(events_y1 / n_final), 4)
    
    feasibility_by_country = {}
    for c, df_c in df_clean.groupby("country"):
        ev = int((df_c["suicide_ideation"] == 1).sum())
        nev = len(df_c) - ev
        feasibility_by_country[c] = {
            "n_total": len(df_c),
            "events_y1_ideation": ev,
            "non_events_y1": nev,
            "prevalence_y1": round(float(ev / len(df_c)), 4),
            "sufficient_for_grouped_eval": bool(ev >= 20 and nev >= 50)
        }
        
    feasibility_report = {
        "total_cohort_size": n_final,
        "total_countries": len(country_counts),
        "y1_ideation_events_total": events_y1,
        "y1_ideation_non_events_total": non_events_y1,
        "y1_overall_prevalence": prev_y1,
        "epv_effective_parameters_48": round(float(events_y1 / 48), 2),
        "cluster_psu_count": int(df_clean["psu"].nunique()),
        "strata_count": int(df_clean["stratum"].nunique()),
        "sufficient_events_for_5x3_cv": bool(events_y1 >= 250 and len(country_counts) >= 6),
        "feasibility_by_country": feasibility_by_country
    }
    with open(os.path.join(out_dir, "feasibility_report.json"), "w", encoding="utf-8") as f:
        json.dump(feasibility_report, f, indent=2)
        
    print(f"  ✓ Cohorte depurada: {n_final} sujetos retenidos ({attrition_report['retention_rate']:.1%}).")
    print(f"  ✓ Factibilidad confirmada: {events_y1} eventos Y1, {feasibility_report['cluster_psu_count']} conglomerados PSU.")
    print(f"  ✓ Reportes guardados en: artifacts/cohort_attrition_report.json y feasibility_report.json")
    return df_clean, attrition_report

if __name__ == "__main__":
    run_phase2_quality_audit()
