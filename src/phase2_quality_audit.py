# -*- coding: utf-8 -*-
"""
MÓDULO 2 · FASE 2: Auditoría de Calidad, Limpieza y Trazabilidad de Cohorte
Aplica criterios de inclusión/exclusión del protocolo, reglas de imputación y valida el cálculo de potencia muestral.
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
    
    # 1. Criterios de inclusión / exclusión
    # Criterio a: Edad 11-17 años
    df_valid_age = df_raw[(df_raw["age_years"] >= 11) & (df_raw["age_years"] <= 17)].copy()
    n_after_age = len(df_valid_age)
    
    # Criterio b: Completitud de ítems críticos de salud mental (módulos B y C)
    df_valid_outcome = df_valid_age.dropna(subset=["loneliness", "anxiety_sleep", "suicide_ideation", "country"]).copy()
    n_after_outcome = len(df_valid_outcome)
    
    # Criterio c: Participación >= 70% de predictores
    predictor_cols = [c for c in df_valid_outcome.columns if c not in ["respondent_id", "country", "survey_year"]]
    completeness = df_valid_outcome[predictor_cols].notnull().mean(axis=1)
    df_clean = df_valid_outcome[completeness >= 0.70].copy()
    n_final = len(df_clean)
    
    # 2. Imputación según protocolo (Mediana para ordinal/numérico, Moda para categóricas)
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
            
    # 3. Fusión con contexto macroeconómico por país
    if df_context is not None and not df_context.empty:
        df_clean = df_clean.merge(df_context, on="country", how="left")
        
    # 4. Verificación de cálculo de tamaño muestral (Protocolo Sección 5)
    # Requisito: >= 4,900 sujetos totales y >= 400 por país en 12 países
    country_counts = df_clean["country"].value_counts().to_dict()
    potency_check = {
        "target_total_sample": 4900,
        "observed_total_sample": n_final,
        "sample_potency_satisfied": bool(n_final >= 4900),
        "min_subjects_per_country_required": 400,
        "min_subjects_observed": int(min(country_counts.values())),
        "country_potency_satisfied": bool(all(cnt >= 400 for cnt in country_counts.values())),
        "total_countries": len(country_counts)
    }
    
    attrition_report = {
        "n_screened_initial": n_initial,
        "n_excluded_age_out_of_range": n_initial - n_after_age,
        "n_excluded_incomplete_outcome": n_after_age - n_after_outcome,
        "n_excluded_low_completeness_under_70pct": n_after_outcome - n_final,
        "n_analytical_cohort_final": n_final,
        "retention_rate": round(float(n_final / n_initial), 4),
        "potency_validation": potency_check,
        "country_distribution": country_counts
    }
    
    report_path = os.path.join(out_dir, "cohort_attrition_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(attrition_report, f, indent=2, ensure_ascii=False)
        
    clean_csv = os.path.join(data_dir, "gshs_adolescents_clean.csv")
    df_clean.to_csv(clean_csv, index=False)
    
    print(f"  ✓ Cohorte depurada: {n_final} sujetos retenidos ({attrition_report['retention_rate']:.1%}).")
    print(f"  ✓ Potencia muestral confirmada: {potency_check['sample_potency_satisfied']} (mínimo por país: {potency_check['min_subjects_observed']}).")
    print(f"  ✓ Reporte de trazabilidad guardado en: {report_path}")
    return df_clean, attrition_report

if __name__ == "__main__":
    run_phase2_quality_audit()

