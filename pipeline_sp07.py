# -*- coding: utf-8 -*-
"""
SP-07 · Orquestador del Pipeline de Entrenamiento por EDA en 7 Módulos

Ejecuta secuencialmente los 7 módulos independientes:
  - Módulo 1 (phase1_ingestion.py): Ingesta Multi-Fuente (OMS-GSHS + UNICEF + Banco Mundial)
  - Módulo 2 (phase2_quality_audit.py): Auditoría de Calidad y Trazabilidad de Cohorte
  - Módulo 3 (phase3_feature_engineering.py): Ingeniería de Variables y Derivación de Desenlaces
  - Módulo 4 (phase4_statistical_eda.py): Análisis Exploratorio de Datos (EDA) Inferencial
  - Módulo 5 (phase5_model_training.py): Motor de Entrenamiento Multimodelo y Multitarea
  - Módulo 6 (phase6_clinical_calibration.py): Calibración Clínica y Curvas de Decisión (Net Benefit)
  - Módulo 7 (phase7_statistical_inference.py): Contraste Formal de Hipótesis (H1-H6), Equidad y SHAP

Uso:
  python pipeline_sp07.py [--data data/gshs_adolescents_clean.csv] [--out artifacts] [--seed 42]
"""

import os
import sys
import time
import argparse

# Agregar directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from phase1_ingestion import run_phase1_ingestion
from phase2_quality_audit import run_phase2_quality_audit
from phase3_feature_engineering import run_phase3_feature_engineering
from phase4_statistical_eda import run_phase4_statistical_eda
from phase5_model_training import run_phase5_model_training
from phase6_clinical_calibration import run_phase6_clinical_calibration
from phase7_statistical_inference import run_phase7_statistical_inference

def run_7_phase_pipeline(data_path=None, data_dir="data", out_dir="artifacts", seed=42):
    t_start = time.time()
    print("=" * 80)
    print("  ORQUESTADOR DEL PIPELINE SP-07 DIVIDIDO EN 7 MÓDULOS INDEPENDIENTES")
    print("  Salud Mental Adolescente (OMS-GSHS + UNICEF Innocenti + Banco Mundial)")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # MÓDULO 1: Ingesta Multi-Fuente
    # -------------------------------------------------------------
    print("\n>>> EJECUTANDO MÓDULO 1: Ingesta Multi-Fuente...")
    if data_path is None or not os.path.exists(data_path):
        df_raw, df_context = run_phase1_ingestion(data_dir=data_dir, seed=seed)
    else:
        import pandas as pd
        df_raw = pd.read_csv(data_path)
        ctx_path = os.path.join(data_dir, "worldbank_country_context.csv")
        df_context = pd.read_csv(ctx_path) if os.path.exists(ctx_path) else None
        print(f"  ✓ Datos cargados desde archivo: {data_path} ({len(df_raw)} registros).")
        
    # -------------------------------------------------------------
    # MÓDULO 2: Auditoría de Calidad y Cohorte
    # -------------------------------------------------------------
    print("\n>>> EJECUTANDO MÓDULO 2: Auditoría de Calidad y Trazabilidad...")
    df_clean, attrition = run_phase2_quality_audit(df_raw=df_raw, df_context=df_context, data_dir=data_dir, out_dir=out_dir)
    
    # -------------------------------------------------------------
    # MÓDULO 3: Ingeniería de Variables y Desenlaces
    # -------------------------------------------------------------
    print("\n>>> EJECUTANDO MÓDULO 3: Ingeniería de Variables y Desenlaces...")
    df_features = run_phase3_feature_engineering(df=df_clean, data_dir=data_dir)
    
    # -------------------------------------------------------------
    # MÓDULO 4: EDA Estadístico Inferencial
    # -------------------------------------------------------------
    print("\n>>> EJECUTANDO MÓDULO 4: EDA Estadístico Inferencial...")
    df_features, eda_report = run_phase4_statistical_eda(df=df_features, data_dir=data_dir, out_dir=out_dir)
    
    # -------------------------------------------------------------
    # MÓDULO 5: Motor de Entrenamiento Multimodelo
    # -------------------------------------------------------------
    print("\n>>> EJECUTANDO MÓDULO 5: Motor de Entrenamiento Multimodelo...")
    oof_preds, trained_models = run_phase5_model_training(df=df_features, data_dir=data_dir, out_dir=out_dir, seed=seed)
    
    # -------------------------------------------------------------
    # MÓDULO 6: Calibración Clínica y Decisión
    # -------------------------------------------------------------
    print("\n>>> EJECUTANDO MÓDULO 6: Calibración Clínica y Curvas de Decisión...")
    leaderboard, best_model, best_single = run_phase6_clinical_calibration(
        df=df_features, oof_predictions=oof_preds, data_dir=data_dir, out_dir=out_dir
    )
    
    # -------------------------------------------------------------
    # MÓDULO 7: Hipótesis Científicas (H1-H6), Equidad y SHAP
    # -------------------------------------------------------------
    print("\n>>> EJECUTANDO MÓDULO 7: Hipótesis Científicas, Equidad y SHAP...")
    hypotheses, fairness = run_phase7_statistical_inference(df=df_features, data_dir=data_dir, out_dir=out_dir)
    
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  LOS 7 MÓDULOS SE EJECUTARON EXITOSAMENTE EN {t_elapsed:.2f} SEGUNDOS")
    print(f"  Mejor Modelo Detectado: {best_model} (PR-AUC: {leaderboard.iloc[0]['pr_auc']:.4f})")
    print(f"  Artefactos generados en: {os.path.abspath(out_dir)}")
    print("=" * 80)
    return leaderboard, hypotheses, eda_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline SP-07 en 7 Módulos")
    parser.add_argument("--data", default=None, help="Ruta al CSV opcional")
    parser.add_argument("--out", default="artifacts", help="Directorio de artefactos")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()
    
    run_7_phase_pipeline(data_path=args.data, out_dir=args.out, seed=args.seed)
