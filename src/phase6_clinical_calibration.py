# -*- coding: utf-8 -*-
"""
MÓDULO 6 · FASE 6: Calibración Clínica, Umbrales Operativos y Curvas de Decisión
Ajusta probabilidades a la escala de riesgo (pendiente e intercepto de Cox),
calcula umbrales de cribado (Youden vs Costo 3:1), genera análisis de Beneficio Neto (DCA)
y persiste el leaderboard en PostgreSQL (sp07.metrics).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sp07:sp07_dev_password@localhost:5432/sp07_db")

def calculate_cox_calibration(y_true, probas):
    """
    Calcula pendiente e intercepto de Cox mediante regresión logística de las log-odds.
    """
    eps = 1e-6
    p_clipped = np.clip(probas, eps, 1 - eps)
    logit_p = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(logit_p, y_true)
    slope = float(lr.coef_[0][0])
    intercept = float(-lr.intercept_[0] / lr.coef_[0][0]) if abs(lr.coef_[0][0]) > 1e-4 else float(lr.intercept_[0])
    return round(slope, 3), round(intercept, 3)

def compute_thresholds(y_true, probas):
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    youden_idx = np.argmax(tpr - fpr)
    youden_thresh = float(thresholds[youden_idx])
    
    best_cost = float("inf")
    best_screening_thresh = 0.20
    for th in np.linspace(0.05, 0.60, 56):
        pred = (probas >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = 3 * fn + fp
        if cost < best_cost:
            best_cost = cost
            best_screening_thresh = float(th)
            
    return round(youden_thresh, 4), round(best_screening_thresh, 4)

def evaluate_predictions(y_true, probas, threshold=None):
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, probas)
        threshold = float(thresholds[np.argmax(tpr - fpr)])
        
    pred = (probas >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    
    roc_auc = float(roc_auc_score(y_true, probas))
    pr_auc = float(average_precision_score(y_true, probas))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, pred))
    brier = float(brier_score_loss(y_true, probas))
    slope, intercept = calculate_cox_calibration(y_true, probas)
    
    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "brier": round(brier, 4),
        "calib_slope": slope,
        "calib_intercept": intercept,
        "threshold": round(threshold, 4)
    }

def compute_decision_curve(y_true, probas, thresholds=np.linspace(0.05, 0.40, 36)):
    """
    Curva de decisión clínica (Net Benefit - Decision Curve Analysis) para cribado poblacional.
    """
    n = len(y_true)
    prevalence = np.mean(y_true)
    results = []
    
    for pt in thresholds:
        pred = (probas >= pt).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        
        nb_model = (tp / n) - (fp / n) * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        
        results.append({
            "threshold_prob": round(float(pt), 4),
            "net_benefit_model": round(float(nb_model), 4),
            "net_benefit_treat_all": round(float(nb_all), 4),
            "net_benefit_treat_none": 0.0
        })
    return results

def run_phase6_clinical_calibration(df=None, oof_predictions=None, data_dir="data", out_dir="artifacts"):
    """
    Ejecuta Fase 6: Calibración, umbrales y evaluación de decisiones clínicas.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("\n[FASE 6] Iniciando calibración clínica, umbrales y curvas de decisión...")
    
    if df is None:
        feat_csv = os.path.join(data_dir, "gshs_features_engineered.csv")
        if os.path.exists(feat_csv):
            df = pd.read_csv(feat_csv)
            
    if oof_predictions is None:
        oof_path = os.path.join(out_dir, "oof_predictions.pkl")
        if os.path.exists(oof_path):
            oof_predictions = joblib.load(oof_path)
        else:
            from phase5_model_training import run_phase5_model_training
            oof_predictions, _ = run_phase5_model_training(df=df, data_dir=data_dir, out_dir=out_dir)
            
    y_true = df["y1_distress"].astype(int).values
    
    metrics_list = []
    for model_name, probas in oof_predictions.items():
        m = evaluate_predictions(y_true, probas)
        m["model_type"] = model_name
        m["fold"] = "OOF_Overall"
        metrics_list.append(m)
        
    df_leaderboard = pd.DataFrame(metrics_list).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    lb_csv = os.path.join(out_dir, "leaderboard_oof.csv")
    df_leaderboard.to_csv(lb_csv, index=False)
    
    best_model_name = df_leaderboard.iloc[0]["model_type"]
    best_single_name = df_leaderboard[df_leaderboard.model_type != "ensemble_stacking"].iloc[0]["model_type"]
    
    # Análisis de curva de decisión sobre el mejor modelo
    dca_data = compute_decision_curve(y_true, oof_predictions[best_model_name])
    dca_json = os.path.join(out_dir, "decision_curve_analysis.json")
    with open(dca_json, "w", encoding="utf-8") as f:
        json.dump(dca_data, f, indent=2)
        
    # Persistir en PostgreSQL sp07.metrics
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE sp07.metrics CASCADE;"))
            m_res = conn.execute(text("SELECT model_id, model_type FROM sp07.model_registry;")).fetchall()
            mid_map = {r[1]: r[0] for r in m_res}
            
            for _, r in df_leaderboard.iterrows():
                mid = mid_map.get(r["model_type"])
                if mid:
                    conn.execute(text("""
                        INSERT INTO sp07.metrics (model_id, fold, roc_auc, pr_auc, sensitivity, specificity, f1, brier, calib_slope, calib_intercept, threshold)
                        VALUES (:mid, :fold, :roc_auc, :pr_auc, :sens, :spec, :f1, :brier, :slope, :intercept, :thr)
                        ON CONFLICT (model_id, fold) DO NOTHING;
                    """), {
                        "mid": mid, "fold": r["fold"], "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                        "sens": r["sensitivity"], "spec": r["specificity"], "f1": r["f1"],
                        "brier": r["brier"], "slope": r["calib_slope"], "intercept": r["calib_intercept"],
                        "thr": r["threshold"]
                    })
        print("  ✓ Métricas de calibración guardadas en PostgreSQL (sp07.metrics).")
    except Exception as e:
        print(f"  Aviso PostgreSQL en Fase 6: {e}")
        
    print(f"  ✓ Leaderboard consolidado: Mejor modelo = {best_model_name} (PR-AUC = {df_leaderboard.iloc[0]['pr_auc']:.4f}).")
    print(f"  ✓ Curva de decisión (DCA) guardada en: {dca_json}")
    return df_leaderboard, best_model_name, best_single_name

if __name__ == "__main__":
    run_phase6_clinical_calibration()

