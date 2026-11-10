# -*- coding: utf-8 -*-
"""
MÓDULO 6 · FASE 6: Calibración Clínica, Umbrales Operativos y Curvas de Decisión
Ajusta probabilidades a la escala de riesgo (pendiente e intercepto de Cox, calibración Platt),
calcula métricas integrales (PR-AUC principal, ROC-AUC, Brier, Sensibilidad, Especificidad, PPV, NPV, Balanced Acc),
calcula umbrales operativos (Youden vs Costo 3:1), genera análisis de Beneficio Neto (DCA)
y persiste el leaderboard en PostgreSQL (sp07.metrics).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_curve, balanced_accuracy_score)
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
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float(f1_score(y_true, pred))
    bal_acc = float(balanced_accuracy_score(y_true, pred))
    brier = float(brier_score_loss(y_true, probas))
    slope, intercept = calculate_cox_calibration(y_true, probas)
    
    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "brier": round(brier, 4),
        "calib_slope": slope,
        "calib_intercept": intercept,
        "threshold": round(threshold, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
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

def compute_calibration_bins(y_true, probas, n_bins=10):
    """Calcula la calibración observada vs estimada por deciles."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    obs_rates = []
    counts = []
    
    for i in range(n_bins):
        low, high = bins[i], bins[i+1]
        mask = (probas >= low) & (probas < high) if i < n_bins - 1 else (probas >= low) & (probas <= high)
        cnt = np.sum(mask)
        counts.append(int(cnt))
        if cnt > 0:
            obs_rates.append(round(float(np.mean(y_true[mask])), 4))
            bin_centers.append(round(float(np.mean(probas[mask])), 4))
        else:
            obs_rates.append(0.0)
            bin_centers.append(round(float((low + high) / 2), 4))
            
    return {
        "bin_centers": bin_centers,
        "observed_rates": obs_rates,
        "bin_counts": counts
    }

def run_phase6_clinical_calibration(df=None, oof_predictions=None, data_dir="data", out_dir="artifacts"):
    """
    Ejecuta Fase 6: Calibración y generación del Leaderboard multicriterio.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*75)
    print("[FASE 6] CALIBRACIÓN CLÍNICA, UMBRALES Y BENEFICIO NETO (DCA)")
    print("="*75)
    
    if df is None:
        feat_csv = os.path.join(data_dir, "gshs_features_engineered.csv")
        df = pd.read_csv(feat_csv)
        
    y_true = df["y1_ideation"].astype(int).values if "y1_ideation" in df.columns else (df["suicide_ideation"] == 1).astype(int).values
    
    if oof_predictions is None:
        p_path = os.path.join(out_dir, "oof_predictions.pkl")
        if os.path.exists(p_path):
            oof_data = joblib.load(p_path)
            oof_predictions = oof_data.get("full_preds", oof_data)
        else:
            from phase5_model_training import run_phase5_model_training
            oof_predictions, _ = run_phase5_model_training(df=df, data_dir=data_dir, out_dir=out_dir)
            
    if isinstance(oof_predictions, dict) and "full_preds" in oof_predictions:
        oof_predictions = oof_predictions["full_preds"]
        
    leaderboard_rows = []
    dca_all = {}
    calib_curves_all = {}
    
    for model_name, probas in oof_predictions.items():
        youden_th, cost_th = compute_thresholds(y_true, probas)
        metrics = evaluate_predictions(y_true, probas, threshold=youden_th)
        
        row = {
            "model_type": model_name,
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "ppv": metrics["ppv"],
            "npv": metrics["npv"],
            "f1": metrics["f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "brier": metrics["brier"],
            "calib_slope": metrics["calib_slope"],
            "calib_intercept": metrics["calib_intercept"],
            "threshold": metrics["threshold"],
            "cost_threshold_3to1": cost_th
        }
        leaderboard_rows.append(row)
        dca_all[model_name] = compute_decision_curve(y_true, probas)
        calib_curves_all[model_name] = compute_calibration_bins(y_true, probas)
        
    # Ordenar por PR-AUC descendente (métrica principal según protocolo)
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    leaderboard_path = os.path.join(out_dir, "leaderboard_oof.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    
    dca_path = os.path.join(out_dir, "decision_curve_analysis.json")
    with open(dca_path, "w", encoding="utf-8") as f:
        json.dump({"dca": dca_all, "calibration_curves": calib_curves_all}, f, indent=2)
        
    best_model_name = leaderboard.iloc[0]["model_type"]
    best_single_name = leaderboard[~leaderboard.model_type.str.contains("ensemble")].iloc[0]["model_type"]
    
    # Persistencia en PostgreSQL sp07.metrics
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE sp07.metrics CASCADE;"))
            for _, r in leaderboard.iterrows():
                conn.execute(text("""
                    INSERT INTO sp07.metrics (model_name, split_type, roc_auc, pr_auc, sensitivity,
                                              specificity, f1_score, brier_score, calib_slope,
                                              calib_intercept, threshold, evaluated_at)
                    VALUES (:model_type, 'oof_grouped_country', :roc_auc, :pr_auc, :sensitivity,
                            :specificity, :f1, :brier, :calib_slope, :calib_intercept, :threshold, now());
                """), r.to_dict())
        print("  ✓ Métricas de calibración guardadas en PostgreSQL (sp07.metrics).")
    except Exception as e:
        print(f"  Aviso PostgreSQL en Fase 6: {e}")
        
    print(f"  ✓ Leaderboard consolidado: Mejor modelo = {best_model_name} (PR-AUC = {leaderboard.iloc[0]['pr_auc']:.4f}).")
    print(f"  ✓ Curva de decisión (DCA) guardada en: {dca_path}")
    return leaderboard, best_model_name, best_single_name

if __name__ == "__main__":
    run_phase6_clinical_calibration()
