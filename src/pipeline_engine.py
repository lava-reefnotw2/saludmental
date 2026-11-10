# -*- coding: utf-8 -*-
"""
SP-07 · Fase 5 y 6: Motor de Entrenamiento Multimodelo, Multitarea y Calibración Clínica
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Agregar directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_curve)

from models import MultiTaskMLPWrapper, WeightedEnsembleClassifier, build_model_factories

warnings.filterwarnings("ignore")

TARGET_PRIMARY = "y1_distress"
TARGETS_MULTI = ["y1_distress", "y2_ideation", "y3_plan", "y4_attempt"]

DEMO_CAT = ["country", "sex", "grade_band"]
DEMO_NUM = ["age_years"]

CONTEXT_CAT = ["urban_rural"]
CONTEXT_NUM = [
    "food_insecurity", "parents_supervision", "parents_support",
    "family_connectedness", "friends_support", "loneliness", "anxiety_sleep",
    "bullied_past_month", "bullied_type_count", "physical_fights",
    "tobacco_use", "alcohol_use", "drug_use", "school_connectedness",
    "gini_index", "health_exp_pc_usd"
]

ALL_CAT = DEMO_CAT + CONTEXT_CAT
ALL_NUM = DEMO_NUM + CONTEXT_NUM

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sp07:sp07_dev_password@localhost:5432/sp07_db")

def make_preprocessor(cat_cols=ALL_CAT, num_cols=ALL_NUM):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

def calculate_cox_calibration(y_true, probas):
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

def run_pipeline_training(df, out_dir="artifacts", seed=42):
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    
    y_primary = df[TARGET_PRIMARY].astype(int).values
    Y_multi = df[TARGETS_MULTI].astype(int).values
    groups = df["country"].values
    
    neg_count = np.sum(y_primary == 0)
    pos_count = np.sum(y_primary == 1)
    scale_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0
    
    X_full = df[ALL_NUM + ALL_CAT]
    X_demo = df[DEMO_NUM + DEMO_CAT]
    
    models = build_model_factories(scale_weight=scale_weight, seed=seed)
    gkf = GroupKFold(n_splits=5)
    
    oof_predictions = {}
    metrics_by_fold = []
    trained_pipelines = {}
    
    print("\n" + "="*70)
    print("INICIANDO ENTRENAMIENTO VALIDADO POR PAÍS (GroupKFold 5-Splits)")
    print("="*70)
    
    for model_name, model_inst in models.items():
        print(f"\n>>> Evaluando modelo: {model_name}...")
        is_demo_only = (model_name == "baseline_demographic")
        is_multitask = (model_name == "mlp_multitask")
        
        X_cur = X_demo if is_demo_only else X_full
        prep = make_preprocessor(DEMO_CAT, DEMO_NUM) if is_demo_only else make_preprocessor(ALL_CAT, ALL_NUM)
        
        oof_p = np.zeros(len(df))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_cur, y_primary, groups)):
            X_tr, y_tr = X_cur.iloc[train_idx], y_primary[train_idx]
            X_va, y_va = X_cur.iloc[val_idx], y_primary[val_idx]
            
            pipe = Pipeline([
                ("prep", prep),
                ("clf", clone(model_inst))
            ])
            
            if is_multitask:
                pipe.fit(X_tr, Y_multi[train_idx])
            else:
                pipe.fit(X_tr, y_tr)
                
            val_probs = pipe.predict_proba(X_va)[:, 1]
            oof_p[val_idx] = val_probs
            
            fold_m = evaluate_predictions(y_va, val_probs)
            fold_m["model_type"] = model_name
            fold_m["fold"] = f"Fold_{fold+1}"
            metrics_by_fold.append(fold_m)
            
        oof_predictions[model_name] = oof_p
        overall_m = evaluate_predictions(y_primary, oof_p)
        overall_m["model_type"] = model_name
        overall_m["fold"] = "OOF_Overall"
        metrics_by_fold.append(overall_m)
        print(f"[{model_name}] OOF PR-AUC: {overall_m['pr_auc']:.4f} | ROC-AUC: {overall_m['roc_auc']:.4f} | Sens: {overall_m['sensitivity']:.4f} | Spec: {overall_m['specificity']:.4f} | Cox Slope: {overall_m['calib_slope']}")
        
        final_pipe = Pipeline([
            ("prep", prep),
            ("clf", clone(model_inst))
        ])
        if is_multitask:
            final_pipe.fit(X_cur, Y_multi)
        else:
            final_pipe.fit(X_cur, y_primary)
            
        trained_pipelines[model_name] = final_pipe
        
    print("\n>>> Construyendo Ensamble de Votación Ponderada (Ensemble Stacking)...")
    res_df = pd.DataFrame([m for m in metrics_by_fold if m["fold"] == "OOF_Overall"])
    weight_map = {row["model_type"]: max(row["pr_auc"] - 0.20, 0.05) for _, row in res_df.iterrows()}
    
    ensemble_oof_p = (
        oof_predictions["logreg_elasticnet"] * weight_map.get("logreg_elasticnet", 1) +
        oof_predictions["random_forest"] * weight_map.get("random_forest", 1) +
        oof_predictions["xgboost"] * weight_map.get("xgboost", 1) +
        oof_predictions["catboost"] * weight_map.get("catboost", 1) +
        oof_predictions["mlp_multitask"] * weight_map.get("mlp_multitask", 1)
    )
    total_w = sum([weight_map.get(m, 1) for m in ["logreg_elasticnet", "random_forest", "xgboost", "catboost", "mlp_multitask"]])
    ensemble_oof_p /= total_w
    oof_predictions["ensemble_stacking"] = ensemble_oof_p
    
    ens_metrics = evaluate_predictions(y_primary, ensemble_oof_p)
    ens_metrics["model_type"] = "ensemble_stacking"
    ens_metrics["fold"] = "OOF_Overall"
    metrics_by_fold.append(ens_metrics)
    print(f"[ensemble_stacking] OOF PR-AUC: {ens_metrics['pr_auc']:.4f} | ROC-AUC: {ens_metrics['roc_auc']:.4f} | Sens: {ens_metrics['sensitivity']:.4f} | Spec: {ens_metrics['specificity']:.4f} | Cox Slope: {ens_metrics['calib_slope']}")
    
    df_metrics = pd.DataFrame(metrics_by_fold)
    leaderboard = df_metrics[df_metrics.fold == "OOF_Overall"].sort_values("pr_auc", ascending=False).reset_index(drop=True)
    leaderboard_csv = os.path.join(out_dir, "leaderboard_oof.csv")
    leaderboard.to_csv(leaderboard_csv, index=False)
    print(f"\nLeaderboard guardado en: {leaderboard_csv}")
    
    best_model_name = leaderboard.iloc[0]["model_type"]
    if best_model_name == "ensemble_stacking":
        best_single_name = leaderboard[leaderboard.model_type != "ensemble_stacking"].iloc[0]["model_type"]
    else:
        best_single_name = best_model_name
    print(f"\n★ MEJOR MODELO DETECTADO: {best_model_name} (Mejor unidesca: {best_single_name})")
    
    dca_data = compute_decision_curve(y_primary, oof_predictions[best_model_name])
    with open(os.path.join(out_dir, "decision_curve_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(dca_data, f, indent=2)
        
    joblib.dump(trained_pipelines, os.path.join(out_dir, "trained_models.pkl"))
    joblib.dump(oof_predictions, os.path.join(out_dir, "oof_predictions.pkl"))
    
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE sp07.metrics CASCADE;"))
            conn.execute(text("TRUNCATE TABLE sp07.model_registry CASCADE;"))
            conn.execute(text("TRUNCATE TABLE sp07.predictions_agg CASCADE;"))
            
            model_id_map = {}
            for m_name in list(models.keys()) + ["ensemble_stacking"]:
                res = conn.execute(text("""
                    INSERT INTO sp07.model_registry (model_name, model_type, hyperparams, artifact_path, git_commit)
                    VALUES (:name, :type, :hyp, :art, 'v1.0-prod')
                    RETURNING model_id;
                """), {
                    "name": f"SP07_{m_name.upper()}",
                    "type": m_name,
                    "hyp": json.dumps({"description": f"Modelo {m_name} optimizado para estratificación psicosocial"}),
                    "art": f"artifacts/trained_models.pkl"
                })
                model_id_map[m_name] = res.scalar()
                
            for row in metrics_by_fold:
                m_type = row["model_type"]
                mid = model_id_map.get(m_type)
                if mid:
                    conn.execute(text("""
                        INSERT INTO sp07.metrics (model_id, fold, roc_auc, pr_auc, sensitivity, specificity, f1, brier, calib_slope, calib_intercept, threshold)
                        VALUES (:mid, :fold, :roc_auc, :pr_auc, :sens, :spec, :f1, :brier, :slope, :intercept, :thr)
                        ON CONFLICT (model_id, fold) DO NOTHING;
                    """), {
                        "mid": mid,
                        "fold": row["fold"],
                        "roc_auc": row["roc_auc"],
                        "pr_auc": row["pr_auc"],
                        "sens": row["sensitivity"],
                        "spec": row["specificity"],
                        "f1": row["f1"],
                        "brier": row["brier"],
                        "slope": row["calib_slope"],
                        "intercept": row["calib_intercept"],
                        "thr": row["threshold"]
                    })
                    
            df["pred_prob"] = oof_predictions[best_model_name]
            best_thresh = leaderboard.iloc[0]["threshold"]
            df["risk_high"] = (df["pred_prob"] >= best_thresh).astype(int)
            
            agg = df.groupby(["country", "sex", "age_band"]).agg(
                n=("respondent_id", "count"),
                prev_obs=("y1_distress", "mean"),
                prev_pred=("pred_prob", "mean"),
                risk_high=("risk_high", "sum")
            ).reset_index()
            
            best_mid = model_id_map.get(best_model_name, 1)
            for _, r in agg.iterrows():
                conn.execute(text("""
                    INSERT INTO sp07.predictions_agg (model_id, country, sex, age_band, n, prev_obs, prev_pred, risk_high)
                    VALUES (:mid, :country, :sex, :age_band, :n, :prev_obs, :prev_pred, :risk_high)
                    ON CONFLICT (model_id, country, sex, age_band) DO UPDATE SET
                        prev_obs = EXCLUDED.prev_obs,
                        prev_pred = EXCLUDED.prev_pred,
                        risk_high = EXCLUDED.risk_high;
                """), {
                    "mid": best_mid,
                    "country": r["country"],
                    "sex": int(r["sex"]),
                    "age_band": str(r["age_band"]),
                    "n": int(r["n"]),
                    "prev_obs": round(float(r["prev_obs"]), 4),
                    "prev_pred": round(float(r["prev_pred"]), 4),
                    "risk_high": int(r["risk_high"])
                })
            print("Métricas, registros y predicciones agregadas persistidas exitosamente en PostgreSQL.")
    except Exception as e:
        print(f"Aviso de persistencia PostgreSQL en pipeline_engine: {e}")
        
    return leaderboard, oof_predictions, trained_pipelines, best_model_name, best_single_name

if __name__ == "__main__":
    csv_path = "data/gshs_adolescents_clean.csv"
    if os.path.exists(csv_path):
        from eda_engine import run_feature_engineering
        df_clean = run_feature_engineering(pd.read_csv(csv_path))
        run_pipeline_training(df_clean)
        print("Fase 5 y 6 completadas con éxito.")
    else:
        print("Dataset no encontrado.")
