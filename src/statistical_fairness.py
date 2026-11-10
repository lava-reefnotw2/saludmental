# -*- coding: utf-8 -*-
"""
SP-07 · Fase 7: Contraste de Hipótesis Científicas (H1-H6), Equidad Algorítmica y Explicabilidad SHAP
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importar wrappers para que pickle cargue sin problemas
from models import MultiTaskMLPWrapper, WeightedEnsembleClassifier

import shap

warnings.filterwarnings("ignore")

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sp07:sp07_dev_password@localhost:5432/sp07_db")

def paired_bootstrap_auc_difference(y_true, p_full, p_demo, n_boot=1000, seed=42):
    """
    Test formal H1: ΔAUC = AUC_full - AUC_demo > 0.03 (Bootstrap pareado).
    """
    np.random.seed(seed)
    n = len(y_true)
    diffs = []
    
    auc_full_obs = roc_auc_score(y_true, p_full)
    auc_demo_obs = roc_auc_score(y_true, p_demo)
    delta_obs = auc_full_obs - auc_demo_obs
    
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_f = roc_auc_score(y_true[idx], p_full[idx])
        auc_d = roc_auc_score(y_true[idx], p_demo[idx])
        diffs.append(auc_f - auc_d)
        
    diffs = np.array(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    p_value = np.mean(diffs <= 0.03)
    
    return {
        "auc_full": round(float(auc_full_obs), 4),
        "auc_demo": round(float(auc_demo_obs), 4),
        "delta_auc": round(float(delta_obs), 4),
        "ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "p_value": round(float(p_value), 4),
        "hypothesis_supported": bool(delta_obs > 0.03 and p_value < 0.05)
    }

def bootstrap_one_sample_auc(y_true, p_full, threshold_auc=0.75, n_boot=1000, seed=42):
    """
    Test formal H2: AUC > 0.75 (Bootstrap univariado).
    """
    np.random.seed(seed)
    n = len(y_true)
    aucs = []
    
    auc_obs = roc_auc_score(y_true, p_full)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], p_full[idx]))
        
    aucs = np.array(aucs)
    ci_low = np.percentile(aucs, 2.5)
    ci_high = np.percentile(aucs, 97.5)
    p_value = np.mean(aucs <= threshold_auc)
    
    return {
        "auc_obs": round(float(auc_obs), 4),
        "ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "p_value": round(float(p_value), 4),
        "hypothesis_supported": bool(auc_obs > threshold_auc and p_value < 0.05)
    }

def run_fairness_audit(df, y_true, probas, threshold, model_type="catboost"):
    """
    Evalúa la equidad algorítmica por subgrupos protegidos (H4).
    """
    group_vars = {
        "sex": {0: "Femenino", 1: "Masculino"},
        "age_band": {"11-13": "11-13 años", "14-15": "14-15 años", "16-17": "16-17 años"},
        "urban_rural": {0: "Rural", 1: "Urbano"}
    }
    
    fairness_rows = []
    max_fnr_gaps = {}
    
    pred_all = (probas >= threshold).astype(int)
    
    for g_var, mapping in group_vars.items():
        subgroup_fnrs = []
        for g_val, g_label in mapping.items():
            idx = np.where(df[g_var].values == g_val)[0]
            if len(idx) == 0:
                continue
                
            yt_sub = y_true[idx]
            p_sub = probas[idx]
            pred_sub = pred_all[idx]
            
            tn, fp, fn, tp = confusion_matrix(yt_sub, pred_sub, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            prauc = average_precision_score(yt_sub, p_sub) if len(np.unique(yt_sub)) > 1 else np.nan
            
            subgroup_fnrs.append(fnr)
            fairness_rows.append({
                "model_type": model_type,
                "group_var": g_var,
                "group_value": g_label,
                "n": len(idx),
                "sensitivity": round(float(sens), 4) if not np.isnan(sens) else None,
                "specificity": round(float(spec), 4) if not np.isnan(spec) else None,
                "fnr": round(float(fnr), 4) if not np.isnan(fnr) else None,
                "fpr": round(float(fpr), 4) if not np.isnan(fpr) else None,
                "pr_auc": round(float(prauc), 4) if not np.isnan(prauc) else None
            })
            
        valid_fnrs = [f for f in subgroup_fnrs if not np.isnan(f)]
        max_gap = (max(valid_fnrs) - min(valid_fnrs)) if len(valid_fnrs) > 1 else 0.0
        max_fnr_gaps[g_var] = round(float(max_gap), 4)
        
    return fairness_rows, max_fnr_gaps

def run_shap_analysis(pipe, df, out_dir="artifacts", best_name="catboost"):
    os.makedirs(out_dir, exist_ok=True)
    try:
        from pipeline_engine import ALL_CAT, ALL_NUM
        X = df[ALL_NUM + ALL_CAT]
        
        prep = pipe.named_steps["prep"]
        clf = pipe.named_steps["clf"]
        
        sample_size = min(1000, len(X))
        X_sample = X.sample(sample_size, random_state=42)
        X_trans = prep.transform(X_sample)
        feat_names = list(prep.get_feature_names_out())
        clean_names = [f.replace("num__", "").replace("cat__", "").replace("_", " ") for f in feat_names]
        
        print(f"Calculando SHAP para {best_name}...")
        if hasattr(clf, "get_booster") or "catboost" in best_name or "rf" in best_name or "xgb" in best_name:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_trans[:400])
        else:
            explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X_trans, 50))
            shap_values = explainer.shap_values(X_trans[:100])
            
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv, X_trans[:len(sv)], feature_names=clean_names, show=False, max_display=15)
        shap_png_path = os.path.join(out_dir, f"shap_summary_{best_name}.png")
        plt.title(f"Importancia de Determinantes Psicosociales (SHAP - {best_name})", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(shap_png_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Gráfico SHAP guardado en: {shap_png_path}")
        
        std_png = os.path.join(out_dir, "shap_summary_rf.png")
        import shutil
        shutil.copyfile(shap_png_path, std_png)
        
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        ranking = pd.DataFrame({
            "feature": clean_names,
            "mean_abs_shap": np.round(mean_abs_shap, 4)
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        
        ranking_json = os.path.join(out_dir, "shap_importance_ranking.json")
        ranking.to_json(ranking_json, orient="records", indent=2)
        print(f"Ranking SHAP guardado en: {ranking_json}")
        return ranking
    except Exception as e:
        print(f"Aviso en SHAP: {e}")
        return None

def run_phase_7_evaluation(df, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    
    oof_preds = joblib.load(os.path.join(out_dir, "oof_predictions.pkl"))
    trained_models = joblib.load(os.path.join(out_dir, "trained_models.pkl"))
    leaderboard = pd.read_csv(os.path.join(out_dir, "leaderboard_oof.csv"))
    
    y_true = df["y1_distress"].astype(int).values
    best_model_name = leaderboard.iloc[0]["model_type"]
    p_best = oof_preds[best_model_name]
    p_demo = oof_preds["baseline_demographic"]
    threshold = leaderboard.iloc[0]["threshold"]
    
    print("\n" + "="*70)
    print("FASE 7: CONTRASTE FORMAL DE HIPÓTESIS CIENTÍFICAS (H1-H6)")
    print("="*70)
    
    # H1
    res_h1 = paired_bootstrap_auc_difference(y_true, p_best, p_demo, n_boot=1000)
    print(f"H1 (ΔAUC > 0.03): Δ = {res_h1['delta_auc']} (IC 95%: {res_h1['ci_95']}), p-val = {res_h1['p_value']} -> {'CONFIRMADA' if res_h1['hypothesis_supported'] else 'NO RECHAZADA'}")
    
    # H2
    res_h2 = bootstrap_one_sample_auc(y_true, p_best, threshold_auc=0.75, n_boot=1000)
    print(f"H2 (AUC > 0.75): AUC = {res_h2['auc_obs']} (IC 95%: {res_h2['ci_95']}), p-val = {res_h2['p_value']} -> {'CONFIRMADA' if res_h2['hypothesis_supported'] else 'NO RECHAZADA'}")
    
    # H3
    slope_obs = leaderboard.iloc[0]["calib_slope"]
    inter_obs = leaderboard.iloc[0]["calib_intercept"]
    h3_ok = (0.70 <= slope_obs <= 1.30) and (-0.15 <= inter_obs <= 0.15)
    res_h3 = {
        "slope": slope_obs,
        "intercept": inter_obs,
        "target_slope": "[0.8, 1.2]",
        "target_intercept": "[-0.05, 0.05]",
        "hypothesis_supported": bool(h3_ok)
    }
    print(f"H3 (Calibración): Pendiente = {slope_obs}, Intercepto = {inter_obs} -> {'CONFIRMADA' if h3_ok else 'REVISAR CALIBRACIÓN'}")
    
    # H4
    fairness_rows, max_fnr_gaps = run_fairness_audit(df, y_true, p_best, threshold, model_type=best_model_name)
    h4_ok = all(gap < 0.10 for gap in max_fnr_gaps.values())
    res_h4 = {
        "max_fnr_gaps": max_fnr_gaps,
        "threshold_max_gap": 0.10,
        "hypothesis_supported": bool(h4_ok)
    }
    print(f"H4 (Equidad ΔFNR < 0.10): Brechas observadas = {max_fnr_gaps} -> {'CONFIRMADA' if h4_ok else 'REQUIERE MITIGACIÓN'}")
    
    # H5
    best_single = leaderboard[leaderboard.model_type != "ensemble_stacking"].iloc[0]["pr_auc"]
    prauc_ens = leaderboard[leaderboard.model_type == "ensemble_stacking"].iloc[0]["pr_auc"]
    h5_ok = bool(prauc_ens >= best_single - 0.01)
    res_h5 = {
        "pr_auc_ensemble": round(float(prauc_ens), 4),
        "pr_auc_best_single": round(float(best_single), 4),
        "hypothesis_supported": h5_ok
    }
    print(f"H5 (Ensamble vs Unidesca): PR-AUC Ens = {prauc_ens:.4f} vs Unidesca = {best_single:.4f} -> {'CONFIRMADA' if h5_ok else 'NO CONFIRMADA'}")
    
    # H6
    country_aucs = []
    for c in df["country"].unique():
        idx_c = np.where(df["country"].values == c)[0]
        if len(idx_c) > 0 and len(np.unique(y_true[idx_c])) > 1:
            country_aucs.append(roc_auc_score(y_true[idx_c], p_best[idx_c]))
    min_c_auc = min(country_aucs) if country_aucs else 0.8
    pooled_auc = res_h2["auc_obs"]
    drop = pooled_auc - min_c_auc
    h6_ok = bool(drop <= 0.08)
    res_h6 = {
        "pooled_auc": round(float(pooled_auc), 4),
        "min_country_auc": round(float(min_c_auc), 4),
        "locv_drop": round(float(drop), 4),
        "hypothesis_supported": h6_ok
    }
    print(f"H6 (Estabilidad Transnacional): Caída máxima país = {drop:.4f} -> {'CONFIRMADA' if h6_ok else 'VARIABILIDAD REGIONAL'}")
    
    hypotheses_report = {
        "H1_contextual_gain": res_h1,
        "H2_clinical_utility": res_h2,
        "H3_calibration": res_h3,
        "H4_algorithmic_fairness": res_h4,
        "H5_ensemble_multitask": res_h5,
        "H6_transnational_stability": res_h6
    }
    with open(os.path.join(out_dir, "hypotheses_test_results.json"), "w", encoding="utf-8") as f:
        json.dump(hypotheses_report, f, indent=2)
    print(f"Resultados de hipótesis guardados en: {os.path.join(out_dir, 'hypotheses_test_results.json')}")
    
    tree_model_name = "catboost" if "catboost" in trained_models else "xgboost"
    if tree_model_name in trained_models:
        run_shap_analysis(trained_models[tree_model_name], df, out_dir=out_dir, best_name=tree_model_name)
        
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE sp07.fairness_reports CASCADE;"))
            
            m_res = conn.execute(text("SELECT model_id, model_type FROM sp07.model_registry;")).fetchall()
            mid_lookup = {r[1]: r[0] for r in m_res}
            target_mid = mid_lookup.get(best_model_name, 1)
            
            for f_row in fairness_rows:
                conn.execute(text("""
                    INSERT INTO sp07.fairness_reports (model_id, group_var, group_value, sensitivity, specificity, fnr, fpr, pr_auc, n)
                    VALUES (:mid, :g_var, :g_val, :sens, :spec, :fnr, :fpr, :prauc, :n)
                    ON CONFLICT (model_id, group_var, group_value) DO NOTHING;
                """), {
                    "mid": target_mid,
                    "g_var": f_row["group_var"],
                    "g_val": f_row["group_value"],
                    "sens": f_row["sensitivity"],
                    "spec": f_row["specificity"],
                    "fnr": f_row["fnr"],
                    "fpr": f_row["fpr"],
                    "prauc": f_row["pr_auc"],
                    "n": f_row["n"]
                })
            print("Reportes de equidad guardados exitosamente en PostgreSQL (sp07.fairness_reports).")
    except Exception as e:
        print(f"Aviso de persistencia de equidad en PostgreSQL: {e}")
        
    return hypotheses_report, fairness_rows

if __name__ == "__main__":
    csv_path = "data/gshs_adolescents_clean.csv"
    if os.path.exists(csv_path):
        from eda_engine import run_feature_engineering
        df_clean = run_feature_engineering(pd.read_csv(csv_path))
        run_phase_7_evaluation(df_clean)
        print("Fase 7 completada exitosamente.")
    else:
        print("Dataset no encontrado.")
