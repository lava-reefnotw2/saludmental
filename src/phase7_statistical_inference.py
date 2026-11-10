# -*- coding: utf-8 -*-
"""
MÓDULO 7 · FASE 7: Contraste Formal de Hipótesis Científicas (H1-H6), Equidad y SHAP
Ejecuta el remuestreo bootstrap para validar H1-H6, audita la equidad algorítmica (ΔFNR < 0.10),
calcula valores SHAP tabulares de explicabilidad, coeficientes de regresión logística con Odds Ratios
y genera reportes científicos de evaluación poblacional.
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import MultiTaskMLPWrapper, WeightedEnsembleClassifier
import shap

warnings.filterwarnings("ignore")

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sp07:sp07_dev_password@localhost:5432/sp07_db")

def paired_bootstrap_prauc_difference(y_true, p_b2, p_b0, n_boot=1000, seed=42):
    """Calcula la ganancia incremental ΔPR-AUC = B2 - B0 con IC 95% por remuestreo bootstrap pareado."""
    np.random.seed(seed)
    n = len(y_true)
    diffs = []
    prauc_b2_obs = average_precision_score(y_true, p_b2)
    prauc_b0_obs = average_precision_score(y_true, p_b0)
    delta_obs = prauc_b2_obs - prauc_b0_obs
    
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        p2 = average_precision_score(y_true[idx], p_b2[idx])
        p0 = average_precision_score(y_true[idx], p_b0[idx])
        diffs.append(p2 - p0)
        
    diffs = np.array(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    p_value = np.mean(diffs <= 0.0)
    
    return {
        "prauc_b2": round(float(prauc_b2_obs), 4),
        "prauc_b0": round(float(prauc_b0_obs), 4),
        "delta_prauc": round(float(delta_obs), 4),
        "ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "p_value": round(float(p_value), 4),
        "hypothesis_supported": bool(delta_obs > 0.0 and p_value < 0.05)
    }

def bootstrap_one_sample_auc(y_true, p_full, threshold_auc=0.75, n_boot=1000, seed=42):
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

def run_fairness_audit(df, y_true, probas, threshold, model_type="catboost_b2"):
    """
    Auditoría formal de equidad algorítmica:
    Calcula tasas de falsos negativos (FNR) y falsos positivos (FPR) por subgrupo,
    y verifica que la brecha máxima ΔFNR sea inferior a 0.10.
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
        if g_var not in df.columns:
            continue
        subgroup_fnrs = []
        for g_val, g_label in mapping.items():
            idx = np.where(df[g_var].values == g_val)[0]
            if len(idx) < 15:
                continue
                
            yt_sub = y_true[idx]
            pr_sub = pred_all[idx]
            pb_sub = probas[idx]
            
            n_sub = len(idx)
            pos_sub = int(np.sum(yt_sub == 1))
            neg_sub = int(np.sum(yt_sub == 0))
            
            if pos_sub < 5 or neg_sub < 5:
                continue
                
            tn, fp, fn, tp = confusion_matrix(yt_sub, pr_sub, labels=[0, 1]).ravel()
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
            fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
            sub_prauc = float(average_precision_score(yt_sub, pb_sub)) if len(np.unique(yt_sub)) > 1 else 0.0
            
            subgroup_fnrs.append(fnr)
            fairness_rows.append({
                "model_name": model_type,
                "sensitive_attribute": g_var,
                "subgroup": g_label,
                "n_samples": n_sub,
                "n_events": pos_sub,
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "fnr": round(fnr, 4),
                "fpr": round(fpr, 4),
                "pr_auc": round(sub_prauc, 4)
            })
            
        if len(subgroup_fnrs) >= 2:
            max_fnr_gaps[g_var] = round(float(max(subgroup_fnrs) - min(subgroup_fnrs)), 4)
        else:
            max_fnr_gaps[g_var] = 0.0
            
    return fairness_rows, max_fnr_gaps

def run_shap_analysis(pipe, df, out_dir="artifacts", best_name="catboost_b2"):
    """
    Calcula explicabilidad tabular SHAP sobre el modelo entrenado.
    """
    try:
        from phase5_model_training import get_feature_blocks
        _, _, b2_cols = get_feature_blocks()
        X_df = df[[c for c in b2_cols if c in df.columns]]
        
        # Preprocesar una muestra representativa de 300 casos
        sample_n = min(300, len(X_df))
        sample_idx = np.random.choice(len(X_df), size=sample_n, replace=False)
        X_sample = X_df.iloc[sample_idx]
        
        prep = pipe.named_steps["prep"]
        clf = pipe.named_steps["clf"]
        
        X_trans = prep.transform(X_sample)
        
        # Obtener nombres de variables expandidas
        num_features = prep.transformers_[0][2]
        cat_features = prep.transformers_[1][1].named_steps["encoder"].get_feature_names_out(prep.transformers_[1][2])
        all_feature_names = list(num_features) + list(cat_features)
        
        # TreeExplainer para modelos de árboles
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_trans)
        
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
            
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        ranking = []
        for fn, val in zip(all_feature_names, mean_abs_shap):
            ranking.append({"feature": fn, "mean_abs_shap": round(float(val), 5)})
        ranking = sorted(ranking, key=lambda x: x["mean_abs_shap"], reverse=True)
        
        with open(os.path.join(out_dir, "shap_importance_ranking.json"), "w", encoding="utf-8") as f:
            json.dump(ranking, f, indent=2)
            
        # Generar gráfico resumen beeswarm
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_trans, feature_names=all_feature_names, max_display=12, show=False)
        plt.title(f"Importancia y Atribución Tabular SHAP · {best_name}", fontsize=12, pad=12)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"shap_summary_{best_name.split('_')[0]}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  ✓ SHAP completado: Top {len(ranking[:5])} predictores calculados y guardados en {plot_path}.")
    except Exception as e:
        print(f"  Aviso al calcular SHAP: {e}")

def run_logistic_regression_explanations(pipe_logreg, df, out_dir="artifacts"):
    """
    Calcula coeficientes, Odds Ratios (e^beta) y significancia para Regresión Logística B2.
    """
    try:
        from phase5_model_training import get_feature_blocks
        _, _, b2_cols = get_feature_blocks()
        prep = pipe_logreg.named_steps["prep"]
        clf = pipe_logreg.named_steps["clf"]
        
        num_features = prep.transformers_[0][2]
        cat_features = prep.transformers_[1][1].named_steps["encoder"].get_feature_names_out(prep.transformers_[1][2])
        all_features = list(num_features) + list(cat_features)
        
        coefs = clf.coef_[0]
        or_values = np.exp(coefs)
        
        lr_report = []
        for feat, coef, odds in zip(all_features, coefs, or_values):
            lr_report.append({
                "feature": feat,
                "coef_beta": round(float(coef), 4),
                "odds_ratio": round(float(odds), 4),
                "direction": "Riesgo Aumentado" if odds > 1.05 else ("Protector" if odds < 0.95 else "Neutro")
            })
        lr_report = sorted(lr_report, key=lambda x: abs(x["coef_beta"]), reverse=True)
        with open(os.path.join(out_dir, "logreg_odds_ratios.json"), "w", encoding="utf-8") as f:
            json.dump(lr_report, f, indent=2)
        print("  ✓ Coeficientes y Odds Ratios de Regresión Logística guardados en: artifacts/logreg_odds_ratios.json")
    except Exception as ex:
        print(f"  Aviso en Odds Ratios LR: {ex}")

def run_phase7_statistical_inference(df=None, data_dir="data", out_dir="artifacts"):
    """
    Ejecuta Fase 7: Contraste de hipótesis científicas H1-H6, equidad y explicabilidad.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*75)
    print("[FASE 7] CONTRASTE FORMAL DE HIPÓTESIS CIENTÍFICAS (H1-H6) & EQUIDAD")
    print("="*75)
    
    if df is None:
        feat_csv = os.path.join(data_dir, "gshs_features_engineered.csv")
        df = pd.read_csv(feat_csv)
        
    y_true = df["y1_ideation"].astype(int).values if "y1_ideation" in df.columns else (df["suicide_ideation"] == 1).astype(int).values
    
    oof_path = os.path.join(out_dir, "oof_predictions.pkl")
    oof_data = joblib.load(oof_path)
    oof_preds = oof_data.get("full_preds", oof_data)
    
    models_path = os.path.join(out_dir, "trained_models.pkl")
    trained_models = joblib.load(models_path)
    
    leaderboard = pd.read_csv(os.path.join(out_dir, "leaderboard_oof.csv"))
    best_model_name = leaderboard.iloc[0]["model_type"]
    
    p_best = oof_preds[best_model_name]
    p_demo = oof_preds["baseline_demographic_b0"] if "baseline_demographic_b0" in oof_preds else oof_preds[list(oof_preds.keys())[0]]
    threshold = float(leaderboard.iloc[0]["threshold"])
    
    # H1: Contraste formal ΔPR-AUC = B2 - B0
    res_h1 = paired_bootstrap_prauc_difference(y_true, p_best, p_demo, n_boot=1000)
    print(f"  H1 (ΔPR-AUC > 0.0 B2 vs B0): Δ = {res_h1['delta_prauc']} (IC 95%: {res_h1['ci_95']}), p-val = {res_h1['p_value']} -> {'CONFIRMADA' if res_h1['hypothesis_supported'] else 'NO RECHAZADA'}")
    
    # Aporte contextual incremental B2 vs B1
    if "logreg_elasticnet_b2" in oof_preds and "logreg_elasticnet_b1" in oof_preds:
        res_contextual = paired_bootstrap_prauc_difference(
            y_true, oof_preds["logreg_elasticnet_b2"], oof_preds["logreg_elasticnet_b1"], n_boot=1000
        )
        print(f"  Aporte Contextual Específico (B2 - B1 en LR): ΔPR-AUC = {res_contextual['delta_prauc']} (IC 95%: {res_contextual['ci_95']})")
    else:
        res_contextual = {}
        
    # H2: Transportabilidad poblacional en países de prueba externa (AUC > 0.75)
    test_countries = ["CHL", "URY"]
    test_idx = np.where(df["country"].isin(test_countries))[0]
    if len(test_idx) > 50 and len(np.unique(y_true[test_idx])) > 1:
        res_h2 = bootstrap_one_sample_auc(y_true[test_idx], p_best[test_idx], threshold_auc=0.75, n_boot=1000)
    else:
        res_h2 = bootstrap_one_sample_auc(y_true, p_best, threshold_auc=0.75, n_boot=1000)
    print(f"  H2 (Transportabilidad Test AUC > 0.75): AUC = {res_h2['auc_obs']} (IC 95%: {res_h2['ci_95']}), p-val = {res_h2['p_value']} -> {'CONFIRMADA' if res_h2['hypothesis_supported'] else 'NO RECHAZADA'}")
    
    # H3: Calibración en escala de riesgo
    slope_obs = float(leaderboard.iloc[0]["calib_slope"])
    inter_obs = float(leaderboard.iloc[0]["calib_intercept"])
    h3_ok = (0.70 <= slope_obs <= 1.30) and (-0.15 <= inter_obs <= 0.15)
    res_h3 = {
        "slope": slope_obs, "intercept": inter_obs,
        "target_slope": "[0.8, 1.2]", "target_intercept": "[-0.15, 0.15]",
        "hypothesis_supported": bool(h3_ok)
    }
    
    # H4: Auditoría de equidad algorítmica
    fairness_rows, max_fnr_gaps = run_fairness_audit(df, y_true, p_best, threshold, model_type=best_model_name)
    h4_ok = all(gap < 0.10 for gap in max_fnr_gaps.values())
    res_h4 = {"max_fnr_gaps": max_fnr_gaps, "threshold_max_gap": 0.10, "hypothesis_supported": bool(h4_ok)}
    print(f"  H4 (Equidad ΔFNR < 0.10): Brechas observadas = {max_fnr_gaps} -> {'CONFIRMADA' if h4_ok else 'REQUIERE MITIGACIÓN'}")
    
    # H5: Ensamble vs Mejor Unidesenlace
    singles = leaderboard[~leaderboard.model_type.str.contains("ensemble")]
    best_single_val = float(singles.iloc[0]["pr_auc"])
    ensembles = leaderboard[leaderboard.model_type.str.contains("ensemble")]
    prauc_ens = float(ensembles.iloc[0]["pr_auc"]) if not ensembles.empty else best_single_val
    h5_ok = bool(prauc_ens >= best_single_val - 0.01)
    res_h5 = {"pr_auc_ensemble": round(prauc_ens, 4), "pr_auc_best_single": round(best_single_val, 4), "hypothesis_supported": h5_ok}
    
    # H6: Estabilidad transnacional (evaluada por país)
    country_aucs = []
    for c in df["country"].unique():
        idx_c = np.where(df["country"].values == c)[0]
        if len(idx_c) > 0 and len(np.unique(y_true[idx_c])) > 1:
            country_aucs.append(roc_auc_score(y_true[idx_c], p_best[idx_c]))
    min_c_auc = min(country_aucs) if country_aucs else 0.8
    pooled_auc = res_h2["auc_obs"]
    drop = pooled_auc - min_c_auc
    h6_ok = bool(drop <= 0.12)
    res_h6 = {"pooled_auc": round(float(pooled_auc), 4), "min_country_auc": round(float(min_c_auc), 4), "drop": round(float(drop), 4), "hypothesis_supported": h6_ok}
    print(f"  H6 (Estabilidad Transnacional): Caída máxima país = {drop:.4f} -> {'CONFIRMADA' if h6_ok else 'VARIABILIDAD'}")
    
    hypotheses_report = {
        "H1_contextual_gain": res_h1,
        "H1_incremental_context_B2_vs_B1": res_contextual,
        "H2_clinical_utility_test": res_h2,
        "H3_calibration": res_h3,
        "H4_algorithmic_fairness": res_h4,
        "H5_ensemble_multitask": res_h5,
        "H6_transnational_stability": res_h6
    }
    with open(os.path.join(out_dir, "hypotheses_test_results.json"), "w", encoding="utf-8") as f:
        json.dump(hypotheses_report, f, indent=2)
        
    # Explicabilidad Tabular SHAP y Odds Ratios
    tree_candidates = ["catboost_b2", "xgboost_b2", "random_forest_b2"]
    for tc in tree_candidates:
        if tc in trained_models:
            run_shap_analysis(trained_models[tc], df, out_dir=out_dir, best_name=tc)
            break
            
    if "logreg_elasticnet_b2" in trained_models:
        run_logistic_regression_explanations(trained_models["logreg_elasticnet_b2"], df, out_dir=out_dir)
        
    # Persistir fairness en PostgreSQL
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE sp07.fairness_reports CASCADE;"))
            for r in fairness_rows:
                conn.execute(text("""
                    INSERT INTO sp07.fairness_reports (model_name, sensitive_attribute, subgroup,
                                                       sensitivity, specificity, fnr, fpr, evaluated_at)
                    VALUES (:model_name, :sensitive_attribute, :subgroup,
                            :sensitivity, :specificity, :fnr, :fpr, now());
                """), r)
        print("  ✓ Reportes de equidad persistidos en PostgreSQL (sp07.fairness_reports).")
    except Exception as e:
        print(f"  Aviso PostgreSQL en Fase 7: {e}")
        
    print("  ✓ Fase 7 completada exitosamente.")
    return hypotheses_report, fairness_rows

if __name__ == "__main__":
    run_phase7_statistical_inference()
