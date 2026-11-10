# -*- coding: utf-8 -*-
"""
MOTOR DE OPTIMIZACIÓN DE HIPERPARÁMETROS OPTUNA (FASE 7)
Ejecuta estudios bayesianos reales (TPE Sampler) para las familias de modelos,
evaluando el objetivo de PR-AUC exclusivamente dentro de los folds internos del
conjunto de Desarrollo (~66% de países), garantizando que el conjunto de prueba
permanezca 100% blindado (sin fuga de información).
"""

import os
import sys
import json
import joblib
import warnings
import optuna
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase5_model_training import get_feature_blocks, get_country_partitions, make_preprocessor
from models import build_model_factories

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_search_spaces_documentation():
    """Retorna la documentación de los hiperparámetros explorados para cada familia."""
    return {
        "catboost": {
            "depth": "Profundidad de los árboles (3 a 8)",
            "learning_rate": "Tasa de aprendizaje logarítmica (0.01 a 0.20)",
            "l2_leaf_reg": "Regularización L2 en hojas (1.0 a 20.0)",
            "iterations": "Número de árboles / iteraciones de boosting (200 a 600)"
        },
        "xgboost": {
            "max_depth": "Profundidad máxima del árbol (2 a 7)",
            "learning_rate": "Tasa de aprendizaje (0.01 a 0.15)",
            "subsample": "Fracción de muestreo por árbol (0.6 a 1.0)",
            "colsample_bytree": "Fracción de variables por split (0.6 a 1.0)",
            "reg_lambda": "Regularización L2 (1.0 a 25.0)"
        },
        "random_forest": {
            "n_estimators": "Número de árboles en el bosque (200 a 600)",
            "max_depth": "Profundidad máxima (4 a 16)",
            "min_samples_leaf": "Mínimo de muestras por nodo terminal (5 a 30)"
        },
        "logreg_elasticnet": {
            "C": "Inverso de la fuerza de regularización (1e-3 a 1e1 en escala log)",
            "l1_ratio": "Proporción de penalización L1 vs L2 (0.05 a 0.95)"
        },
        "mlp_multitask": {
            "hidden_units": "Neuronas en capas ocultas (32, 64, 128)",
            "alpha": "Regularización L2 (1e-5 a 1e-2)"
        }
    }

def run_optuna_study(model_family="catboost", n_trials=10, timeout=120, seed=42, data_dir="data", out_dir="artifacts"):
    """
    Ejecuta un estudio Optuna real sobre el conjunto de desarrollo.
    Maximiza PR-AUC en inner folds agrupados por país.
    """
    os.makedirs(out_dir, exist_ok=True)
    feat_path = os.path.join(data_dir, "gshs_features_engineered.csv")
    if not os.path.exists(feat_path):
        from phase3_feature_engineering import run_phase3_feature_engineering
        df = run_phase3_feature_engineering(data_dir=data_dir)
    else:
        df = pd.read_csv(feat_path)
        
    dev_c, _, _ = get_country_partitions()
    df_dev = df[df["country"].isin(dev_c)].copy().reset_index(drop=True)
    
    y_dev = df_dev["y1_ideation"].values if "y1_ideation" in df_dev.columns else (df_dev["suicide_ideation"] == 1).astype(int).values
    groups_dev = df_dev["country"].values
    
    b0_cols, b1_cols, b2_cols = get_feature_blocks()
    
    from phase5_model_training import B0_CAT, B1_ADD_CAT, B0_NUM, B1_ADD_NUM, B2_ADD_NUM
    cat_cols = B0_CAT + B1_ADD_CAT
    num_cols = B0_NUM + B1_ADD_NUM + B2_ADD_NUM
    X_dev = df_dev[num_cols + cat_cols]
    
    inner_gkf = GroupKFold(n_splits=3)
    
    def objective(trial):
        if model_family == "catboost":
            import catboost as cb
            depth = trial.suggest_int("depth", 3, 7)
            lr = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
            l2 = trial.suggest_float("l2_leaf_reg", 1.0, 15.0)
            iters = trial.suggest_int("iterations", 150, 400, step=50)
            clf = cb.CatBoostClassifier(
                depth=depth, learning_rate=lr, l2_leaf_reg=l2, iterations=iters,
                auto_class_weights="Balanced", verbose=0, random_seed=seed
            )
        elif model_family == "xgboost":
            import xgboost as xgb
            md = trial.suggest_int("max_depth", 2, 6)
            lr = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
            sub = trial.suggest_float("subsample", 0.65, 0.95)
            lam = trial.suggest_float("reg_lambda", 1.0, 20.0)
            clf = xgb.XGBClassifier(
                max_depth=md, learning_rate=lr, subsample=sub, reg_lambda=lam,
                n_estimators=250, eval_metric="aucpr", random_state=seed, tree_method="hist"
            )
        elif model_family == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            n_est = trial.suggest_int("n_estimators", 150, 400, step=50)
            md = trial.suggest_int("max_depth", 4, 12)
            msl = trial.suggest_int("min_samples_leaf", 5, 25)
            clf = RandomForestClassifier(
                n_estimators=n_est, max_depth=md, min_samples_leaf=msl,
                class_weight="balanced_subsample", random_state=seed, n_jobs=-1
            )
        else: # logreg_elasticnet
            from sklearn.linear_model import LogisticRegression
            c_val = trial.suggest_float("C", 1e-2, 5.0, log=True)
            l1 = trial.suggest_float("l1_ratio", 0.1, 0.8)
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", C=c_val, l1_ratio=l1,
                max_iter=1500, class_weight="balanced", random_state=seed
            )
            
        cv_scores = []
        for tr_idx, val_idx in inner_gkf.split(X_dev, y_dev, groups_dev):
            prep = make_preprocessor(cat_cols, num_cols)
            pipe = Pipeline([("prep", prep), ("clf", clf)])
            pipe.fit(X_dev.iloc[tr_idx], y_dev[tr_idx])
            p_val = pipe.predict_proba(X_dev.iloc[val_idx])[:, 1]
            cv_scores.append(average_precision_score(y_dev[val_idx], p_val))
            
        return float(np.mean(cv_scores))
        
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Serializar resultados
    study_path = os.path.join(out_dir, f"optuna_study_{model_family}.pkl")
    joblib.dump(study, study_path)
    
    trials_df = study.trials_dataframe()
    trials_summary = []
    for t in study.trials:
        trials_summary.append({
            "number": t.number,
            "value_prauc": round(float(t.value), 4) if t.value is not None else None,
            "params": t.params,
            "duration_s": round(float(t.duration.total_seconds()), 2) if t.duration else 0.0,
            "state": str(t.state).split(".")[-1]
        })
        
    summary_path = os.path.join(out_dir, f"optuna_summary_{model_family}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_family": model_family,
            "best_trial_number": study.best_trial.number,
            "best_prauc": round(float(study.best_value), 4),
            "best_params": study.best_params,
            "total_trials": len(study.trials),
            "trials": trials_summary
        }, f, indent=2)
        
    print(f"  ✓ Estudio Optuna para {model_family} completado: Mejor PR-AUC = {study.best_value:.4f} (Trial #{study.best_trial.number})")
    return study

def load_or_run_default_study(model_family="catboost", out_dir="artifacts"):
    """Carga un estudio previo o ejecuta uno breve si no existe."""
    study_path = os.path.join(out_dir, f"optuna_study_{model_family}.pkl")
    if os.path.exists(study_path):
        return joblib.load(study_path)
    return run_optuna_study(model_family=model_family, n_trials=6, timeout=60, out_dir=out_dir)

if __name__ == "__main__":
    study = run_optuna_study("catboost", n_trials=5, timeout=60)
    print("Best params:", study.best_params)

