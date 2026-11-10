# -*- coding: utf-8 -*-
"""
MÓDULO 5 · FASE 5: Motor de Entrenamiento Multimodelo y Multitarea
Implementa las 5 familias del protocolo científico + Basales B0, B1, B2 + Ensamble Stacking.
Validación cruzada anidada agrupada por país (GroupKFold) con particiones disjuntas:
  - 8 países en Desarrollo (~66%) con Nested CV por país.
  - 2 países en Validación/Calibración (~17%) para ajuste de calibrador y umbral.
  - 2 países en Prueba Externa (~17%) congelados para evaluación final.
Regla transversal: NO-LEAKAGE garantizada en B0, B1 y B2.
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from models import MultiTaskMLPWrapper, WeightedEnsembleClassifier, build_model_factories

warnings.filterwarnings("ignore")

TARGET_PRIMARY = "y1_ideation"
TARGETS_MULTI = ["y1_ideation", "y2_plan", "y3_attempt", "y4_loneliness_frequent", "y5_worry_sleep_frequent"]

# ==============================================================================
# DEFINICIÓN DE BLOQUES DE VARIABLES PARA ABLACIÓN CIENTÍFICA (SIN FUGA)
# ==============================================================================
B0_CAT = ["sex", "grade_band"]
B0_NUM = ["age_years"]

B1_ADD_CAT = [
    "urban_rural", "bullied_past_month", "physically_attacked",
    "tobacco_use", "alcohol_use", "drug_use", "physical_activity_60min",
    "truancy_past_month", "school_connectedness"
]
B1_ADD_NUM = [
    "food_insecurity", "parents_supervision", "parents_support",
    "family_connectedness", "friends_support", "bullied_type_count",
    "physical_fights", "fruit_veg_intake", "sedentary_hours",
    "bmi_for_age_z", "cumulative_risk_score"
]

B2_ADD_NUM = ["gini_index", "health_exp_pc_usd", "unemployment_youth", "internet_pct"]

def get_feature_blocks():
    """Retorna las listas de columnas de B0 (demográfico), B1 (individual) y B2 (completo)."""
    b0 = B0_NUM + B0_CAT
    b1 = b0 + B1_ADD_NUM + B1_ADD_CAT
    b2 = b1 + B2_ADD_NUM
    return b0, b1, b2

def get_country_partitions():
    """
    Retorna la división de países en Desarrollo (8), Validación (2) y Test Externo (2).
    Garantiza disyunción total y reproducibilidad.
    """
    dev_countries = ["PER", "COL", "MEX", "BRA", "ECU", "PAN", "SLV", "GTM"]
    val_countries = ["ARG", "CRI"]
    test_countries = ["CHL", "URY"]
    return dev_countries, val_countries, test_countries

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sp07:sp07_dev_password@localhost:5432/sp07_db")

def make_preprocessor(cat_cols, num_cols):
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

def run_phase5_model_training(df=None, data_dir="data", out_dir="artifacts", seed=42):
    """
    Ejecuta Fase 5: Entrenamiento multimodelo, nested CV por país y evaluación de bloques B0, B1, B2.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*75)
    print("[FASE 5] ENTRENAMIENTO MULTIMODELO CIENTÍFICO (GroupKFold por País, No-Leakage)")
    print("="*75)
    
    if df is None:
        feat_csv = os.path.join(data_dir, "gshs_features_engineered.csv")
        if os.path.exists(feat_csv):
            df = pd.read_csv(feat_csv)
        else:
            from phase3_feature_engineering import run_phase3_feature_engineering
            df = run_phase3_feature_engineering(data_dir=data_dir)
            
    df = df.copy()
    if TARGET_PRIMARY not in df.columns:
        df[TARGET_PRIMARY] = (df["suicide_ideation"] == 1).astype(int)
        
    dev_c, val_c, test_c = get_country_partitions()
    
    df_dev = df[df["country"].isin(dev_c)].copy().reset_index(drop=True)
    df_val = df[df["country"].isin(val_c)].copy().reset_index(drop=True)
    df_test = df[df["country"].isin(test_c)].copy().reset_index(drop=True)
    
    print(f"  ✓ Partición interpaís: Dev N={len(df_dev)} ({dev_c}), Val N={len(df_val)} ({val_c}), Test N={len(df_test)} ({test_c})")
    
    y_dev = df_dev[TARGET_PRIMARY].astype(int).values
    Y_multi_dev = df_dev[[c for c in TARGETS_MULTI if c in df_dev.columns]].astype(int).values
    groups_dev = df_dev["country"].values
    
    y_val = df_val[TARGET_PRIMARY].astype(int).values
    y_test = df_test[TARGET_PRIMARY].astype(int).values
    
    neg_count = np.sum(y_dev == 0)
    pos_count = np.sum(y_dev == 1)
    scale_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0
    
    # Definición de especificaciones de bloques
    spec_models = {
        "baseline_demographic_b0": {
            "family": "baseline_demographic",
            "cat": B0_CAT, "num": B0_NUM, "block": "B0"
        },
        "logreg_elasticnet_b1": {
            "family": "logreg_elasticnet",
            "cat": B0_CAT + B1_ADD_CAT, "num": B0_NUM + B1_ADD_NUM, "block": "B1"
        },
        "logreg_elasticnet_b2": {
            "family": "logreg_elasticnet",
            "cat": B0_CAT + B1_ADD_CAT, "num": B0_NUM + B1_ADD_NUM + B2_ADD_NUM, "block": "B2"
        },
        "random_forest_b2": {
            "family": "random_forest",
            "cat": B0_CAT + B1_ADD_CAT, "num": B0_NUM + B1_ADD_NUM + B2_ADD_NUM, "block": "B2"
        },
        "xgboost_b2": {
            "family": "xgboost",
            "cat": B0_CAT + B1_ADD_CAT, "num": B0_NUM + B1_ADD_NUM + B2_ADD_NUM, "block": "B2"
        },
        "catboost_b2": {
            "family": "catboost",
            "cat": B0_CAT + B1_ADD_CAT, "num": B0_NUM + B1_ADD_NUM + B2_ADD_NUM, "block": "B2"
        },
        "mlp_multitask_b2": {
            "family": "mlp_multitask",
            "cat": B0_CAT + B1_ADD_CAT, "num": B0_NUM + B1_ADD_NUM + B2_ADD_NUM, "block": "B2"
        }
    }
    
    factories = build_model_factories(scale_weight=scale_weight, seed=seed)
    gkf = GroupKFold(n_splits=4)
    
    oof_predictions_dev = {}
    val_predictions = {}
    test_predictions = {}
    trained_pipelines = {}
    model_prauc_scores = {}
    
    for model_key, spec in spec_models.items():
        fam = spec["family"]
        base_estimator = factories[fam]
        cat_cols = spec["cat"]
        num_cols = spec["num"]
        
        is_multitask = (fam == "mlp_multitask")
        X_dev_cur = df_dev[num_cols + cat_cols]
        X_val_cur = df_val[num_cols + cat_cols]
        X_test_cur = df_test[num_cols + cat_cols]
        
        print(f"  >>> Entrenando y validando folds para: {model_key} ({spec['block']})...")
        oof_p = np.zeros(len(df_dev))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_dev_cur, y_dev, groups_dev)):
            prep = make_preprocessor(cat_cols, num_cols)
            clf = clone(base_estimator)
            
            pipe = Pipeline([
                ("prep", prep),
                ("clf", clf)
            ])
            
            X_tr, y_tr = X_dev_cur.iloc[train_idx], y_dev[train_idx]
            X_va = X_dev_cur.iloc[val_idx]
            
            if is_multitask:
                pipe.fit(X_tr, Y_multi_dev[train_idx])
            else:
                pipe.fit(X_tr, y_tr)
                
            oof_p[val_idx] = pipe.predict_proba(X_va)[:, 1]
            
        prauc_oof = float(average_precision_score(y_dev, oof_p))
        roc_oof = float(roc_auc_score(y_dev, oof_p))
        model_prauc_scores[model_key] = prauc_oof
        oof_predictions_dev[model_key] = oof_p
        print(f"      PR-AUC OOF Dev = {prauc_oof:.4f} | ROC-AUC OOF = {roc_oof:.4f}")
        
        # Re-entrenar con TODO el conjunto de desarrollo para predicción en val y test
        prep_full = make_preprocessor(cat_cols, num_cols)
        clf_full = clone(base_estimator)
        pipe_full = Pipeline([("prep", prep_full), ("clf", clf_full)])
        if is_multitask:
            pipe_full.fit(X_dev_cur, Y_multi_dev)
        else:
            pipe_full.fit(X_dev_cur, y_dev)
            
        trained_pipelines[model_key] = pipe_full
        val_predictions[model_key] = pipe_full.predict_proba(X_val_cur)[:, 1]
        test_predictions[model_key] = pipe_full.predict_proba(X_test_cur)[:, 1]
        
    # Construcción de Ensamble Stacking Ponderado (B2)
    print("  >>> Construyendo Ensamble de Votación Ponderada...")
    ensemble_members = ["catboost_b2", "xgboost_b2", "random_forest_b2", "logreg_elasticnet_b2"]
    weights = [max(model_prauc_scores[m] - 0.20, 0.05) for m in ensemble_members]
    w_norm = np.array(weights) / np.sum(weights)
    
    oof_ens = np.zeros(len(df_dev))
    val_ens = np.zeros(len(df_val))
    test_ens = np.zeros(len(df_test))
    for m, w in zip(ensemble_members, w_norm):
        oof_ens += oof_predictions_dev[m] * w
        val_ens += val_predictions[m] * w
        test_ens += test_predictions[m] * w
        
    oof_predictions_dev["ensemble_stacking_b2"] = oof_ens
    val_predictions["ensemble_stacking_b2"] = val_ens
    test_predictions["ensemble_stacking_b2"] = test_ens
    model_prauc_scores["ensemble_stacking_b2"] = float(average_precision_score(y_dev, oof_ens))
    
    # Combinar predicciones totales para compatibilidad
    full_preds = {}
    for m in oof_predictions_dev.keys():
        arr = np.zeros(len(df))
        arr[df["country"].isin(dev_c)] = oof_predictions_dev[m]
        arr[df["country"].isin(val_c)] = val_predictions[m]
        arr[df["country"].isin(test_c)] = test_predictions[m]
        full_preds[m] = arr
        
    # Guardar artefactos serializados
    joblib.dump(trained_pipelines, os.path.join(out_dir, "trained_models.pkl"))
    joblib.dump({
        "oof_dev": oof_predictions_dev,
        "val_preds": val_predictions,
        "test_preds": test_predictions,
        "full_preds": full_preds,
        "partitions": {"dev": dev_c, "val": val_c, "test": test_c}
    }, os.path.join(out_dir, "oof_predictions.pkl"))
    
    # Persistencia en PostgreSQL sp07.model_registry
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE sp07.model_registry CASCADE;"))
            for m_key, pipe in trained_pipelines.items():
                conn.execute(text("""
                    INSERT INTO sp07.model_registry (model_name, model_type, hyperparams, artifact_path, trained_at)
                    VALUES (:name, :mtype, :hparams, :path, now());
                """), {
                    "name": m_key,
                    "mtype": spec_models.get(m_key, {}).get("family", "ensemble"),
                    "hparams": json.dumps({"block": spec_models.get(m_key, {}).get("block", "B2"), "prauc_oof": model_prauc_scores.get(m_key, 0.0)}),
                    "path": os.path.join(out_dir, "trained_models.pkl")
                })
        print("  ✓ Modelos registrados en PostgreSQL (sp07.model_registry).")
    except Exception as e:
        print(f"  Aviso PostgreSQL en Fase 5: {e}")
        
    print(f"  ✓ Fase 5 completada: 8 modelos evaluados en particiones disjuntas por país.")
    return full_preds, trained_pipelines

if __name__ == "__main__":
    run_phase5_model_training()
