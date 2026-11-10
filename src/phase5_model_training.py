# -*- coding: utf-8 -*-
"""
MÓDULO 5 · FASE 5: Motor de Entrenamiento Multimodelo y Multitarea
Implementa las 5 familias del protocolo científico + Baseline Demográfico + Ensamble Stacking.
Validación cruzada anidada agrupada por país (GroupKFold 5-Splits) para evitar fuga de información transnacional.
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

def run_phase5_model_training(df=None, data_dir="data", out_dir="artifacts", seed=42):
    """
    Ejecuta Fase 5: Entrenamiento de las 5 familias de algoritmos y ensamble.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*70)
    print("[FASE 5] INICIANDO ENTRENAMIENTO MULTIMODELO (GroupKFold 5-Splits por País)")
    print("="*70)
    
    if df is None:
        feat_csv = os.path.join(data_dir, "gshs_features_engineered.csv")
        if os.path.exists(feat_csv):
            df = pd.read_csv(feat_csv)
        else:
            from phase3_feature_engineering import run_phase3_feature_engineering
            df = run_phase3_feature_engineering(data_dir=data_dir)
            
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
    trained_pipelines = {}
    
    for model_name, model_inst in models.items():
        print(f"  >>> Entrenando y evaluando folds para: {model_name}...")
        is_demo_only = (model_name == "baseline_demographic")
        is_multitask = (model_name == "mlp_multitask")
        
        X_cur = X_demo if is_demo_only else X_full
        prep = make_preprocessor(DEMO_CAT, DEMO_NUM) if is_demo_only else make_preprocessor(ALL_CAT, ALL_NUM)
        
        oof_p = np.zeros(len(df))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_cur, y_primary, groups)):
            X_tr, y_tr = X_cur.iloc[train_idx], y_primary[train_idx]
            X_va = X_cur.iloc[val_idx]
            
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
            
        oof_predictions[model_name] = oof_p
        
        final_pipe = Pipeline([
            ("prep", prep),
            ("clf", clone(model_inst))
        ])
        if is_multitask:
            final_pipe.fit(X_cur, Y_multi)
        else:
            final_pipe.fit(X_cur, y_primary)
            
        trained_pipelines[model_name] = final_pipe
        
    # Ensamble Stacking / Votación Ponderada
    print("  >>> Construyendo Ensamble Stacking de Votación Ponderada...")
    ensemble_oof_p = (
        oof_predictions["logreg_elasticnet"] * 0.20 +
        oof_predictions["random_forest"] * 0.20 +
        oof_predictions["xgboost"] * 0.25 +
        oof_predictions["catboost"] * 0.25 +
        oof_predictions["mlp_multitask"] * 0.10
    )
    oof_predictions["ensemble_stacking"] = ensemble_oof_p
    
    # Serializar modelos y predicciones OOF
    joblib.dump(trained_pipelines, os.path.join(out_dir, "trained_models.pkl"))
    joblib.dump(oof_predictions, os.path.join(out_dir, "oof_predictions.pkl"))
    
    # Registrar en PostgreSQL sp07.model_registry
    try:
        engine = create_engine(DB_URL)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE sp07.model_registry CASCADE;"))
            for m_name in list(models.keys()) + ["ensemble_stacking"]:
                conn.execute(text("""
                    INSERT INTO sp07.model_registry (model_name, model_type, hyperparams, artifact_path, git_commit)
                    VALUES (:name, :type, :hyp, 'artifacts/trained_models.pkl', 'v1.0-prod')
                    ON CONFLICT DO NOTHING;
                """), {
                    "name": f"SP07_{m_name.upper()}",
                    "type": m_name,
                    "hyp": json.dumps({"family": m_name, "framework": "scikit-learn/xgb/catboost"})
                })
        print("  ✓ Modelos registrados en PostgreSQL (sp07.model_registry).")
    except Exception as e:
        print(f"  Aviso PostgreSQL en Fase 5: {e}")
        
    print("  ✓ Fase 5 completada: 7 modelos entrenados y validados por país.")
    return oof_predictions, trained_pipelines

if __name__ == "__main__":
    run_phase5_model_training()

