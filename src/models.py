# -*- coding: utf-8 -*-
"""
SP-07 · Definición de Modelos y Ensamble Multitarea
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as cb

class MultiTaskMLPWrapper(BaseEstimator, ClassifierMixin):
    """
    Modelo multitarea para desenlaces simultáneos [Y1, Y2, Y3, Y4] usando MLP.
    Aprende representaciones compartidas entre desenlaces psicológicos.
    """
    def __init__(self, hidden_layer_sizes=(128, 64, 32), random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.base_mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=150,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=self.random_state
        )
        self.multi_clf = MultiOutputClassifier(self.base_mlp)
        self.classes_ = np.array([0, 1])

    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.multi_clf.fit(X, Y)
        return self

    def predict_proba(self, X):
        probas_all = self.multi_clf.predict_proba(X)
        if isinstance(probas_all, list):
            return probas_all[0]
        return probas_all

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensamble de votación probabilística ponderada por el PR-AUC OOF de cada modelo.
    """
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        for name, est in self.estimators:
            est.fit(X, y)
        return self

    def predict_proba(self, X):
        probas = [est.predict_proba(X)[:, 1] for name, est in self.estimators]
        w = self.weights if self.weights is not None else np.ones(len(probas))
        w = np.array(w) / np.sum(w)
        blended = np.zeros(len(X))
        for p, weight in zip(probas, w):
            blended += p * weight
        return np.vstack([1 - blended, blended]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def build_model_factories(scale_weight=4.0, seed=42):
    models = {
        "baseline_demographic": LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", class_weight="balanced", random_state=seed
        ),
        "logreg_elasticnet": LogisticRegression(
            penalty="elasticnet", solver="saga", C=0.2, l1_ratio=0.3,
            max_iter=3000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=15, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=450, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
            reg_lambda=5.0, scale_pos_weight=scale_weight,
            tree_method="hist", eval_metric="aucpr", random_state=seed
        ),
        "catboost": cb.CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=6.0,
            auto_class_weights="Balanced", verbose=0, random_seed=seed
        ),
        "mlp_multitask": MultiTaskMLPWrapper(
            hidden_layer_sizes=(128, 64, 32), random_state=seed
        )
    }
    return models

