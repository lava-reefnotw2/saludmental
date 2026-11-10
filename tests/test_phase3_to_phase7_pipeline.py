# -*- coding: utf-8 -*-
"""
Tests unitarios y de integración para el pipeline científico SP-07 (Fases 3 a 7).
"""
import os
import sys
import unittest
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestPipelineScience(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from pipeline_sp07 import run_7_phase_pipeline
        # Ejecutar pipeline completo sobre artefactos de prueba
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
        run_7_phase_pipeline(out_dir=out_dir, seed=42)

    def test_leaderboard_metrics_integrity(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        lb_path = os.path.join(root, "artifacts", "leaderboard_oof.csv")
        self.assertTrue(os.path.exists(lb_path), "leaderboard_oof.csv debe existir")
        
        df_lb = pd.read_csv(lb_path)
        self.assertGreaterEqual(len(df_lb), 5, "Deben evaluarse al menos 5 modelos")
        
        # Verificar que PR-AUC es la métrica de ordenamiento
        self.assertTrue(df_lb["pr_auc"].is_monotonic_decreasing, "Leaderboard debe estar ordenado por PR-AUC descendente")
        
        # Verificar rangos válidos de métricas
        for col in ["pr_auc", "roc_auc", "sensitivity", "specificity", "f1", "brier"]:
            self.assertTrue((df_lb[col] >= 0.0).all() and (df_lb[col] <= 1.0).all(), f"Columna {col} fuera de rango [0, 1]")

    def test_hypotheses_results_generated(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        hypo_path = os.path.join(root, "artifacts", "hypotheses_test_results.json")
        self.assertTrue(os.path.exists(hypo_path), "hypotheses_test_results.json debe existir")
        
        with open(hypo_path, "r", encoding="utf-8") as f:
            hypo = json.load(f)
            
        self.assertIn("H1_contextual_gain", hypo)
        self.assertIn("H2_clinical_utility_test", hypo)
        self.assertIn("H3_calibration", hypo)
        self.assertIn("H4_algorithmic_fairness", hypo)
        self.assertIn("H5_ensemble_multitask", hypo)
        self.assertIn("H6_transnational_stability", hypo)

    def test_tabular_explainability_artifacts(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        shap_path = os.path.join(root, "artifacts", "shap_importance_ranking.json")
        self.assertTrue(os.path.exists(shap_path), "shap_importance_ranking.json debe existir")
        
        with open(shap_path, "r", encoding="utf-8") as f:
            ranking = json.load(f)
        self.assertGreater(len(ranking), 5, "SHAP ranking debe tener al menos 5 predictores")
        
        lr_path = os.path.join(root, "artifacts", "logreg_odds_ratios.json")
        self.assertTrue(os.path.exists(lr_path), "logreg_odds_ratios.json debe existir")

if __name__ == "__main__":
    unittest.main()
