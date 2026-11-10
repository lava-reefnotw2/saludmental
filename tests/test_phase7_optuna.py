# -*- coding: utf-8 -*-
"""
Tests unitarios para FASE 7: Optimización Bayesiana Optuna y seguimiento de trials.
"""
import os
import sys
import unittest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestPhase7Optuna(unittest.TestCase):

    def setUp(self):
        self.out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))

    def test_search_spaces_documentation(self):
        from optuna_tuner import get_search_spaces_documentation
        docs = get_search_spaces_documentation()
        self.assertIn("catboost", docs)
        self.assertIn("xgboost", docs)
        self.assertIn("random_forest", docs)
        self.assertIn("logreg_elasticnet", docs)

    def test_optuna_study_artifact_and_trials(self):
        summary_path = os.path.join(self.out_dir, "optuna_summary_catboost.json")
        self.assertTrue(os.path.exists(summary_path), "optuna_summary_catboost.json debe existir")
        
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            
        self.assertEqual(summary.get("model_family"), "catboost")
        self.assertGreater(summary.get("total_trials", 0), 0)
        self.assertIn("best_params", summary)
        self.assertGreater(summary.get("best_prauc", 0), 0)

if __name__ == "__main__":
    unittest.main()

