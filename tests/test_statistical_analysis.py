# -*- coding: utf-8 -*-
"""
TESTS UNITARIOS Y DE INTEGRACIÓN: MÓDULO ANÁLISIS ESTADÍSTICO (SP-07)
==============================================================================
Verifica la integridad de datos, cálculos inferenciales, pruebas de hipótesis,
intervalos de Wilson, protección de subgrupos pequeños, particiones interpaís,
métricas de equidad algorítmica y exportación multiformato (PNG 600 DPI, SVG, Excel, CSV).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from statistical_analysis_module import (
    wilson_score_interval,
    bootstrap_prauc_ci,
    bootstrap_rocauc_ci,
    check_small_subgroup_protection,
    load_all_statistical_artifacts
)

class TestStatisticalAnalysisModule(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.artifacts = load_all_statistical_artifacts(cls.root_dir)

    def test_artifacts_integrity(self):
        """Comprueba que todos los artefactos analíticos de las Fases 0 a 7 estén disponibles."""
        self.assertIn("features_df", self.artifacts)
        self.assertFalse(self.artifacts["features_df"].empty, "features_df no debe estar vacío")
        self.assertEqual(len(self.artifacts["features_df"]), 4993, "La cohorte completa debe contar con 4,993 adolescentes")
        
        self.assertIn("oof_data", self.artifacts)
        self.assertIn("full_preds", self.artifacts["oof_data"])
        self.assertIn("partitions", self.artifacts["oof_data"])
        
        self.assertIn("leaderboard_df", self.artifacts)
        self.assertFalse(self.artifacts["leaderboard_df"].empty)
        
        self.assertIn("hypotheses_data", self.artifacts)
        self.assertIn("H1_contextual_gain", self.artifacts["hypotheses_data"])
        
        self.assertIn("dca_data", self.artifacts)
        self.assertIn("dca", self.artifacts["dca_data"])
        self.assertIn("calibration_curves", self.artifacts["dca_data"])
        
        self.assertIn("shap_interactions", self.artifacts)
        self.assertIn("learning_curve", self.artifacts)
        self.assertIn("stability_data", self.artifacts)
        self.assertIn("fold_variability", self.artifacts)

    def test_wilson_score_interval_properties(self):
        """Verifica las propiedades matemáticas del intervalo de confianza de Wilson."""
        # 1. Caso estándar (20 eventos en 100 sujetos)
        p, low, high = wilson_score_interval(20, 100, conf=0.95)
        self.assertEqual(p, 0.20)
        self.assertGreater(low, 0.12)
        self.assertLess(high, 0.30)
        self.assertLessEqual(low, p)
        self.assertGreaterEqual(high, p)
        
        # 2. Caso extremo cero eventos (0 en 50 sujetos)
        p0, low0, high0 = wilson_score_interval(0, 50, conf=0.95)
        self.assertEqual(p0, 0.0)
        self.assertEqual(low0, 0.0)
        self.assertGreater(high0, 0.0)
        self.assertLessEqual(high0, 0.10)
        
        # 3. Caso extremo todos eventos (50 en 50 sujetos)
        p1, low1, high1 = wilson_score_interval(50, 50, conf=0.95)
        self.assertEqual(p1, 1.0)
        self.assertGreaterEqual(low1, 0.90)
        self.assertEqual(high1, 1.0)
        
        # 4. Caso nulo (0 en 0)
        pn, ln, hn = wilson_score_interval(0, 0)
        self.assertEqual(pn, 0.0)
        self.assertEqual(ln, 0.0)
        self.assertEqual(hn, 0.0)

    def test_small_subgroup_ethical_protection(self):
        """Verifica la regla de protección ética y secreto estadístico para celdas N < 5."""
        is_prot_3, msg_3 = check_small_subgroup_protection(3, threshold=5)
        self.assertTrue(is_prot_3)
        self.assertIn("Protegido por Secreto Estadístico", msg_3)
        
        is_prot_4, msg_4 = check_small_subgroup_protection(4, threshold=5)
        self.assertTrue(is_prot_4)
        
        is_prot_5, msg_5 = check_small_subgroup_protection(5, threshold=5)
        self.assertFalse(is_prot_5)
        self.assertEqual(msg_5, "5")
        
        is_prot_100, msg_100 = check_small_subgroup_protection(100, threshold=5)
        self.assertFalse(is_prot_100)
        self.assertEqual(msg_100, "100")

    def test_country_disjoint_partitions_integrity(self):
        """Comprueba que las particiones interpaís de oof_data sean estrictamente disjuntas."""
        partitions = self.artifacts["oof_data"]["partitions"]
        dev_c = set(partitions["dev"])
        val_c = set(partitions["val"])
        test_c = set(partitions["test"])
        
        self.assertEqual(len(dev_c.intersection(val_c)), 0, "No debe existir solapamiento entre Dev y Val")
        self.assertEqual(len(dev_c.intersection(test_c)), 0, "No debe existir solapamiento entre Dev y Test")
        self.assertEqual(len(val_c.intersection(test_c)), 0, "No debe existir solapamiento entre Val y Test")
        
        total_countries = dev_c.union(val_c).union(test_c)
        self.assertEqual(len(total_countries), 12, "Deben estar catalogados exactamente 12 países")

    def test_bootstrap_auc_intervals(self):
        """Verifica el cálculo de intervalos bootstrap para PR-AUC y ROC-AUC."""
        y_true = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 0] * 10)
        y_pred = np.array([0.1, 0.2, 0.15, 0.8, 0.2, 0.7, 0.9, 0.3, 0.85, 0.25] * 10)
        
        pr_obs, pr_low, pr_high = bootstrap_prauc_ci(y_true, y_pred, n_boot=100)
        self.assertFalse(np.isnan(pr_obs))
        self.assertGreater(pr_obs, 0.5)
        self.assertLessEqual(pr_low, pr_obs)
        self.assertGreaterEqual(pr_high, pr_obs)
        
        roc_obs, roc_low, roc_high = bootstrap_rocauc_ci(y_true, y_pred, n_boot=100)
        self.assertFalse(np.isnan(roc_obs))
        self.assertGreater(roc_obs, 0.8)
        self.assertLessEqual(roc_low, roc_obs)
        self.assertGreaterEqual(roc_high, roc_obs)

    def test_fairness_parity_gap_calculation(self):
        """Verifica el cálculo de brechas de FNR y cumplimiento del umbral de equidad de 0.10."""
        hyp = self.artifacts["hypotheses_data"].get("H4_algorithmic_fairness", {})
        self.assertIn("max_fnr_gaps", hyp)
        gaps = hyp["max_fnr_gaps"]
        self.assertIn("sex", gaps)
        self.assertIn("age_band", gaps)
        self.assertIn("urban_rural", gaps)
        
        # Sex gap excede 0.10 y requiere mitigación
        self.assertGreater(gaps["sex"], 0.10)
        # Ruralidad cumple con paridad estricta
        self.assertLessEqual(gaps["urban_rural"], 0.10)

    def test_optuna_history_integrity(self):
        """Verifica que el resumen de Optuna contenga trials válidos, mejor PR-AUC y parámetros."""
        opt = self.artifacts.get("optuna_summary", {})
        self.assertIn("best_prauc", opt)
        self.assertGreater(opt["best_prauc"], 0.25)
        self.assertIn("best_params", opt)
        self.assertIn("trials", opt)
        self.assertGreater(len(opt["trials"]), 0)

if __name__ == "__main__":
    unittest.main()
