# -*- coding: utf-8 -*-
"""
Tests para FASE 1 y FASE 2:
1. Verificación de desenlaces Y1-Y5 según PROTOCOLO_SP07_CORREGIDO.md (v2.0).
2. Verificación ESTRICTA de NO FUGA de información (no-leakage) en matrices de predictores B0, B1 y B2.
3. Verificación de disyunción de países entre desarrollo, validación y prueba externa.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestLeakageAndOutcomes(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        self.raw_path = os.path.join(self.data_dir, "gshs_raw_responses.csv")
        self.feat_path = os.path.join(self.data_dir, "gshs_features_engineered.csv")

    def test_outcomes_definition_protocol_v2(self):
        """Verifica que los desenlaces corresponden a las definiciones del Protocolo v2.0."""
        from phase3_feature_engineering import derive_outcomes_and_features
        df_raw = pd.read_csv(self.raw_path)
        df_feat = derive_outcomes_and_features(df_raw)
        
        # Y1 debe ser ideación suicida autorreportada en últimos 12 meses
        self.assertIn("y1_ideation", df_feat.columns, "Y1 debe llamarse y1_ideation")
        expected_y1 = (df_raw["suicide_ideation"] == 1).astype(int)
        np.testing.assert_array_equal(df_feat["y1_ideation"].values, expected_y1.values)
        
        # Y2 plan, Y3 intento, Y4 soledad frecuente, Y5 sueño frecuente
        self.assertIn("y2_plan", df_feat.columns)
        self.assertIn("y3_attempt", df_feat.columns)
        self.assertIn("y4_loneliness_frequent", df_feat.columns)
        self.assertIn("y5_worry_sleep_frequent", df_feat.columns)
        
        np.testing.assert_array_equal(df_feat["y2_plan"].values, (df_raw["suicide_plan"] == 1).astype(int).values)
        np.testing.assert_array_equal(df_feat["y3_attempt"].values, (df_raw["suicide_attempt"] == 1).astype(int).values)
        np.testing.assert_array_equal(df_feat["y4_loneliness_frequent"].values, (df_raw["loneliness"] >= 4).astype(int).values)
        np.testing.assert_array_equal(df_feat["y5_worry_sleep_frequent"].values, (df_raw["anxiety_sleep"] >= 4).astype(int).values)

    def test_strict_no_leakage_in_predictor_blocks(self):
        """Verifica que NINGÚN desenlace ni variable base (loneliness, anxiety_sleep, suicide_*) esté en B0, B1 o B2."""
        from phase5_model_training import get_feature_blocks
        b0_cols, b1_cols, b2_cols = get_feature_blocks()
        
        forbidden = [
            "suicide_ideation", "suicide_plan", "suicide_attempt", "suicide_attempt_n",
            "loneliness", "anxiety_sleep", "y1_ideation", "y2_plan", "y3_attempt",
            "y4_loneliness_frequent", "y5_worry_sleep_frequent", "y1_distress"
        ]
        
        for block_name, cols in [("B0", b0_cols), ("B1", b1_cols), ("B2", b2_cols)]:
            for f in forbidden:
                self.assertNotIn(f, cols, f"¡FUGA CRÍTICA! Variable prohibida '{f}' detectada en bloque predictor {block_name}")

    def test_country_disjunction_in_partitions(self):
        """Verifica que los conjuntos de países de desarrollo, validación y prueba sean estrictamente disjuntos."""
        from phase5_model_training import get_country_partitions
        dev_countries, val_countries, test_countries = get_country_partitions()
        
        set_dev = set(dev_countries)
        set_val = set(val_countries)
        set_test = set(test_countries)
        
        self.assertTrue(set_dev.isdisjoint(set_val), "Desarrollo y Validación comparten países")
        self.assertTrue(set_dev.isdisjoint(set_test), "Desarrollo y Test comparten países")
        self.assertTrue(set_val.isdisjoint(set_test), "Validación y Test comparten países")
        self.assertGreaterEqual(len(set_dev), 6, "Desarrollo debe tener al menos 6 países")
        self.assertGreaterEqual(len(set_test), 2, "Test debe tener al menos 2 países")

if __name__ == "__main__":
    unittest.main()
