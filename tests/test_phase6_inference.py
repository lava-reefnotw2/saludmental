# -*- coding: utf-8 -*-
"""
Tests unitarios para FASE 6: Inferencia Restringida de Investigación (Sandbox).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestPhase6Inference(unittest.TestCase):

    def setUp(self):
        self.out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))

    def test_schema_rejection_of_invalid_age(self):
        from restricted_inference import validate_inference_inputs, get_predefined_synthetic_cases
        cases = get_predefined_synthetic_cases()
        c = dict(list(cases.values())[0])
        
        # Edad menor a 11
        c["age_years"] = 9
        _, errors = validate_inference_inputs(c)
        self.assertTrue(any("edad" in e.lower() or "age_years" in e for e in errors), "Debe rechazar edad < 11")
        
        # Edad mayor a 17
        c["age_years"] = 20
        _, errors = validate_inference_inputs(c)
        self.assertTrue(any("edad" in e.lower() or "age_years" in e for e in errors), "Debe rechazar edad > 17")

    def test_synthetic_profile_inference_execution(self):
        from restricted_inference import get_predefined_synthetic_cases, run_research_inference
        cases = get_predefined_synthetic_cases()
        c = list(cases.values())[0]
        
        res = run_research_inference("catboost_b2", c, out_dir=self.out_dir)
        self.assertTrue(res.get("success"), f"Inferencia falló: {res.get('errors')}")
        self.assertIn("predicted_probability", res)
        self.assertIn("ci_95_approx", res)
        self.assertIn("descriptive_tier", res)
        self.assertIn("disclaimer", res)
        
        # Comprobar que no hay etiquetas de diagnóstico médico
        self.assertNotIn("diagnóstico", res["descriptive_tier"].lower())

if __name__ == "__main__":
    unittest.main()
