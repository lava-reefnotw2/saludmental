# -*- coding: utf-8 -*-
"""
Tests unitarios y de humo para FASE 0: Arquitectura, dependencias y preparación.
"""
import os
import sys
import unittest

# Asegurar que src está en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestPhase0Architecture(unittest.TestCase):

    def test_python_version_and_core_dependencies(self):
        self.assertGreaterEqual(sys.version_info, (3, 10), "Se requiere Python 3.10+")
        import streamlit
        import sklearn
        import xgboost
        import catboost
        import shap
        import optuna
        import reportlab
        import docx
        import openpyxl
        import pandas
        import numpy
        self.assertTrue(streamlit.__version__)
        self.assertTrue(sklearn.__version__)
        self.assertTrue(xgboost.__version__)
        self.assertTrue(catboost.__version__)

    def test_directory_structure(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.assertTrue(os.path.isdir(os.path.join(root, "src")), "Directorio src debe existir")
        self.assertTrue(os.path.isdir(os.path.join(root, "data")), "Directorio data debe existir")
        self.assertTrue(os.path.isdir(os.path.join(root, "artifacts")), "Directorio artifacts debe existir")
        self.assertTrue(os.path.isdir(os.path.join(root, "static")), "Directorio static debe existir")
        self.assertTrue(os.path.isfile(os.path.join(root, "app.py")), "app.py debe existir en la raíz")
        self.assertTrue(os.path.isfile(os.path.join(root, "pipeline_sp07.py")), "pipeline_sp07.py debe existir en la raíz")

    def test_database_and_fallback(self):
        from app import get_db_engine, query_db
        engine = get_db_engine()
        df = query_db("SELECT 1 AS test_col;")
        self.assertIsNotNone(df, "query_db debe retornar un DataFrame válido")

if __name__ == "__main__":
    unittest.main()
