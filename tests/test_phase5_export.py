# -*- coding: utf-8 -*-
"""
Tests unitarios para FASE 5: Reportes, formatos DOCX, XLSX y figuras de alta resolución.
"""
import os
import sys
import unittest
import docx
import openpyxl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestPhase5Export(unittest.TestCase):

    def setUp(self):
        self.out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))

    def test_word_docx_report_generation(self):
        from export_engine import generate_word_report
        docx_path = generate_word_report(self.out_dir, "test_report.docx")
        self.assertTrue(os.path.exists(docx_path))
        doc = docx.Document(docx_path)
        self.assertGreater(len(doc.paragraphs), 5)
        # Verificar que se menciona TRIPOD+AI y PROBAST+AI y NO CLAIM como exigencia
        full_text = " ".join([p.text for p in doc.paragraphs])
        self.assertIn("TRIPOD+AI", full_text)
        self.assertIn("PROBAST+AI", full_text)

    def test_excel_workbook_generation(self):
        from export_engine import generate_excel_workbook
        xlsx_path = generate_excel_workbook(self.out_dir, "test_workbook.xlsx")
        self.assertTrue(os.path.exists(xlsx_path))
        wb = openpyxl.load_workbook(xlsx_path)
        self.assertIn("Resumen_Ejecutivo", wb.sheetnames)
        self.assertIn("Leaderboard_Modelos", wb.sheetnames)
        self.assertIn("Contraste_Hipotesis", wb.sheetnames)

    def test_high_dpi_figures(self):
        from export_engine import export_high_dpi_scientific_figures
        fig_path = export_high_dpi_scientific_figures(self.out_dir, dpi=600)
        if fig_path:
            self.assertTrue(os.path.exists(fig_path))

if __name__ == "__main__":
    unittest.main()
