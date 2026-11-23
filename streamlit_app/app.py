# -*- coding: utf-8 -*-
"""Alternate Streamlit entrypoint for the empirical P07 dashboard."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.venv_training').exists():sys.path.insert(0,str(ROOT/'.venv_training'))
sys.path.insert(0,str(ROOT/'src'))
from real_dashboard import render
render()
