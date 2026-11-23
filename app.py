# -*- coding: utf-8 -*-
"""P07: entrypoint for the empirical GSHS dashboard."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
if (ROOT/'.venv_training').exists():sys.path.insert(0,str(ROOT/'.venv_training'))
sys.path.insert(0,str(ROOT/'src'))
from real_dashboard import render
render()
