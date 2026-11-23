"""Start the existing Streamlit dashboard on loopback using local dependencies."""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if (ROOT / '.venv_training').exists():
    sys.path.insert(0, str(ROOT / '.venv_training'))
from streamlit.web import cli
sys.argv = ['streamlit','run',str(ROOT/'app.py'),'--server.address=127.0.0.1',
            '--server.port=8501','--server.headless=true','--browser.gatherUsageStats=false',
            '--global.developmentMode=false']
cli.main()
