"""Execute the Streamlit app to catch rendering errors, beyond an HTTP health check."""
from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'.venv_training'))
sys.path.insert(0,str(ROOT/'src'))
from streamlit.testing.v1 import AppTest
app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run(timeout=60)
errors=[str(e.message) for e in app.exception]
print(json.dumps(dict(exceptions=errors,tabs=len(app.tabs)),ensure_ascii=False))
if errors:sys.exit(1)
