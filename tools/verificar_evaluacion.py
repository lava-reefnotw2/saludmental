from evaluar_protocolo import active_results,threadpool_limits
from protocol_explanations import elaborate
from protocol_views import exports
from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest
run=active_results()
with threadpool_limits(limits=4):elaborate(run)
exports(run,run/'protocolo')
app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=90)
assert not app.exception,[e.message for e in app.exception]
print('UI OK',len(app.tabs))
