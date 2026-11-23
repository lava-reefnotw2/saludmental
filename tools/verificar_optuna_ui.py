from evaluar_protocolo import active_results
from pathlib import Path
from streamlit.testing.v1 import AppTest
import json,pandas as pd
run=active_results();before=set((run/'protocolo'/'optuna').iterdir())
app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=120)
for n in app.number_input:
    if n.label=='Ensayos por país externo':n.set_value(1)
next(b for b in app.button if b.label=='Iniciar estudio anidado').click().run(timeout=120)
assert not app.exception,[e.message for e in app.exception]
dest,=(set((run/'protocolo'/'optuna').iterdir())-before)
assert len(pd.read_csv(dest/'trials.csv'))==5
assert len(pd.read_csv(dest/'external.csv'))==5
assert (dest/'estudio.sqlite').stat().st_size>0
print('Botón Optuna verificado: cinco estudios externos, cinco ensayos y registro persistente.')
