from evaluar_protocolo import active_results,threadpool_limits
from protocol_sensitivity import supplemental
from protocol_views import exports,provenance
from training_jobs import atomic_json
from pathlib import Path
from PIL import Image
from docx import Document
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest
import pandas as pd
run=active_results();folder=run/'protocolo'
with threadpool_limits(limits=4):supplemental(run)
exports(run,folder);atomic_json(folder/'procedencia.json',provenance(run))
assert len(pd.read_csv(folder/'resultados_secundarios.csv'))==20
assert len(pd.read_csv(folder/'perfil_sintetico.csv'))==7
assert len(pd.read_csv(folder/'H1.csv'))==6
assert len(pd.read_csv(folder/'equidad.csv'))==75
assert len(pd.read_csv(folder/'estabilidad.csv'))==225
assert len(pd.read_csv(folder/'interacciones.csv'))==180
assert (folder/'evaluacion.pdf').read_bytes().startswith(b'%PDF')
assert Document(folder/'evaluacion.docx').inline_shapes
assert 'H1' in load_workbook(folder/'evaluacion.xlsx',read_only=True).sheetnames
assert Image.open(folder/'H1.png').info['dpi'][0]>599
app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=90)
assert not app.exception,[e.message for e in app.exception]
next(b for b in app.button if 'perfil sintético' in b.label).click().run(timeout=90)
assert not app.exception,[e.message for e in app.exception]
print('Verificados H1, H3, H4, H5, secundarios, perfil sintético, descargas y renderizado.')
