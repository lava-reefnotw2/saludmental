from pathlib import Path
import csv
import json
import sys

out = Path(__file__).resolve().parent
root = out.parents[1]
sys.path[:0] = [str(root / "src"), str(root / ".venv_training")]
from publication_charts import png600
from discussion_figures import COUNTRIES, VIEWS, comparison
from protocol_views import external_figures
from PIL import Image

run = root / "entrenamiento_real/runs" / json.loads((root / "entrenamiento_real/active.json").read_text())["run_id"]
assert json.loads((run / "status.json").read_text())["state"] == "completed"
folder = run / "protocolo/nucleo_8"
def first(label, options, **kwargs):
    return options[0]

qa = out / "control"
qa.mkdir(exist_ok=True)
exports = out / "Figuras_600dpi"
exports.mkdir(exist_ok=True)
data = {}
for name in ("metricas", "equidad", "explicacion", "H1", "transporte", "sensibilidad_AP"):
    with (folder / (name + ".csv")).open(encoding="utf-8-sig", newline="") as f:
        data[name] = list(csv.DictReader(f))
(qa / "agregados.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
external = list(external_figures(folder, list(COUNTRIES), True))
jobs = [comparison(run,VIEWS[k],first,core=True) for k in (0,2,4)] + [external[0], external[2], comparison(run,VIEWS[3],first,core=True)]
colors = dict(zip(COUNTRIES.values(), ["#0072B2","#D55E00","#00856A","#A44990","#6857B7","#965B18","#167B87","#454545"]))
titles = ["Ganancia de precisión promedio frente a edad y sexo","Precisión promedio en desarrollo y evaluación externa","Diferencia de precisión promedio frente a la logística","Precisión y sensibilidad en el país excluido","Probabilidad predicha y frecuencia observada","Diferencia de falsos negativos por sexo"]
for n,(fig,_) in enumerate(jobs,1):
    fig.update_layout(template="plotly_white",title=dict(text=titles[n-1],font=dict(size=24)),font=dict(family="Arial",size=19),legend=dict(font=dict(size=17)))
    if n in (2,4,5):
        assert {trace.name for trace in fig.data} == set(colors)
        for trace in fig.data:
            trace.update(marker_color=colors[trace.name],line_color=colors[trace.name])
            if n==5: trace.update(error_y_color=colors[trace.name])
    if n==2:
        fig.update_yaxes(title_text="Precisión promedio")
        fig.update_xaxes(title_text=None,tickvals=["Desarrollo (OOF)","País excluido"],ticktext=["Desarrollo (validación interna)","País excluido"])
    if n==3:
        fig.update_xaxes(title_text=None)
        fig.update_yaxes(title_text=None)
        fig.update_traces(textfont_size=19)
    if n==1: fig.update_xaxes(title_text="Diferencia de precisión promedio")
    if n==6: fig.update_xaxes(title_text="Tasa femenina menos tasa masculina")
    dest=exports / f"Figura_{n:02}.png"
    dest.write_bytes(png600(fig))
    with Image.open(dest) as im:
        assert im.size==(3840,2320) and all(abs(v-600)<.01 for v in im.info["dpi"])
    print(dest.name, flush=True)
