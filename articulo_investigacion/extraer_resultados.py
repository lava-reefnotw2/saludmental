from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.venv_training'))
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import numpy as np
from discussion_figures import comparison,VIEWS,observed,COUNTRIES
from protocol_views import external_figures
from publication_charts import png600
OUT=ROOT/'articulo_investigacion'
FIG=OUT/'figuras';FIG.mkdir(exist_ok=True)
run=ROOT/'entrenamiento_real/runs/20261117T222805Z_88089e7c'
folder=run/'protocolo'
def choice(mapping):
 def select(label,opts,**kwargs):return mapping.get(label,opts[0])
 return select
jobs=[]
for view,sel in [(VIEWS[0],{}),(VIEWS[1],{}),(VIEWS[4],{}),(VIEWS[3],{})]:
 fig,note=comparison(run,view,choice(sel));jobs.append((fig,note,view))
external=list(external_figures(folder,['ARG','BOL','PER','PRY','URY'],True))
jobs=[jobs[0],jobs[1],jobs[2],(*external[0],'PR cinco países'),(*external[2],'Calibración cinco países'),jobs[3]]
manifest=[]
for i,(fig,note,view) in enumerate(jobs,1):
 titles={1:'Información familiar y conductual frente a edad y sexo',2:'Comprensión parental e ideación autorreportada · Argentina',3:'Algoritmos frente a logística · ocho países',4:'Precisión y sensibilidad · cinco países',5:'Calibración externa · cinco países',6:'Brecha de falsos negativos · femenino menos masculino'}
 fig.update_layout(font=dict(family='Arial',size=16),legend=dict(font=dict(size=14)),title=dict(text=titles[i],font=dict(size=21)))
 if i==2:fig.update_xaxes(tickvals=['Nunca','Rara vez','A veces','La mayoría de las veces','Siempre'],ticktext=['Nunca','Rara vez','A veces','La mayoría<br>de las veces','Siempre'])
 if i==6:fig.update_xaxes(title_text='Diferencia de proporciones')
 if i==3:fig.update_layout(xaxis_title=None,yaxis_title=None)
 if i==5:fig.update_layout(legend=dict(x=0.02,xanchor='left',y=0.98,yanchor='top',bgcolor='rgba(255,255,255,0.85)'))
 path=FIG/f'Figura_{i:02}.png'
 path.write_bytes(png600(fig))
 (FIG/f'Figura_{i:02}.json').write_text(fig.to_json(),encoding='utf-8')
 manifest.append({'figura':i,'vista':view,'nota_original':note,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 print('FIGURA',i,note,flush=True)
(OUT/'figuras_procedencia.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['H1','metricas','equidad','explicacion','H4','estabilidad','sensibilidad_AP']:
 df=pd.read_csv(folder/(name+'.csv'))
 if name=='equidad':df=df[(df.subgroup=='sex_code')&df.metric.isin(['FNR','FPR'])]
 if name=='explicacion':df=df.groupby('variable')[['mean_abs_SHAP','permutation_delta_AP']].mean().sort_values('mean_abs_SHAP',ascending=False)
 if name=='H4':df=df.groupby('model').mean(numeric_only=True)
 if name=='estabilidad':df=df[df.model=='B1_catboost'].groupby('variable').agg(rank_median=('rank','median'),rank_min=('rank','min'),rank_max=('rank','max'))
 print('\n'+name+'\n'+df.to_string(index=True),flush=True)
print('FAMILIA\n',observed(str(run/'cohorte_gshs_8paises.csv'),0,'parental_understanding','ARG').to_string(index=False))
print('MISSING\n',pd.read_csv(run/'faltantes.csv').to_string(index=False))
print('RAW/AGE/Y1\n',pd.read_csv(run/'auditoria_cohorte.csv').sum(numeric_only=True).to_dict())
