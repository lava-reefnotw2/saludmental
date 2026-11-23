from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'.venv_training'),str(root/'src')]
from training_jobs import active_results,atomic_json
from protocol_views import exports,provenance
from protocol_optuna import optimise
from sklearn.metrics import average_precision_score
import pandas as pd,numpy as np
from threadpoolctl import threadpool_limits
def extend(run):
    folder=run/'protocolo'
    preds=pd.read_csv(run/'gshs8_core/predicciones_externas.csv')
    comparisons=[]
    for c,g in preds.groupby('country_iso3'):
        a=g.pivot(index='record_key',columns='model',values='p_calibrated')
        info=g.drop_duplicates('record_key').set_index('record_key').loc[a.index]
        # Fixed equal weights, all members predicted with country held out.
        members=['B1_random_forest','B1_xgboost','B1_catboost','B1_mlp','B1_logreg_C01']
        a['ensamble_media']=a[members].mean(axis=1)
        ap=lambda p:average_precision_score(info.y1_ideation,p,sample_weight=info.weight)
        for m in members+['ensamble_media']:
            comparisons.append(dict(country=c,model=m,n=len(a),AP=ap(a[m]),delta_vs_logistica=ap(a[m])-ap(a['B1_logreg_C01'])))
    pd.DataFrame(comparisons).to_csv(folder/'H4.csv',index=False)
if __name__=='__main__':
    run=active_results();folder=run/'protocolo';extend(run)
    with threadpool_limits(limits=4):
        trial,ext=optimise(run,'B1_logreg_C1','B2',2,42,lambda *a: print(a,flush=True))
    dest=folder/'optuna'/'auditoria_B2_logistica';dest.mkdir(parents=True,exist_ok=True)
    trial.to_csv(dest/'trials.csv',index=False);ext.to_csv(dest/'external.csv',index=False)
    atomic_json(dest/'config.json',dict(family='B1_logreg_C1',block='B2',budget=2,seed=42))
    exports(run,folder);atomic_json(folder/'procedencia.json',provenance(run))

