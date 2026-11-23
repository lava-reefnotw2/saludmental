"""Exploratory nested Optuna studies. Each study is private to one outer fold."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
import real_data_training as rt
from protocol_analysis import scores

def optimise(run,family,block,budget,seed,progress,dest=None):
    d=pd.read_csv(run/'cohorte_gshs_8paises.csv')
    d=d[d.country_iso3.isin(rt.FIVE)].reset_index(drop=True)
    categorical=rt.CORE[:2] if block=='B0' else rt.FULL
    macro=[]
    if block=='B2':
        ctx=pd.read_csv(rt.METADATA/'WDI_cobertura_GSHS.csv').rename(columns={'year':'survey_year'})
        macro=list(ctx.columns.difference(['survey_year','country_iso3']))
        d=d.merge(ctx,on=['country_iso3','survey_year'],how='left',validate='many_to_one')
        if d[macro].isna().any().any():raise ValueError('Contexto país-año incompleto')
    cols=categorical+macro
    logs=[];results=[]
    def spec(params):
        estimator=rt.specs()[family][1]['model'].set_params(**params,random_state=seed) if family!='B1_catboost' else rt.CatBoostClassifier(iterations=180,verbose=False,allow_writing_files=False,thread_count=4,random_seed=seed,**params)
        prep=ColumnTransformer([('items',rt.pipe(estimator)[:-1],categorical)]+([('context',StandardScaler(),macro)] if macro else []))
        return cols,Pipeline([('prep',prep),('model',estimator)])
    for country in sorted(d.country_iso3.unique()):
        tr=d[d.country_iso3!=country].reset_index(drop=True);te=d[d.country_iso3==country]
        folds=list(GroupKFold(3).split(tr,groups=tr.country_iso3))
        storage='sqlite:///'+str((dest/'estudio.sqlite').resolve()) if dest else None
        study=optuna.create_study(study_name=country,storage=storage,direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed),pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
        def objective(trial):
            if family=='B1_logreg_C1': params={'C':trial.suggest_float('C',.01,10,log=True)}
            elif family=='B1_mlp':params={'alpha':trial.suggest_float('alpha',.01,1,log=True)}
            elif family=='B1_catboost':params={'depth':trial.suggest_int('depth',3,6)}
            else:params={'max_depth':trial.suggest_int('max_depth',3,8)}
            aps=[]
            for i,(a,b) in enumerate(folds):
                va=tr.iloc[b].reset_index(drop=True)
                _,p=rt.fit_predict(spec(params),tr.iloc[a],va)
                for key,rows in va.groupby('country_iso3').indices.items():
                    aps.append(average_precision_score(va.iloc[rows].y1_ideation,p[rows],sample_weight=va.iloc[rows].weight))
                trial.set_user_attr('fold_'+str(i),dict(validation=sorted(va.country_iso3.unique()),macro_AP=float(np.mean(aps[-va.country_iso3.nunique():]))))
                trial.report(np.mean(aps),i)
                if trial.should_prune():raise optuna.TrialPruned()
            return np.mean(aps)
        def log(study,trial):
            logs.append(dict(country=country,trial=trial.number,state=trial.state.name,objective=trial.value,parameters=json.dumps(trial.params),folds=json.dumps(trial.user_attrs),seconds=trial.duration.total_seconds()))
            if dest:pd.DataFrame(logs).to_csv(dest/'trials.csv',index=False)
            progress(country,trial.number+1,trial.state.name)
        study.optimize(objective,n_trials=budget,callbacks=[log])
        best=spec(study.best_params)
        _,inner,_=rt.inner_search(tr,{'selected':best})
        cal,th=rt.calibrate(tr,inner['selected'])
        _,raw=rt.fit_predict(best,tr,te)
        p=cal.predict_proba(rt.sigmoid(raw))[:,1]
        results.append(dict(country=country,**scores(te.y1_ideation.to_numpy(),p,te.weight.to_numpy(),th)))
    return pd.DataFrame(logs),pd.DataFrame(results)
