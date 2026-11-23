"""Empirical GSHS training. Never calls the synthetic ingestion or writes old artifacts.

Run from repository root: python src/real_data_training.py
The optional local dependency directory is installed by tools/bootstrap_training.ps1.
"""
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / '.venv_training').exists(): sys.path.insert(0, str(ROOT / '.venv_training'))
os.environ.setdefault('OMP_NUM_THREADS', '4')
import json, hashlib, platform, time, warnings
import numpy as np
import pandas as pd
import sklearn, joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

OUT = Path(os.environ.get('SP07_TRAINING_OUT', ROOT / 'entrenamiento_real'))
METADATA = ROOT / 'entrenamiento_real' / 'metadata'
CORE = ['age_years','sex_code','food_insecurity','physical_attack','physical_fight',
        'close_friends','alcohol_30d','physical_activity','sedentary_time']
FULL = CORE + ['truancy','peer_kindness','parental_supervision','parental_understanding','parental_knowledge']
RAW_MAP = {'q2':'sex_code','q6':'food_insecurity','q15':'physical_attack','q16':'physical_fight',
           'q27':'close_friends','q35':'alcohol_30d','q49':'physical_activity','q52':'sedentary_time',
           'q53':'truancy','q54':'peer_kindness','q55':'parental_supervision','q56':'parental_understanding','q57':'parental_knowledge'}
FILES = {'ARG2018':'Argentina.csv','BOL2018':'BoliviaNacional.csv','GTM2015':'Guatemala.csv',
         'HND2012':'Honduras.csv','PAN2018':'Panama.csv','PRY2017':'Paraguay.csv','PER2010':'Peru.csv','URY2019':'Uruguay.csv'}
FIVE = ['ARG','BOL','PRY','PER','URY']

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    OUT.mkdir(exist_ok=True)
    frames, audit, mapping, hashes = [], [], [], []
    for tag, fn in FILES.items():
        p = ROOT/'datasets'/fn
        d = pd.read_csv(p); d.columns = d.columns.str.lower()
        dic = pd.read_csv(METADATA/f'{tag}_diccionario.csv')
        dic.variable = dic.variable.str.lower()
        dic = dic.drop_duplicates('variable').set_index('variable')
        required = list(RAW_MAP) + ['q1','q22','q23','q24','q25','q26','weight','stratum','psu']
        if tag == 'GTM2015': required = [q for q in required if q not in ['q53','q54','q55','q56','q57']]
        absent = set(required) - set(d.columns)
        if absent: raise ValueError(f'{tag}: faltan columnas requeridas: {sorted(absent)}')
        domains = {'q1':8,'q2':2,'q6':5,'q15':8,'q16':8,'q22':5,'q23':5,'q24':2,'q25':2,'q26':5,'q27':4,'q35':7,'q49':8,'q52':6,'q53':5,'q54':5,'q55':5,'q56':5,'q57':5}
        for q, upper in domains.items():
            if q in d and not d[q].dropna().isin(range(1,upper+1)).all():
                raise ValueError(f'{tag}/{q}: códigos fuera del dominio documentado')
        z = pd.DataFrame(index=d.index)
        z['country_iso3'], z['survey_id'], z['survey_year'] = tag[:3], tag, int(tag[3:])
        z['record_key'] = [f'{tag}_{i+1:06}' for i in range(len(d))]
        z['age_years'] = d.q1.map({3:13,4:14,5:15})
        for src, name in RAW_MAP.items():
            z[name] = d[src] if src in d else np.nan
            if src in d:
                if src not in dic.index: raise ValueError(f'No dictionary: {tag}/{src}')
                row = dic.loc[src]
                mapping.append(dict(survey_id=tag,source=src,canonical=name,label=row['label'],question=row['question'],categories=row['categories']))
        z['y1_ideation'] = d.q24.map({1:1,2:0})
        z['y2_plan'] = d.q25.map({1:1,2:0})
        z['y3_attempt'] = d.q26.map({1:0,2:1,3:1,4:1,5:1})
        for q, name in [('q22','y4_loneliness'),('q23','y5_worry_sleep')]:
            z[name] = d[q].map({1:0,2:0,3:0,4:1,5:1})
        for c in ['weight','stratum','psu']: z[c] = d[c]
        eligible = z.age_years.notna() & z.y1_ideation.notna()
        if not (z.loc[eligible,'weight'].gt(0) & np.isfinite(z.loc[eligible,'weight'])).all():
            raise ValueError(f'Invalid weights: {tag}')
        if z.loc[eligible,['psu','stratum']].isna().any().any(): raise ValueError('Missing survey design')
        audit.append(dict(survey_id=tag,n_raw=len(d),n_eligible=int(eligible.sum()),events=int(z.loc[eligible,'y1_ideation'].sum()),excluded_age=int(z.age_years.isna().sum()),excluded_y1_after_age=int((z.age_years.notna()&z.y1_ideation.isna()).sum()),structural_missing=','.join(c for c in FULL if z[c].isna().all())))
        frames.append(z.loc[eligible].copy())
        hashes.append(dict(survey_id=tag,file=str(p.relative_to(ROOT)),sha256=digest(p),dictionary_sha256=digest(METADATA/f'{tag}_diccionario.csv')))
    df = pd.concat(frames,ignore_index=True)
    assert df.record_key.is_unique
    df.to_csv(OUT/'cohorte_gshs_8paises.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'auditoria_cohorte.csv',index=False)
    pd.DataFrame(mapping).to_csv(OUT/'armonizacion.csv',index=False)
    (OUT/'fuentes_huellas.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    df.groupby('country_iso3')[FULL].agg(lambda x:x.isna().mean()).to_csv(OUT/'faltantes.csv')
    return df

def weights(d):
    """Equal country mass; relative survey weights within each country. Mean one."""
    w = d.weight / d.groupby('country_iso3').weight.transform('sum')
    return (w / w.mean()).to_numpy()

def pipe(model):
    # All answers are treated categorically; missingness is explicit, no fake ordinal distances.
    return Pipeline([('missing',SimpleImputer(strategy='constant',fill_value=-99,keep_empty_features=True)),
                     ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False)),('model',model)])

def specs():
    return {
      'B0_logreg': (CORE[:2], pipe(LogisticRegression(C=1,max_iter=600,random_state=42))),
      'B1_logreg_C01': (CORE,pipe(LogisticRegression(C=.1,max_iter=600,random_state=42))),
      'B1_logreg_C1': (CORE,pipe(LogisticRegression(C=1,max_iter=600,random_state=42))),
      'B1_random_forest': (CORE,pipe(RandomForestClassifier(n_estimators=120,max_depth=9,min_samples_leaf=30,n_jobs=4,random_state=42))),
      'B1_xgboost': (CORE,pipe(XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.85,colsample_bytree=.85,reg_lambda=5,n_jobs=4,random_state=42,eval_metric='logloss'))),
      'B1_catboost': (CORE,pipe(CatBoostClassifier(iterations=180,depth=4,learning_rate=.05,l2_leaf_reg=5,thread_count=4,random_seed=42,verbose=False,allow_writing_files=False))),
      'B1_mlp': (CORE,pipe(MLPClassifier(hidden_layer_sizes=(32,16),alpha=.1,max_iter=100,early_stopping=False,random_state=42))),
    }

def fit_predict(spec, tr, te):
    cols, estimator = spec
    assert not set(cols) & {'country_iso3','survey_id','survey_year','weight','psu','stratum','record_key','y1_ideation','y2_plan','y3_attempt','y4_loneliness','y5_worry_sleep'}
    p = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        p.fit(tr[cols],tr.y1_ideation.astype(int),model__sample_weight=weights(tr))
    return p, p.predict_proba(te[cols])[:,1]

def metrics(d,p,threshold=.2):
    y=d.y1_ideation.astype(int).to_numpy(); w=weights(d)
    tn,fp,fn,tp = confusion_matrix(y,p>=threshold,sample_weight=w,labels=[0,1]).ravel()
    return dict(n=len(d),events=int(y.sum()),prevalence=float(np.average(y,weights=w)),ap=float(average_precision_score(y,p,sample_weight=w)),roc_auc=float(roc_auc_score(y,p,sample_weight=w)),brier=float(brier_score_loss(y,p,sample_weight=w)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),ppv=float(tp/(tp+fp)) if tp+fp else 0.,threshold=float(threshold))

def sigmoid(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)).reshape(-1,1)

def inner_search(d, candidates):
    folds=list(GroupKFold(n_splits=min(3,d.country_iso3.nunique())).split(d,groups=d.country_iso3))
    predictions, scores={},{}
    for name,spec in candidates.items():
        p=np.full(len(d),np.nan)
        for tr,va in folds:
            assert set(d.iloc[tr].country_iso3).isdisjoint(d.iloc[va].country_iso3)
            _,p[va]=fit_predict(spec,d.iloc[tr],d.iloc[va])
        predictions[name]=p
        # Macro AP: Argentina cannot dominate model selection through its row count.
        scores[name]=float(np.mean([average_precision_score(d.iloc[idx].y1_ideation,p[idx],sample_weight=d.iloc[idx].weight) for idx in d.groupby('country_iso3').indices.values()]))
    best=max(scores,key=scores.get)
    return best,predictions,scores

def calibrate(d,p):
    cal=LogisticRegression(C=1e6,max_iter=500).fit(sigmoid(p),d.y1_ideation,sample_weight=weights(d))
    cp=cal.predict_proba(sigmoid(p))[:,1]
    # Prespecified FN:FP cost 3:1. Selected exclusively on inner OOF.
    y=d.y1_ideation.to_numpy(); w=weights(d)
    grid=np.linspace(.05,.60,56)
    costs=[np.sum(w*(3*((cp<t)&(y==1))+((cp>=t)&(y==0)))) for t in grid]
    return cal,float(grid[np.argmin(costs)])

def experiment(d,name,candidates):
    dest=OUT/name;dest.mkdir(exist_ok=True)
    rows,preds,selections=[],[],[]
    for tr,te in LeaveOneGroupOut().split(d,groups=d.country_iso3):
        train,test=d.iloc[tr].reset_index(drop=True),d.iloc[te].reset_index(drop=True)
        country=test.country_iso3.iloc[0]
        print(f'{name}: país externo {country}; train={len(train)}, test={len(test)}',flush=True)
        best,inner,scores=inner_search(train,candidates)
        selections.append(dict(test_country=country,train_countries=sorted(train.country_iso3.unique()),selected=best,inner_macro_ap=scores))
        for model,spec in candidates.items():
            _,raw=fit_predict(spec,train,test)
            cal,threshold=calibrate(train,inner[model]); p=cal.predict_proba(sigmoid(raw))[:,1]
            m=metrics(test,p,threshold);m.update(country=country,model=model,raw_ap=metrics(test,raw)['ap'],selected=(model==best))
            rows.append(m)
            q=test[['record_key','country_iso3','survey_id','stratum','psu','weight','sex_code','age_years','y1_ideation']].copy()
            q['model'],q['p_raw'],q['p_calibrated'],q['threshold'],q['selected']=model,raw,p,threshold,model==best
            preds.append(q)
        pd.DataFrame(rows).to_csv(dest/'metricas_por_pais.csv',index=False)
        (dest/'seleccion_interna.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    pred=pd.concat(preds,ignore_index=True);pred.to_csv(dest/'predicciones_externas.csv',index=False)
    table=pd.DataFrame(rows)
    summary=table.groupby('model')[['ap','roc_auc','brier','sensitivity','specificity']].mean().reset_index()
    sel=table[table.selected].select_dtypes(include='number').mean().to_dict();sel['model']='seleccion_anidada'
    summary=pd.concat([summary,pd.DataFrame([sel])],ignore_index=True)
    summary.to_csv(dest/'resumen_macro.csv',index=False)
    return pred

def main(d=None):
    t=time.time()
    if d is None: d=prepare()
    candidates=specs()
    recipe=dict(seed=42,primary='y1_ideation',age=[13,14,15],features=CORE,full_features=FULL,
                training_weights='equal country mass, survey weights within country',selection='inner GroupKFold 3, macro country AP',outer='LeaveOneGroupOut country',
                calibration='sigmoid fitted only to selected inner OOF scores',threshold='inner OOF FN:FP 3:1, grid .05:.60',
                models={n:str(s[1]) for n,s in candidates.items()},versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__))
    (OUT/'receta_predefinida.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    d5=d[d.country_iso3.isin(FIVE)].reset_index(drop=True)
    # Controlled data comparison: same held-out countries, same variables, same algorithm search.
    logistic={k:v for k,v in candidates.items() if k.startswith('B1_logreg')}
    for frame,name,sp in [(d5,'gshs5_core',logistic),(d5,'gshs5_familiar',{k:(FULL,v[1]) for k,v in logistic.items()}),(d,'gshs8_core',candidates)]:
        experiment(frame,name,sp)
    # Final recipe is selected with internal grouped CV on all countries; outer scores do not select it.
    best,inner,scores=inner_search(d,candidates)
    cal,threshold=calibrate(d,inner[best]); final,_=fit_predict(candidates[best],d,d.iloc[:1])
    joblib.dump(dict(pipeline=final,calibrator=cal,threshold=threshold,features=candidates[best][0],target='y1_ideation',selected=best,source='empirical GSHS eight countries',research_only=True),OUT/'modelo_investigacion.joblib')
    (OUT/'ajuste_final.json').write_text(json.dumps(dict(selected=best,inner_macro_ap=scores,threshold=threshold,seconds=time.time()-t),indent=2),encoding='utf-8')
    print('ENTRENAMIENTO TERMINADO',best,round(time.time()-t,1),'segundos',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4): main()
