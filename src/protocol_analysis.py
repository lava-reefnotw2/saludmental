"""Country-held-out analyses for the common and family predictor blocks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import real_data_training as rt
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc, precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool
from threadpoolctl import threadpool_limits

def scores(y,p,w,t):
    pos=y==1
    call=p>=t
    tp,fn,fp,tn=[w[m].sum() for m in (pos&call,pos&~call,~pos&call,~pos&~call)]
    def div(a,b): return a/b if b else np.nan
    sens=div(tp,tp+fn)
    spec=div(tn,tn+fp)
    return dict(AP=ap(y,p,sample_weight=w),AUC=auc(y,p,sample_weight=w),Brier=np.average((p-y)**2,weights=w),
                sensitivity=sens,specificity=spec,FNR=1-sens,FPR=1-spec,PPV=div(tp,tp+fp),NPV=div(tn,tn+fn),
                F1=div(2*tp,2*tp+fp+fn),balanced_accuracy=(sens+spec)/2,
                calibration_bias=np.average(p-y,weights=w),prevalence=np.average(y,weights=w),
                TP_weight=tp,FN_weight=fn,FP_weight=fp,TN_weight=tn)

def write_tables(folder,tables):
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(folder/(name+'.csv'),index=False)

def decision_rows(predictions):
    rows = []
    for country, group in predictions.groupby('country_iso3'):
        weights = group.weight / group.weight.sum()
        y = group.y1_ideation
        for threshold in np.arange(.1, .41, .05):
            flag = group.p_calibrated >= threshold
            cost = threshold / (1 - threshold)
            benefit = weights * (y - cost * (1 - y))
            rows.append(dict(country=country, threshold=threshold, FP_cost_relative_FN=cost,
                             observed_prevalence=(weights * y).sum(), model=(benefit * flag).sum(),
                             all=benefit.sum(), none=0))
    return rows


def analyse_core(run):
    """Publish complete eight-country charts from the saved external predictions."""
    parent = run / 'protocolo'
    parent.mkdir(exist_ok=True)
    target = parent / 'nucleo_8'
    if not target.exists():
        with TemporaryDirectory(dir=parent, prefix='nucleo_8_') as staging:
            output = Path(staging)
            _analyse(run, output, core=True)
            output.rename(target)
    return target


def analyse(run):
    folder=run/'protocolo';folder.mkdir(exist_ok=True)
    _analyse(run, folder, core=False)
    return folder


def _analyse(run, folder, *, core):
    d=pd.read_csv(run/'cohorte_gshs_8paises.csv')
    features = rt.CORE if core else rt.FULL
    experiment = 'gshs8_core' if core else 'gshs5_familiar'
    predictions = pd.read_csv(run / experiment / 'predicciones_externas.csv')
    if core:
        expected_countries = {survey[:3] for survey in rt.FILES}
        if set(d.country_iso3) != expected_countries:
            raise ValueError('La cohorte del núcleo debe contener los ocho países GSHS.')
        full = predictions[predictions.model.eq('B1_logreg_C01')].set_index('record_key')
        baseline = predictions[predictions.model.eq('B0_logreg')].set_index('record_key')
        for model_predictions in (full, baseline):
            if not model_predictions.index.is_unique or set(model_predictions.index) != set(d.record_key):
                raise ValueError('Faltan predicciones externas del núcleo de ocho países.')
    else:
        d = d[d.country_iso3.isin(rt.FIVE)].reset_index(drop=True)
        full = predictions[predictions.selected].set_index('record_key')
    tables={k:[] for k in ['metricas','transporte','equidad','curvas','calibracion','explicacion','OR','secundarios','H1','sensibilidad_AP']}
    bootstrap=[]; pooled=[]
    for country,g in d.groupby('country_iso3'):
        if core:
            print(f'Comparaciones del núcleo de ocho países: {country}', flush=True)
        tr=d[d.country_iso3!=country].reset_index(drop=True)
        g=g.reset_index(drop=True)
        if core:
            base = baseline.loc[g.record_key]
            basal, bt = base.p_calibrated.to_numpy(), base.threshold.iloc[0]
        else:
            cols=rt.CORE[:2]
            est=rt.pipe(LogisticRegression(C=1,max_iter=600,random_state=42))
            _,oof,_=rt.inner_search(tr,{'basal':(cols,est)})
            bcal,bt=rt.calibrate(tr,oof['basal'])
            _,raw=rt.fit_predict((cols,est),tr,g)
            basal=bcal.predict_proba(rt.sigmoid(raw))[:,1]
        ext=full.loc[g.record_key]
        p=ext.p_calibrated.to_numpy(); t=ext.threshold.iloc[0]
        y=g.y1_ideation.to_numpy();w=g.weight.to_numpy()
        c_value=.1 if ext.model.iloc[0].endswith('C01') else 1.
        spec=(features,rt.pipe(LogisticRegression(C=c_value,max_iter=600,random_state=42)))
        _,inner,_=rt.inner_search(tr,{'familiar':spec})
        cal,_=rt.calibrate(tr,inner['familiar'])
        dev=cal.predict_proba(rt.sigmoid(inner['familiar']))[:,1]
        train_metrics=[scores(tr.iloc[ix].y1_ideation.to_numpy(),dev[ix],tr.iloc[ix].weight.to_numpy(),t) for ix in tr.groupby('country_iso3').indices.values()]
        for model,probs,th in [('B0',basal,bt),('B1',p,t)]:
            result=scores(y,probs,w,th)
            slope=LogisticRegression(C=1e6).fit(rt.sigmoid(probs),y,sample_weight=w/w.mean())
            tables['metricas'].append(dict(country=country,model=model,n=len(g),threshold=th,intercept=slope.intercept_[0],slope=slope.coef_[0,0],AP_unweighted=ap(y,probs),**result))
            if model=='B1':
                for key in ['AP','AUC','Brier','calibration_bias']:
                    ref=np.mean([row[key] for row in train_metrics])
                    tables['transporte'].append(dict(country=country,metric=key,development_oof=ref,external=result[key],external_minus_development=result[key]-ref))
                precision,recall,_=precision_recall_curve(y,probs,sample_weight=w)
                fpr,tpr,_=roc_curve(y,probs,sample_weight=w)
                for typ,x,v in [('PR',recall,precision),('ROC',fpr,tpr)]:
                    for i in np.unique(np.linspace(0,len(x)-1,min(201,len(x))).astype(int)):
                        tables['curvas'].append(dict(country=country,curve=typ,x=x[i],y=v[i]))
        rng=np.random.default_rng(42)
        keys,ix=np.unique(g[['stratum','psu']].to_numpy(),axis=0,return_inverse=True)
        ix=ix.ravel()
        strata=[np.where(keys[:,0]==s)[0] for s in np.unique(keys[:,0])]
        draws=[]
        for _ in range(500):
            freq=np.bincount(np.concatenate([rng.choice(a,len(a),replace=True) for a in strata]),minlength=len(keys))
            draws.append(w*freq[ix])
        dy=np.array([ap(y,p,sample_weight=v)-ap(y,basal,sample_weight=v) for v in draws])
        delta=ap(y,p,sample_weight=w)-ap(y,basal,sample_weight=w)
        bootstrap.append((delta,dy))
        tables['H1'].append(dict(country=country,delta_AP=delta,low=np.quantile(dy,.025),high=np.quantile(dy,.975)))
        for lo in np.arange(10)/10:
            m=(p>=lo)&(p<(lo+.1) if lo<.9 else p<=1)
            if m.sum()<20:continue
            rates=[np.average(y[m],weights=v[m]) for v in draws if v[m].sum()>0]
            tables['calibracion'].append(dict(country=country,lower=lo,n=m.sum(),predicted=np.average(p[m],weights=w[m]),observed=np.average(y[m],weights=w[m]),low=np.quantile(rates,.025),high=np.quantile(rates,.975)))
        for field,ref,other in [('sex_code',1,2),('age_years',13,14),('age_years',13,15)]:
            masks=[g[field].to_numpy()==a for a in [ref,other]]
            if any(min(y[m].sum(),(1-y[m]).sum())<10 for m in masks):continue
            def differences(v):
                vals=[]
                for m in masks:
                    if any(v[m&(y==k)].sum()==0 for k in [0,1]):return None
                    pred=p[m]>=t; yy=y[m]; ww=v[m]
                    sens=np.average(pred[yy==1],weights=ww[yy==1])
                    vals.append(np.array([sens,1-sens,np.average(pred[yy==0],weights=ww[yy==0]),np.average(p[m]-yy,weights=ww),ap(yy,p[m],sample_weight=ww)]))
                return vals[1]-vals[0]
            dif=differences(w); boot=np.array([v for weights in draws if (v:=differences(weights)) is not None])
            for j,key in enumerate(['sensitivity','FNR','FPR','calibration_bias','AP']):
                tables['equidad'].append(dict(country=country,subgroup=field,reference=ref,compared=other,n_ref=masks[0].sum(),n_comp=masks[1].sum(),metric=key,difference=dif[j],low=np.quantile(boot[:,j],.025),high=np.quantile(boot[:,j],.975)))
        if not core:
            model,_=rt.fit_predict(spec,tr,g)
            # Pair each category with the first observed category of its item (conditional odds ratio).
            offset=0
            for col,categories in zip(rt.FULL,model['encode'].categories_):
                coef=model['model'].coef_[0,offset:offset+len(categories)];offset+=len(categories)
                refidx=next(i for i,v in enumerate(categories) if v!=-99)
                for i,v in enumerate(categories):
                    tables['OR'].append(dict(country_held_out=country,variable=col,category=v,reference=categories[refidx],adjusted_OR=np.exp(coef[i]-coef[refidx])))
        tree=rt.specs()['B1_catboost']
        fitted,_=rt.fit_predict(tree,tr,g)
        sample=g.sample(min(512,len(g)),random_state=42)
        transformed=fitted[:-1].transform(sample[tree[0]])
        shap=fitted['model'].get_feature_importance(Pool(transformed),type='ShapValues')[:,:-1]
        offset=0
        for col,cats in zip(tree[0],fitted['encode'].categories_):
            local=shap[:,offset:offset+len(cats)].sum(axis=1);offset+=len(cats)
            perm=sample[tree[0]].copy();perm[col]=rng.permutation(perm[col])
            target=sample.y1_ideation
            base=ap(target,fitted.predict_proba(sample[tree[0]])[:,1],sample_weight=sample.weight)
            tables['explicacion'].append(dict(country=country,variable=col,mean_abs_SHAP=np.average(abs(local),weights=sample.weight),permutation_delta_AP=base-ap(target,fitted.predict_proba(perm)[:,1],sample_weight=sample.weight),n=len(sample)))
        pooled.append((y,p,w/w.sum()))
        for col in [c for c in d if c.startswith(('y2_','y3_','y4_','y5_'))]:
            tables['secundarios'].append(dict(country=country,outcome=col,observed=g[col].count(),events=g[col].sum(),missing=g[col].isna().sum()))
    means=np.mean([x[1] for x in bootstrap],axis=0)
    macro_label = 'Macro · ocho países' if core else 'Macro · cinco países fijos'
    tables['H1'].append(dict(country=macro_label,delta_AP=np.mean([x[0] for x in bootstrap]),low=np.quantile(means,.025),high=np.quantile(means,.975)))
    yy,pp,ww=[np.concatenate(x) for x in zip(*pooled)]
    tables['sensibilidad_AP'].append(dict(pooled_AP_equal_country_weight=ap(yy,pp,sample_weight=ww),macro_AP_unweighted=np.mean([r['AP_unweighted'] for r in tables['metricas'] if r['model']=='B1'])))
    if core:
        del tables['OR']
        tables['decision_exploratoria'] = decision_rows(full)
        for name in ('H1', 'transporte', 'equidad', 'curvas', 'calibracion', 'explicacion', 'metricas', 'decision_exploratoria'):
            countries = {row['country'] for row in tables[name] if row['country'] in expected_countries}
            if countries != expected_countries:
                raise ValueError(f'{name}: faltan resultados de {sorted(expected_countries - countries)}.')
    write_tables(folder,tables)
