"""Extended family-feature analysis, paired uncertainty and reproducibility checks."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import real_data_training as rt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

def selected_predictions(folder, logistic_only=False):
    d=pd.read_csv(folder/'predicciones_externas.csv')
    if not logistic_only:return d[d.selected].copy()
    sels=json.loads((folder/'seleccion_interna.json').read_text())
    picks={s['test_country']:max({k:v for k,v in s['inner_macro_ap'].items() if k.startswith('B1_logreg')},key=lambda k:s['inner_macro_ap'][k]) for s in sels}
    return d[d.apply(lambda r:r.model==picks[r.country_iso3],axis=1)].copy()

def weighted_ap(y,p,w):
    # Exact weighted AP with tied scores aggregated, like sklearn, much faster for bootstrap.
    order=np.argsort(-p,kind='stable'); y,p,w=y[order],p[order],w[order]
    ends=np.r_[np.flatnonzero(np.diff(p)),len(p)-1]
    tp=np.cumsum(w*y)[ends]; total=np.cumsum(w)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])*np.divide(tp,total,out=np.zeros_like(tp),where=total>0))/tp[-1])

def compare(a,b,title):
    d=a.merge(b[['record_key','p_calibrated']],on='record_key',suffixes=('_a','_b'),validate='one_to_one')
    rng=np.random.default_rng(42);rows=[];draws=[]
    for country,g in d.groupby('country_iso3'):
        g=g.reset_index(drop=True)
        y=g.y1_ideation.to_numpy();pa=g.p_calibrated_a.to_numpy();pb=g.p_calibrated_b.to_numpy();w=g.weight.to_numpy()
        key=pd.MultiIndex.from_frame(g[['stratum','psu']]);codes,keys=pd.factorize(key)
        strata=np.array([k[0] for k in keys]);groups=[np.flatnonzero(strata==s) for s in np.unique(strata)]
        bs=[]
        for _ in range(500):
            multiplicity=np.zeros(len(keys))
            for cluster in groups:
                picks=rng.choice(cluster,len(cluster),replace=True)
                multiplicity+=np.bincount(picks,minlength=len(keys))
            wb=w*multiplicity[codes]
            bs.append(weighted_ap(y,pb,wb)-weighted_ap(y,pa,wb))
        delta=weighted_ap(y,pb,w)-weighted_ap(y,pa,w)
        draws.append(bs)
        rows.append(dict(comparison=title,country=country,n=len(g),ap_a=weighted_ap(y,pa,w),ap_b=weighted_ap(y,pb,w),delta=delta,ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975),psus=len(keys),singleton_strata=sum(len(x)==1 for x in groups)))
    boot=np.mean(draws,axis=0)
    rows.append(dict(comparison=title,country='MACRO_paises_fijos',n=len(d),delta=np.mean([x['delta'] for x in rows]),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
    return rows

def main():
    d=pd.read_csv(rt.OUT/'cohorte_gshs_8paises.csv')
    # Guatemala lacks the family questions. Preserve those features in a seven-country analysis.
    d7=d[d.country_iso3!='GTM'].reset_index(drop=True)
    sp={k:(rt.FULL,v[1]) for k,v in rt.specs().items() if k.startswith('B1_logreg')}
    if '--summarize-only' not in sys.argv:
        rt.experiment(d7,'gshs7_familiar',sp)
        best7,inner7,scores7=rt.inner_search(d7,sp)
        cal7,th7=rt.calibrate(d7,inner7[best7])
        model7,_=rt.fit_predict(sp[best7],d7,d7.iloc[:1])
        rt.joblib.dump(dict(pipeline=model7,calibrator=cal7,features=rt.FULL,threshold=th7,selected=best7,source='GSHS seven countries; family questions available',research_only=True),rt.OUT/'modelo_familiar_7paises.joblib')
        (rt.OUT/'ajuste_final_familiar.json').write_text(json.dumps(dict(selected=best7,inner_macro_ap=scores7,threshold=th7),indent=2),encoding='utf-8')
    comparisons=[]
    five=selected_predictions(rt.OUT/'gshs5_core')
    fivefull=selected_predictions(rt.OUT/'gshs5_familiar')
    eight=selected_predictions(rt.OUT/'gshs8_core',True)
    sevenfull=selected_predictions(rt.OUT/'gshs7_familiar')
    for a,b,label in [(five,eight,'5_vs_8_mismas_variables_logistica'),(five,fivefull,'core_vs_familiar_5paises'),(fivefull,sevenfull,'5_vs_7_familiar')]:
        comparisons+=compare(a,b,label)
    pd.DataFrame(comparisons).to_csv(rt.OUT/'comparaciones_pareadas.csv',index=False)
    checks={}
    for name in ['gshs5_core','gshs5_familiar','gshs8_core','gshs7_familiar']:
        folder=rt.OUT/name
        p=pd.read_csv(folder/'predicciones_externas.csv')
        sels=json.loads((folder/'seleccion_interna.json').read_text())
        assert all(s['test_country'] not in s['train_countries'] for s in sels)
        assert not p.duplicated(['record_key','model']).any()
        assert p.p_calibrated.between(0,1).all() and p.p_calibrated.notna().all()
        expected=d7 if name=='gshs7_familiar' else d if name=='gshs8_core' else d[d.country_iso3.isin(rt.FIVE)]
        countries = set(expected.country_iso3)
        if name == 'gshs8_core' and countries != {survey[:3] for survey in rt.FILES}:
            raise ValueError('El núcleo común debe incluir los ocho países GSHS.')
        if len(sels) != len(countries) or {s['test_country'] for s in sels} != countries:
            raise ValueError(f'{name}: la evaluación no cubre cada país esperado exactamente una vez.')
        if not p.groupby('model').record_key.nunique().eq(len(expected)).all():
            raise ValueError(f'{name}: hay modelos sin predicciones para toda la cohorte evaluada.')
        assert set(p.record_key)==set(expected.record_key)
        chosen=selected_predictions(folder)
        assert chosen.record_key.is_unique and len(chosen)==len(expected)
        checks[name]=dict(n=len(expected),n_models=p.model.nunique(),held_out_countries=len(sels),country_codes=sorted(countries),disjoint=True,unique_predictions=True)
    # Verify the saved model actually reproduces valid probabilities on the raw column schema.
    bundle=rt.joblib.load(rt.OUT/'modelo_investigacion.joblib')
    raw=bundle['pipeline'].predict_proba(d[bundle['features']].iloc[:20])[:,1]
    cp=bundle['calibrator'].predict_proba(rt.sigmoid(raw))[:,1]
    assert np.isfinite(cp).all() and ((cp>=0)&(cp<=1)).all()
    checks['serialized_model_roundtrip']=True
    # Independent implementation agrees with sklearn, including tied values and survey weights.
    assert abs(weighted_ap(np.array([0,1,1,0]),np.array([.5,.5,.8,.1]),np.array([2.,3.,1.,2.]))-rt.average_precision_score([0,1,1,0],[.5,.5,.8,.1],sample_weight=[2,3,1,2]))<1e-12
    (rt.OUT/'verificaciones.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('COMPARACIONES Y VERIFICACIONES TERMINADAS',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
