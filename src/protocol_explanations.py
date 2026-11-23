"""Country-held-out explanation summaries and design-based uncertainty for logistic ORs."""
import numpy as np
import pandas as pd
from scipy.linalg import pinvh
from scipy.special import expit
from catboost import Pool
import real_data_training as rt

def elaborate(run):
    out=run/'protocolo'
    data=pd.read_csv(run/'cohorte_gshs_8paises.csv')
    data=data[data.country_iso3.isin(rt.FIVE)]
    pred=pd.read_csv(run/'gshs5_familiar/metricas_por_pais.csv')
    ranks=[]; pairs=[]; odds=[]; dependence=[]
    for country in sorted(rt.FIVE):
        tr=data[data.country_iso3!=country].reset_index(drop=True)
        g=data[data.country_iso3==country].sample(n=384,random_state=42)
        c=.1 if pred.loc[(pred.country==country)&pred.selected,'model'].iloc[0].endswith('C01') else 1.
        logistic,_=rt.fit_predict((rt.FULL,rt.pipe(rt.LogisticRegression(C=c,max_iter=600,random_state=42))),tr,g)
        x=logistic[:-1].transform(tr[rt.FULL])
        x=np.column_stack((np.ones(len(tr)),x))
        weights=rt.weights(tr); p=logistic.predict_proba(tr[rt.FULL])[:,1]
        bread=x.T@((weights*p*(1-p))[:,None]*x)+np.diag([0]+[1/c]*(x.shape[1]-1))
        score=pd.DataFrame(x*(weights*(tr.y1_ideation.to_numpy()-p))[:,None])
        clusters=score.groupby([tr.country_iso3,tr.stratum,tr.psu]).sum()
        meat=np.zeros_like(bread)
        for _,ss in clusters.groupby(level=[0,1]):
            if len(ss)>1:
                a=ss.to_numpy()-ss.to_numpy().mean(axis=0)
                meat+=a.T@a*(len(ss)/(len(ss)-1))
        inv=pinvh(bread);cov=inv@meat@inv;offset=1
        coef=logistic['model'].coef_[0]
        for col,categories in zip(rt.FULL,logistic['encode'].categories_):
            ref=next(i for i,v in enumerate(categories) if v!=-99)
            for j,v in enumerate(categories):
                idx=offset+j;ri=offset+ref
                diff=coef[idx-1]-coef[ri-1]
                se=np.sqrt(max(0,cov[idx,idx]+cov[ri,ri]-2*cov[idx,ri]))
                odds.append(dict(country_held_out=country,variable=col,category=v,reference=categories[ref],adjusted_OR=np.exp(diff),low=np.exp(diff-1.96*se),high=np.exp(diff+1.96*se)))
            offset+=len(categories)
        for name in ['B1_logreg_C01','B1_random_forest','B1_xgboost','B1_catboost','B1_mlp']:
            spec=rt.specs()[name];cols=spec[0];f,_=rt.fit_predict(spec,tr,g)
            base=rt.average_precision_score(g.y1_ideation,f.predict_proba(g[cols])[:,1],sample_weight=g.weight)
            rng=np.random.default_rng(42)
            for col in cols:
                changed=g[cols].copy();changed[col]=rng.permutation(changed[col])
                ap=rt.average_precision_score(g.y1_ideation,f.predict_proba(changed)[:,1],sample_weight=g.weight)
                ranks.append(dict(country=country,model=name,variable=col,delta_AP=base-ap))
            if name=='B1_catboost':
                sm=g.iloc[:128]
                interactions=f['model'].get_feature_importance(Pool(f[:-1].transform(sm[cols])),type='ShapInteractionValues')
                starts=np.cumsum([0]+[len(a) for a in f['encode'].categories_]);ww=sm.weight
                for i,col in enumerate(cols):
                    for j in range(i):
                        term=interactions[:,starts[i]:starts[i+1],starts[j]:starts[j+1]].sum(axis=(1,2))
                        pairs.append(dict(country=country,first=cols[j],second=col,mean_abs=np.average(abs(term),weights=ww)))
                for col in ['food_insecurity','physical_attack','close_friends']:
                    for val in sorted(g[col].dropna().unique()):
                        repl=g[cols].copy();repl[col]=val
                        dependence.append(dict(country=country,variable=col,category=val,mean_prediction=np.average(f.predict_proba(repl)[:,1],weights=g.weight)))
    ranking=pd.DataFrame(ranks);ranking['rank']=ranking.groupby(['country','model']).delta_AP.rank(ascending=False)
    ranking.to_csv(out/'estabilidad.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'interacciones.csv',index=False)
    pd.DataFrame(dependence).to_csv(out/'dependencia.csv',index=False)
    pd.DataFrame(odds).to_csv(out/'OR.csv',index=False)
