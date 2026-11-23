"""Sensitivity analyses for the fixed research profile and secondary outcomes."""
import numpy as np
import pandas as pd
import real_data_training as rt
from protocol_analysis import decision_rows
PROFILE=dict(age_years=14,sex_code=1,food_insecurity=2,physical_attack=1,physical_fight=1,close_friends=3,alcohol_30d=1,physical_activity=4,sedentary_time=2,truancy=1,peer_kindness=3,parental_supervision=3,parental_understanding=3,parental_knowledge=3)
def supplemental(run):
    d=pd.read_csv(run/'cohorte_gshs_8paises.csv')
    family=d[d.country_iso3!='GTM']
    probs=[];rows=[]
    log=rt.specs()['B1_logreg_C01'][1]
    for c in sorted(family.country_iso3.unique()):
        train=family[family.country_iso3!=c].reset_index(drop=True)
        spec=(rt.FULL,log)
        _,pred,_=rt.inner_search(train,{'profile':spec})
        cal,_=rt.calibrate(train,pred['profile'])
        _,raw=rt.fit_predict(spec,train,pd.DataFrame([PROFILE],dtype=float))
        probs.append(dict(country_held_out=c,p=cal.predict_proba(rt.sigmoid(raw))[0,1]))
    five=d[d.country_iso3.isin(rt.FIVE)]
    for target in ['y2_plan','y3_attempt','y4_loneliness','y5_worry_sleep']:
        dd=five.dropna(subset=[target]).copy()
        dd['y1_ideation']=dd[target]
        for c in rt.FIVE:
            tr=dd[dd.country_iso3!=c];te=dd[dd.country_iso3==c]
            _,p=rt.fit_predict((rt.FULL,log),tr,te)
            rows.append(dict(country=c,outcome=target,n=len(te),events=int(te[target].sum()),AP=rt.average_precision_score(te[target],p,sample_weight=te.weight),ROC_AUC=rt.roc_auc_score(te[target],p,sample_weight=te.weight)))
    pred=pd.read_csv(run/'gshs5_familiar/predicciones_externas.csv')
    benefit=decision_rows(pred[pred.selected])
    for name,values in [('perfil_sintetico',probs),('resultados_secundarios',rows),('decision_exploratoria',benefit)]:
        pd.DataFrame(values).to_csv(run/'protocolo'/(name+'.csv'),index=False)

