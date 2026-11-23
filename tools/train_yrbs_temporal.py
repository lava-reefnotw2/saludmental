"""Independent US replication: development 2021, calibration 2023, test 2025."""
import sys, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import real_data_training as rt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

O=rt.OUT/'yrbs_temporal'; O.mkdir(exist_ok=True)
COLS=['age_years','sex_code','physical_fight','alcohol_30d','physical_activity','school_bullying','electronic_bullying','breakfast']
MAP={2021:[1,2,16,41,77,23,24,76,26],2023:[1,2,16,42,76,24,25,75,27],2025:[1,2,15,41,73,23,24,72,27]}

def prepare():
    frames={}; audit=[];mapping=[]
    for year,qs in MAP.items():
        p=ROOT/f'revision_p07/datasets/tabulares/YRBS_{year}_completo.csv'
        d=pd.read_csv(p,low_memory=False)
        dic=pd.read_csv(p.with_name(f'YRBS_{year}_diccionario.csv')).set_index('variable')
        z=pd.DataFrame(index=d.index)
        formats=(ROOT/f'revision_p07/datasets/originales/{year}XXH-Formats-Program.sas').read_text()
        for c,q in zip(COLS+['y1_ideation'],qs):
            src=f'Q{q}'
            # Save exact official response labels, not just field numbers.
            match=re.search(r'value\s+\$H'+str(q)+r'S\b(.*?);',formats,re.S|re.I)
            if not match: raise ValueError(f'No SAS format for {year}/{src}')
            cats=dict(re.findall(r'"([^\"]+)"\s*=\s*"([^\"]+)"',match[1]))
            allowed=[int(k) for k in cats if k.isdigit()]
            z[c]=pd.to_numeric(d[src],errors='coerce').where(lambda x:x.isin(allowed))
            mapping.append(dict(year=year,canonical=c,source=src,label=dic.loc[src,'label'],categories=cats))
        z.age_years=z.age_years.map({2:13,3:14,4:15,5:16,6:17})
        z.sex_code=z.sex_code.map({1:2,2:1}) # CDC 1=female; canonical 1=male
        z.y1_ideation=z.y1_ideation.map({1:1,2:0})
        design={c.lower():c for c in d.columns}
        for c in ['weight','stratum','psu']: z[c]=pd.to_numeric(d[design[c]],errors='coerce')
        z['country_iso3']='USA';z['survey_id']=f'YRBS{year}'
        z['record_key']=[f'YRBS{year}_{i+1:06}' for i in range(len(d))]
        keep=z.age_years.notna()&z.y1_ideation.notna()
        frames[year]=z.loc[keep].reset_index(drop=True)
        audit.append(dict(year=year,n_raw=len(d),n_eligible=int(keep.sum()),events=int(z.loc[keep,'y1_ideation'].sum()),sha256=rt.digest(p)))
    pd.DataFrame(audit).to_csv(O/'cohorte.csv',index=False)
    (O/'armonizacion.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    return frames

def main():
    frames=prepare();tr,cal,te=[frames[y] for y in [2021,2023,2025]]
    candidates={k: (COLS[:2] if k.startswith('B0') else COLS,v[1]) for k,v in rt.specs().items()}
    groups=tr.stratum.astype(str)+'_'+tr.psu.astype(str)
    scores={}
    for name,spec in candidates.items():
        pred=np.zeros(len(tr))
        for a,b in GroupKFold(3).split(tr,groups=groups):
            _,pred[b]=rt.fit_predict(spec,tr.iloc[a],tr.iloc[b])
        scores[name]=float(average_precision_score(tr.y1_ideation,pred,sample_weight=tr.weight))
    best=max(scores,key=scores.get)
    rows=[];predictions=[]
    for name,spec in candidates.items():
        model,pcal=rt.fit_predict(spec,tr,cal)
        calibrator,threshold=rt.calibrate(cal,pcal)
        raw=model.predict_proba(te[spec[0]])[:,1]
        p=calibrator.predict_proba(rt.sigmoid(raw))[:,1]
        row=rt.metrics(te,p,threshold);row.update(model=name,selected=name==best)
        rows.append(row)
        q=te[['record_key','y1_ideation','weight','stratum','psu','sex_code','age_years']].copy()
        q['model'],q['p_calibrated']=name,p;predictions.append(q)
        if name==best:
            rt.joblib.dump(dict(pipeline=model,calibrator=calibrator,features=spec[0],threshold=threshold,source='YRBS USA 2021; calibration 2023',research_only=True),O/'modelo_temporal.joblib')
    pd.DataFrame(rows).to_csv(O/'evaluacion_2025.csv',index=False)
    pd.concat(predictions).to_csv(O/'predicciones_externas.csv',index=False)
    (O/'seleccion_2021.json').write_text(json.dumps(dict(selected=best,inner_scores=scores),indent=2),encoding='utf-8')
    print('YRBS terminado',best,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
