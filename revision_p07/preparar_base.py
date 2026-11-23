from pathlib import Path
import pandas as pd,json,hashlib
R=Path(__file__).resolve().parent;D=R/'datasets'/'originales';O=R/'datasets'/'tabulares'
files={'PER2010':'PER2010_PER2010_Public_Use.csv','ARG2018':'ARG2018_National.csv','PRY2017':'PRY2017_PYH2017.csv','URY2019':'URY2019_URY2019.csv','BOL2018':'BOL2018_National Dataset.csv'}
features={'q2':'sex_code','q6':'food_insecurity','q15':'physical_attack','q16':'physical_fight','q27':'close_friends','q35':'alcohol_30d','q49':'physical_activity','q52':'sedentary_time','q53':'truancy','q54':'peer_kindness','q55':'parental_supervision','q56':'parental_understanding','q57':'parental_knowledge'}
frames=[];stats=[];mapping=[];miss=[]
for tag,fn in files.items():
 d=pd.read_csv(D/fn);d.columns=d.columns.str.lower()
 dic=pd.read_csv(O/(tag+'_diccionario.csv'));dic.variable=dic.variable.str.lower();dic=dic.drop_duplicates('variable').set_index('variable')
 z=pd.DataFrame(index=d.index);z['survey_id']=tag;z['country_iso3']=tag[:3];z['survey_year']=int(tag[3:]);z['record_key']=[f'{tag}_{i+1:06}' for i in range(len(d))]
 z['age_years']=d.q1.map({3:13,4:14,5:15})
 for src,name in features.items():
  z[name]=d[src] if src in d else float('nan')
  mapping.append(dict(survey_id=tag,source_variable=src,canonical=name,label=dic.loc[src,'label'] if src in dic.index else '',question=dic.loc[src,'question'] if src in dic.index else '',categories=dic.loc[src,'categories'] if src in dic.index else ''))
 for src,name in [('q24','y1_ideation'),('q25','y2_plan')]:z[name]=d[src].map({1:1,2:0})
 z['y3_attempt']=d.q26.map({1:0,2:1,3:1,4:1,5:1})
 for src,name in [('q22','y4_loneliness'),('q23','y5_worry_sleep')]:z[name]=d[src].map({1:0,2:0,3:0,4:1,5:1})
 for v in ['weight','psu','stratum']:z[v]=d[v]
 eligible=z.age_years.notna() & z.y1_ideation.notna()
 stats.append(dict(dataset=tag,n_raw=len(d),columns_raw=len(d.columns),n_age13_15=int(z.age_years.notna().sum()),n_eligible=int(eligible.sum()),events=int(z.loc[eligible,'y1_ideation'].sum()),non_events=int((z.loc[eligible,'y1_ideation']==0).sum()),psu=int(z.loc[eligible,['stratum','psu']].drop_duplicates().shape[0]),missing_y1_raw=int(z.y1_ideation.isna().sum()),duplicate_full_rows=int(d.duplicated().sum()),nonpositive_weights=int((z.weight<=0).sum())))
 for c in features.values():miss.append(dict(dataset=tag,variable=c,missing_pct=100*z.loc[eligible,c].isna().mean()))
 frames.append(z.loc[eligible])
out=pd.concat(frames,ignore_index=True);out.to_csv(O/'GSHS_5paises_13a15_base.csv',index=False)
pd.DataFrame(mapping).to_csv(O/'GSHS_mapeo_predictores.csv',index=False)
pd.DataFrame(stats).to_csv(O/'GSHS_factibilidad.csv',index=False)
pd.DataFrame(miss).to_csv(O/'GSHS_faltantes.csv',index=False)
w=pd.read_csv(O/'WDI_contexto_largo.csv');wide=w.pivot(index=['country_iso3','year'],columns='indicator',values='value').reset_index()
cohort=pd.DataFrame([dict(country_iso3=t[:3],year=int(t[3:])) for t in files]);coverage=cohort.merge(wide,on=['country_iso3','year'],how='left');coverage.to_csv(O/'WDI_cobertura_GSHS.csv',index=False)
print(pd.DataFrame(stats).to_string(index=False));print('TOTAL',len(out));print(coverage.to_string(index=False))
manifest=[]
for name in ['descargas_cdc.json','descargas_gshs.json','descargas_contexto.json','descargas_guias.json']:
 manifest+=json.loads((R/name).read_text())
for x in manifest:x['verified_date']='2026-11-17'
pd.DataFrame(manifest).to_csv(R/'datasets'/'manifest_descargas.csv',index=False)
for p in O.glob('*.csv'):
 manifest.append(dict(file='tabulares/'+p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),status='derivado local'))
(R/'datasets'/'manifest_completo.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
