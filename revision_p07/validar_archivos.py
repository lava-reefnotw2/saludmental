from pathlib import Path
import pandas as pd,json,re,hashlib,zipfile
R=Path(__file__).resolve().parent;D=R/'datasets'/'originales';O=R/'datasets'/'tabulares';results={}
df=pd.read_csv(O/'GSHS_5paises_13a15_base.csv')
assert len(df)==44837 and df.record_key.is_unique
assert set(df.age_years)=={13,14,15} and df.country_iso3.nunique()==5
assert set(df.y1_ideation)=={0,1} and df.y1_ideation.sum()==9509
assert (df.weight>0).all() and df[['psu','stratum']].notna().all().all()
for c in ['y2_plan','y3_attempt','y4_loneliness','y5_worry_sleep']:assert set(df[c].dropna())<={0,1}
results['GSHS']={'rows':len(df),'columns':len(df.columns),'events':int(df.y1_ideation.sum()),'unique_record_keys':True,'design_complete':True,'age_range':'13-15','labels_binary':True}
for y in [2021,2023,2025]:
 d=pd.read_csv(O/f'YRBS_{y}_completo.csv',low_memory=False);d.columns=d.columns.str.lower();s=(D/f'{y}XXH-SAS-Input-Program.sas').read_text();spec=re.findall(r'^@(\d+)\s+(\w+)\s+(\$?)(\d+)\.(\d*)',s,re.M)
 lines=(D/f'XXH{y}_YRBS_Data.dat').read_text().splitlines();maxpos=max(int(a)+int(w)-1 for a,n,c,w,dec in spec)
 assert len(lines)==len(d) and min(map(len,lines))>=maxpos
 q='q26' if y==2021 else 'q27';qn=q.replace('q','qn')
 assert d[q].equals(d[qn]);assert d.weight.gt(0).all();assert d[['psu','stratum']].notna().all().all()
 results[f'YRBS{y}']={'rows':len(d),'columns':len(d.columns),'min_line_length':min(map(len,lines)),'last_field_end':maxpos,'label_matches_official_derived_variable':True,'weights_positive':True}
for y in [2019,2024]:
 p=O/f'PENSE{y}_Dados_PENSE{y}_MICRODADOS.csv';d=pd.read_csv(p,usecols=['B12005','B12008','PESO_ALUNO_FREQ','IND_EXPANSAO'],low_memory=False)
 with zipfile.ZipFile(D/f'PENSE{y}_Dados.zip') as z:
  member=next(n for n in z.namelist() if n.lower().endswith('.csv'))
  assert hashlib.sha256(z.read(member)).hexdigest()==hashlib.sha256(p.read_bytes()).hexdigest()
 results[f'PENSE{y}']={'rows':len(d),'csv_matches_official_zip':True,'tristeza_valid_1to5':int(d.B12005.isin([1,2,3,4,5]).sum()),'vida_valid_1to5':int(d.B12008.isin([1,2,3,4,5]).sum()),'nonpositive_or_missing_final_weights':int((~d.PESO_ALUNO_FREQ.gt(0)).sum())}
manifest=json.loads((R/'datasets'/'manifest_completo.json').read_text());verified=0
for item in manifest:
 if 'sha256' not in item:continue
 p=R/'datasets'/item['file'] if item['file'].startswith('tabulares/') else D/item['file']
 assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],str(p)
 verified+=1
results['manifest_files_verified']=verified
(R/'validacion_archivos.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2))
