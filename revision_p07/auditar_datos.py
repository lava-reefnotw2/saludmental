from pathlib import Path
import pandas as pd, re,json,zipfile,xml.etree.ElementTree as E,hashlib
R=Path(__file__).resolve().parent;D=R/'datasets'/'originales';O=R/'datasets'/'tabulares';O.mkdir(exist_ok=True)
summary=[]
for y in [2021,2023,2025]:
 s=(D/f'{y}XXH-SAS-Input-Program.sas').read_text();spec=re.findall(r'^@(\d+)\s+(\w+)\s+(\$?)(\d+)\.(\d*)',s,re.M)
 labels=dict(re.findall(r'^\s*(\w+)\s*=\s*"([^"]+)"',s,re.M)); rows=[]
 for line in (D/f'XXH{y}_YRBS_Data.dat').read_text().splitlines():
  row=[]
  for pos,n,char,w,dec in spec:
   v=line[int(pos)-1:int(pos)-1+int(w)].strip()
   if not char and v not in ['','.']:
    v=float(v) if '.' in v else int(v)/(10**int(dec or 0))
   row.append(None if v in ['','.'] else v)
  rows.append(row)
 df=pd.DataFrame(rows,columns=[x[1] for x in spec]);df.to_csv(O/f'YRBS_{y}_completo.csv',index=False)
 pd.DataFrame([dict(variable=n,position=pos,width=w,decimals=dec,label=labels.get(n,''),type='string' if c else 'number') for pos,n,c,w,dec in spec]).to_csv(O/f'YRBS_{y}_diccionario.csv',index=False)
 target='Q26' if y==2021 else 'Q27'
 summary.append(dict(dataset=f'YRBS{y}',rows=len(df),columns=len(df.columns),target=target,counts=df[target].value_counts(dropna=False).to_dict()))
for p in D.glob('*metadata.ddi'):
 root=E.parse(p).getroot();rec=[]
 for v in root.iter():
  if v.tag.split('}')[-1]!='var':continue
  def txt(tag):return ' '.join(' '.join(x.itertext()).split()) if (x:=next((a for a in v.iter() if a.tag.split('}')[-1]==tag),None)) is not None else ''
  cats=[]
  for c in v:
   if c.tag.split('}')[-1]=='catgry':cats.append(' '.join(' '.join(c.itertext()).split()))
  rec.append(dict(variable=v.get('name'),file=v.get('files'),label=txt('labl'),question=txt('qstnLit'),categories=' | '.join(cats)))
 pd.DataFrame(rec).to_csv(O/(p.stem.replace('_metadata','')+'_diccionario.csv'),index=False)
for p in D.glob('*.csv'):
 df=pd.read_csv(p); summary.append(dict(dataset=p.stem,rows=len(df),columns=len(df.columns)))
for p in D.glob('PENSE*zip'):
 try:
  with zipfile.ZipFile(p) as z:
   print(p.name,[(x.filename,x.file_size) for x in z.infolist()])
   for n in z.namelist():
    if n.lower().endswith(('.xlsx','.xls','.txt','.csv')):
     target=O/(p.stem+'_'+Path(n).name)
     if not target.exists():target.write_bytes(z.read(n))
 except zipfile.BadZipFile:print('ZIP INCOMPLETO',p.name)
w=[]
for p in D.glob('WDI*.json'):
 data=json.loads(p.read_text())
 for row in data[1]:w.append(dict(country_iso3=row['countryiso3code'],year=int(row['date']),indicator=row['indicator']['id'],value=row['value']))
if w:pd.DataFrame(w).to_csv(O/'WDI_contexto_largo.csv',index=False)
(R/'auditoria_conteos.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8');print(json.dumps(summary,indent=2,default=str))
