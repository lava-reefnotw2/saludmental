"""Aggregate report downloads, traceability, and protocol views."""
import io,json,platform,subprocess,uuid
from pathlib import Path
from importlib.metadata import version
from publication_charts import chart
from research_notes import TABLE_NOTES
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import real_data_training as rt
from training_jobs import atomic_json

def provenance(run):
    import hashlib
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=rt.ROOT,capture_output=True,text=True)
    return dict(run=run.name,git_HEAD=git.stdout.strip() or 'indisponible',python=platform.python_version(),
                scripts={p:rt.digest(rt.ROOT/p) for p in ['src/real_data_training.py','src/protocol_analysis.py','src/protocol_optuna.py','src/protocol_views.py','src/discussion_figures.py','tools/run_real_workflow.py']},
                versions={p:version(p) for p in ['pandas','scikit-learn','catboost','optuna','streamlit']})

def exports(run,folder):
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    tab=pd.read_csv(folder/'H1.csv')
    fig,ax=plt.subplots(figsize=(8,4))
    ax.errorbar(tab.delta_AP,tab.country,xerr=[tab.delta_AP-tab.low,tab.high-tab.delta_AP],fmt='o',capsize=3)
    ax.axvline(0,color='grey');ax.set_xlabel('Diferencia de average precision (B1 menos B0)')
    ax.set_title('GSHS · Intervalos pareados por conglomerado');fig.tight_layout()
    fig.savefig(folder/'H1.png',dpi=600)
    description=['GSHS · cinco países · estudiantes de 13–15 años',run.name,'Fuente: CSV nacionales, pesos de encuesta y validación por país.',
                 'Y1: ideación autorreportada en los últimos 12 meses. Semilla 42.',
                 'IC percentiles: 500 remuestreos de PSU dentro de estrato, modelos fijos.',
                 'Promedio igual entre países. Uso agregado de investigación.',
                 'Los intervalos describen las encuestas y los modelos ajustados.',
                 'Las brechas por subgrupo y SHAP se interpretan de forma descriptiva.']
    from word_evaluation import write_report
    write_report(run,folder)
    with PdfPages(folder/'evaluacion.pdf') as pdf:
        cover=plt.figure(figsize=(9,6));cover.text(.07,.9,'Evaluación del protocolo P07',fontsize=18)
        cover.text(.07,.8,'\n\n'.join(description),va='top',fontsize=10);pdf.savefig(cover);plt.close(cover)
        pdf.savefig(fig)
    plt.close(fig)
    with pd.ExcelWriter(folder/'evaluacion.xlsx',engine='openpyxl') as writer:
        for path in sorted(folder.glob('*.csv')):
            pd.read_csv(path).to_excel(writer,sheet_name=path.stem[:31],index=False)
        for path in sorted((folder/'nucleo_8').glob('*.csv')):
            pd.read_csv(path).to_excel(writer,sheet_name=f'8p_{path.stem}'[:31],index=False)
        pd.DataFrame(json.loads((run/'fuentes_huellas.json').read_text())).to_excel(writer,sheet_name='fuentes',index=False)
        pd.DataFrame({'registro':[json.dumps(provenance(run),ensure_ascii=False)]}).to_excel(writer,sheet_name='versiones',index=False)

def render_protocol(run):
    folder=run/'protocolo'
    if not (folder/'H1.csv').exists():
        st.info('Los resultados del protocolo se generan al completar el entrenamiento actualizado.');return
    st.header('Comparación entre países')
    core_folder = folder / 'nucleo_8'
    has_core = (core_folder / 'H1.csv').is_file()
    core_label = 'Núcleo común · 8 países'
    family_label = 'Bloque familiar · 5 países comparables'
    scope = st.radio('Datos de la comparación', (core_label, family_label) if has_core else (family_label,),
                     horizontal=True)
    core = scope == core_label
    results = core_folder if core else folder
    if core:
        st.caption('Argentina, Bolivia, Guatemala, Honduras, Panamá, Paraguay, Perú y Uruguay. La comparación principal usa los nueve predictores comunes, con un país excluido por ajuste.')
    else:
        st.caption('Argentina, Bolivia, Paraguay, Perú y Uruguay. El bloque familiar conserva el grupo original para las comparaciones pareadas. Guatemala carece de las cinco preguntas adicionales.')
    from discussion_figures import discussion
    discussion(run, core=core)
    tables = [('H1','Mejora sobre edad y sexo'),('transporte','Transporte frente al desarrollo'),('equidad','Diferencias por sexo y edad'),('explicacion','Explicaciones entre países'),('H4','Familias y ensamble de pesos iguales · 8 países'),('estabilidad','Estabilidad de rangos por permutación'),('interacciones','Interacciones SHAP de CatBoost'),('dependencia','Dependencia parcial de CatBoost'),('OR','Razones de odds ajustadas'),('secundarios','Factibilidad de desenlaces secundarios'),('resultados_secundarios','Desenlaces secundarios · logística exploratoria entre estudiantes con Y1 observado'),('decision_exploratoria','Utilidad exploratoria · escenario de invitación voluntaria a apoyo'),('metricas','Métricas y matrices ponderadas'),('sensibilidad_AP','Sensibilidad de AP a la ponderación')]
    for name,title in tables:
        path = (folder if name == 'H4' else results) / f'{name}.csv'
        if not path.is_file():
            continue
        with st.expander(title,expanded=name=='H1'):
            st.dataframe(pd.read_csv(path),hide_index=True)
            st.caption('ΔAP compara la logística del núcleo común (C=0,1) con edad y sexo en los mismos estudiantes de cada país.' if core and name == 'H1' else TABLE_NOTES[name])
    for fig,note in explanation_figures(results):
        chart(fig)
        st.caption(note)
    from discussion_figures import COUNTRIES
    available=sorted(pd.read_csv(results/'curvas.csv',usecols=['country']).country.unique())
    multi=st.checkbox(f'Mostrar los {len(available)} países en las curvas',value=True,key='external_all')
    country=st.selectbox('País para curvas externas',available,format_func=COUNTRIES.get,
                         disabled=multi,key=f'external_country_{core}')
    selected=available if multi else [country]
    for fig,note in external_figures(results,selected,multi):
        chart(fig)
        st.caption(note)
    with st.expander('Descargas del protocolo familiar'):
        for file,label in [('evaluacion.docx','Word'),('evaluacion.pdf','PDF'),('evaluacion.xlsx','Excel'),('H1.png','Figura 600 DPI')]:
            if (folder/file).exists():st.download_button(label,(folder/file).read_bytes(),file_name=file)
    with st.expander('Fase 6 · Ejemplo sintético reproducible'):
        if st.button('Evaluar perfil sintético de investigación'):
            from protocol_sensitivity import PROFILE
            pack=rt.joblib.load(run/'modelo_familiar_7paises.joblib')
            example=PROFILE
            raw=pack['pipeline'].predict_proba(pd.DataFrame([example],dtype=float)[pack['features']])[:,1]
            p=pack['calibrator'].predict_proba(rt.sigmoid(raw))[0,1]
            st.json(example);st.metric('Probabilidad calibrada del ejemplo',f'{p:.1%}')
            if (folder/'perfil_sintetico.csv').exists():
                folds=pd.read_csv(folder/'perfil_sintetico.csv')
                st.write(f'Rango de sensibilidad al excluir un país del ajuste logístico C=0,1: {folds.p.min():.1%} a {folds.p.max():.1%}.')
            st.caption('Modelo '+pack['selected']+'. La curva de calibración ilustra incertidumbre poblacional. La incertidumbre individual del ejemplo no está estimada.')
    with st.expander('Fases 2 y 7 · Estudio Optuna'):
        family=st.selectbox('Familia del estudio',[k for k in rt.specs() if k.startswith('B1') and k!='B1_logreg_C01'])
        block=st.selectbox('Predictores',['B0','B1','B2'])
        st.caption('B0 edad y sexo. B1 familiar completo. B2 añade cuatro indicadores país-año, exploratorio. Búsqueda interna por AP macro. Espacios: C 0,01–10, alfa MLP 0,01–1, profundidad CatBoost 3–6 y otros árboles 3–8.')
        budget=st.number_input('Ensayos por país externo',1,30,3)
        seed=st.number_input('Semilla Optuna',0,999999,42)
        if st.button('Iniciar estudio anidado'):
            from protocol_optuna import optimise
            msg=st.empty()
            dest=folder/'optuna'/uuid.uuid4().hex;dest.mkdir(parents=True)
            atomic_json(dest/'config.json',dict(family=family,block=block,budget=budget,seed=seed))
            try:
                trial,ext=optimise(run,family,block,int(budget),int(seed),lambda c,i,s:msg.write(f'{c} · ensayo {i}/{budget} · {s}'),dest)
                ext.to_csv(dest/'external.csv',index=False)
            except Exception as exc:
                st.error(f'El estudio se interrumpió. {exc}')
        for dest in sorted((folder/'optuna').glob('*')):
            if (dest/'config.json').exists():
                st.json(json.loads((dest/'config.json').read_text()),expanded=False)
                for result in ('trials.csv','external.csv'):
                    if (dest/result).is_file():
                        st.dataframe(pd.read_csv(dest/result))
                        st.caption("Cada fila muestra un ensayo interno. Una AP mayor favorece esa configuración en desarrollo." if result=="trials.csv" else "Cada fila evalúa la configuración elegida en un país excluido. La variación entre países describe su transporte.")

def inspect_import(run):
    with st.expander('Fase 1 · Inspección y armonización de archivos'):
        upload=st.file_uploader('Encuesta para inspección',type=['csv','xlsx','parquet','dta','sav'])
        if upload is None:return
        readers={'.csv':pd.read_csv,'.xlsx':pd.read_excel,'.parquet':pd.read_parquet,'.dta':lambda f:pd.read_stata(f,convert_categoricals=False),'.sav':pd.read_spss}
        try: frame=readers[Path(upload.name).suffix.lower()](upload)
        except Exception as exc:st.error(f'No se pudo leer la encuesta. {exc}');return
        st.write(f'{len(frame):,} registros · {len(frame.columns)} columnas')
        mapping=pd.DataFrame({'columna_origen':frame.columns,'tipo':frame.dtypes.astype(str).values,'ausentes':frame.isna().sum().values,'variable_armonizada':['']*len(frame.columns)})
        edited=st.data_editor(mapping,disabled=['columna_origen','tipo','ausentes'],hide_index=True)
        codes=edited.variable_armonizada.str.strip()
        if codes[codes!=''].duplicated().any():st.error('Hay variables armonizadas duplicadas.');return
        st.download_button('Exportar mapeo revisado',edited.to_csv(index=False).encode(),file_name='mapeo_revisado.csv')
        st.caption('Vista de preparación. El mapeo exportado permite revisar la equivalencia de preguntas antes de incorporar una encuesta a la cohorte.')








def external_figures(folder,selected,multi):
    from discussion_figures import COUNTRIES
    curves=pd.read_csv(folder/'curvas.csv')
    available=sorted(curves.country.unique())
    title=f'{len(selected)} países' if multi else COUNTRIES[selected[0]]
    def visible(frame):
        frame=frame.loc[frame.country.isin(selected)].copy()
        frame['País']=frame.country.map(COUNTRIES)
        return frame
    curves=visible(curves)
    legend=dict(color='País',category_orders={'País':[COUNTRIES[c] for c in available]}) if multi else {}
    for typ in ['PR','ROC']:
        fig=px.line(curves[curves.curve==typ],x='x',y='y',title=typ+' · '+title,
                    line_dash='País' if multi else None,**legend,
                    labels={'x':'Sensibilidad' if typ=='PR' else 'Tasa de falsos positivos','y':'Precisión' if typ=='PR' else 'Sensibilidad'})
        if typ=='PR':
            metrics=visible(pd.read_csv(folder/'metricas.csv'))
            refs=metrics.loc[metrics.model=='B1'].set_index('País').prevalence
            if multi:
                for trace in fig.data:
                    trace.hovertemplate+=f'<br>Prevalencia de referencia {refs[trace.name]:.1%}'
            else:fig.add_hline(y=refs.iloc[0],line_dash='dot',annotation_text=f'Prevalencia {refs.iloc[0]:.1%}')
        else:fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line_dash='dot')
        yield fig, ('Cada punto corresponde a un umbral. La precisión indica qué proporción de los señalados respondió Y1 positivamente y la sensibilidad qué proporción de los positivos se recuperó. La prevalencia del país sirve de referencia.' if typ=='PR' else 'Al bajar el umbral suelen aumentar tanto la sensibilidad como la tasa de falsos positivos. La diagonal representa una clasificación al azar.')
    bins=visible(pd.read_csv(folder/'calibracion.csv'))
    bins['más']=(bins.high-bins.observed).clip(lower=0)
    bins['menos']=(bins.observed-bins.low).clip(lower=0)
    fig=px.scatter(bins,x='predicted',y='observed',error_y='más',error_y_minus='menos',
                   symbol='País' if multi else None,**legend,hover_data=['n','low','high'],
                   title='Calibración externa con IC95% · '+title,range_x=[0,1],range_y=[0,1],
                   labels={'predicted':'Probabilidad predicha','observed':'Proporción observada'})
    fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line_dash='dot')
    yield fig, ('Cada punto agrupa probabilidades similares. Por encima de la diagonal la frecuencia observada supera la predicción media y el modelo subestima Y1. Las barras muestran el IC95% de la frecuencia observada.')


def explanation_figures(folder):
    if folder.name == 'nucleo_8':
        from discussion_figures import country_labels
        frame = country_labels(pd.read_csv(folder/'explicacion.csv'))
        fig = px.scatter(frame, x='mean_abs_SHAP', y='variable', color='País',
                         title='Aportes SHAP de CatBoost · núcleo común de ocho países',
                         labels={'mean_abs_SHAP':'Media ponderada de |SHAP|', 'variable':'Predictor'})
        yield fig, 'Cada punto resume la magnitud de los aportes en hasta 512 estudiantes del país excluido. SHAP mide cuánto contribuye cada respuesta a la salida del modelo.'
    if (folder/'estabilidad.csv').exists():
        ranks=pd.read_csv(folder/'estabilidad.csv')
        yield px.box(ranks,x='variable',y='rank',color='model',title='Estabilidad del rango de importancia en países externos'), 'El rango 1 identifica la variable cuya permutación más reduce la AP. Una caja amplia indica que su posición varía entre países.'
        dep=pd.read_csv(folder/'dependencia.csv')
        yield px.line(dep[dep.variable=='food_insecurity'],x='category',y='mean_prediction',color='country',title='Dependencia parcial de inseguridad alimentaria · CatBoost'), 'Cada trazo resume cómo cambia la predicción media al fijar la categoría de hambre para los estudiantes de ese país. Es una descripción del modelo, sin interpretación causal.'
