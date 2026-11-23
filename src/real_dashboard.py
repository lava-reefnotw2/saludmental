"""Streamlit views backed exclusively by verified empirical training artifacts."""
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from publication_charts import chart
import plotly.express as px
from training_jobs import BASE,active_results,start_run,job_status,read_json

@st.cache_data(show_spinner=False)
def table(path,stamp):return pd.read_csv(path)

def load(folder,name):
    p=folder/name
    return table(str(p),p.stat().st_mtime_ns) if p.is_file() else pd.DataFrame()

@st.fragment(run_every='3s')
def progress_panel():
    status=job_status()
    if not status:return
    state=status.get('state')
    if state in ['starting','running']:
        st.progress(min(status.get('phase',0)/7,.99),text=status.get('message','Entrenando'))
        st.caption('El entrenamiento continúa aunque cierres esta pestaña.')
    elif state=='failed':st.error('Falló la ejecución: '+status.get('message','Consulta el registro.'))
    elif state=='completed':
        st.success('Entrenamiento real completado.')
        active=active_results()
        if active and str(active)!=st.session_state.get('displayed_results'):
            st.session_state['displayed_results']=str(active)
            table.clear()
            st.rerun()
    log=BASE/'runs'/status['run_id']/'ejecucion.log'
    if log.exists():
        with st.expander('Registro de ejecución',expanded=state in ['starting','running']):
            st.code('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-14:]),language='text')

def training_controls():
    st.subheader('Reentrenamiento')
    include_yrbs = st.checkbox('Incluir réplica independiente YRBS', value=False,
                               help='Requiere CSV YRBS, diccionarios y formatos SAS de revision_p07. Se analiza por separado.')
    busy = job_status().get('state') in ('starting', 'running')
    if st.button('Re-ejecutar los 7 módulos', icon=':material/refresh:', type='primary', disabled=busy):
        try:
            start_run(include_yrbs)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(str(error))
        else:
            st.rerun()
    st.caption('La aplicación muestra la última ejecución verificada mientras se entrena la nueva.')
    progress_panel()


def survey_sources(folder, audit):
    from discussion_figures import COUNTRIES
    from real_data_training import FILES

    sources = read_json(folder / 'fuentes_huellas.json', [])
    by_file = {source['file'].replace('\\', '/'): source for source in sources}
    by_survey = {source.get('survey_id'): source for source in sources if source.get('survey_id')}
    rows = []
    for row in audit.itertuples(index=False):
        code, year = row.survey_id[:3], row.survey_id[3:]
        file = FILES.get(row.survey_id)
        source = by_survey.get(row.survey_id) or by_file.get(f'datasets/{file}', {})
        rows.append({'País': COUNTRIES.get(code, code), 'Año': int(year),
                     'Dataset': source.get('file', file or 'Sin ruta registrada'),
                     'Registros de origen': row.n_raw, 'Estudiantes elegibles': row.n_eligible})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    with st.expander('Huellas de los archivos fuente'):
        st.dataframe(pd.DataFrame(sources), hide_index=True, width='stretch')


def render():
    st.set_page_config(page_title='P07 · Salud mental adolescente',page_icon='🧠',layout='wide')
    st.html(Path(__file__).with_name('dashboard_theme.css'))
    if 'p07_page' not in st.session_state:
        st.session_state['p07_page'] = 'Entrenamiento'
    page = st.session_state['p07_page']
    with st.sidebar:
        st.html('''<div class="p07-brand">
            <span class="p07-brand-mark">P07</span>
            <strong>Salud mental adolescente</strong>
        </div>''')
        with st.container(key='p07_navigation'):
            for label, icon, key in (
                ('Entrenamiento', ':material/monitoring:', 'training'),
                ('Explorador de patrones', ':material/insights:', 'explorer'),
            ):
                active = page == label
                st.button(label, icon=icon, key=f'p07_nav_{key}', width='stretch',
                          type='primary' if active else 'secondary',
                          on_click=st.session_state.update, kwargs={'p07_page': label})
    folder = active_results()
    if page == 'Explorador de patrones':
        if folder is None:
            st.info('Ejecuta el entrenamiento desde Entrenamiento → 7 · Modelos para habilitar el explorador.')
        else:
            from pattern_explorer import render_pattern_explorer
            render_pattern_explorer(folder)
        return
    with st.container(key='p07_training'):
        render_training(folder)


def render_training(folder):
    st.html('''<header class="p07-hero">
        <p class="p07-eyebrow">Encuestas GSHS</p>
        <h1>Entrenamiento con microdatos reales</h1>
        <p>Ideación autorreportada en 12 meses en adolescentes escolarizados de 13 a 15 años.</p>
    </header>''')
    st.session_state['displayed_results']=str(folder) if folder else None
    audit = load(folder, 'auditoria_cohorte.csv') if folder else pd.DataFrame()
    if audit.empty:
        st.info('Ve a 7 · Modelos para preparar los datos y ejecutar el entrenamiento.')
    else:
        with st.container(key='p07_overview'):
            a, b, c = st.columns(3)
            a.metric('Estudiantes elegibles', f'{int(audit.n_eligible.sum()):,}')
            b.metric('Países GSHS', len(audit))
            c.metric('Respuestas positivas Y1', f'{int(audit.events.sum()):,}')
        st.caption('Ejecución visible: '+('entrenamiento real inicial verificado' if folder==BASE else folder.name))
        st.info('Resultados de investigación sobre encuestas transversales. No constituyen diagnóstico ni pronóstico individual. Los conteos no son estimaciones poblacionales.')
    tabs=st.tabs(['1 · Ingesta','2 · Calidad','3 · Variables','4 · EDA','5 · Calibración','6 · Comparaciones e informe','7 · Modelos'])
    with tabs[6]:
        with st.container(key='p07_retraining'):
            training_controls()
    if audit.empty:
        return
    with tabs[0]:
        st.subheader('Fuentes y trazabilidad')
        survey_sources(folder, audit)
        st.write('Ingesta desde datasets y diccionarios de entrenamiento_real/metadata. El flujo GSHS no requiere los archivos grandes de revision_p07.')
        st.caption('PeNSE y los indicadores macroeconómicos no se incorporan al entrenamiento de Y1.')
        from protocol_views import inspect_import
        inspect_import(folder)
    with tabs[1]:
        st.subheader('Selección de cohorte y faltantes')
        st.dataframe(audit,hide_index=True,width='stretch')
        st.write('Edades observadas 13–15, Y1 respondido, dominios de respuesta válidos, pesos positivos y PSU/estratos completos. Las etiquetas ausentes no se imputan.')
        st.dataframe(load(folder,'faltantes.csv'),hide_index=True,width='stretch')
        st.caption('Compare la proporción ausente por variable y encuesta. Una pregunta omitida por diseño limita las comparaciones entre países.')
    with tabs[2]:
        st.subheader('Variables comparables y bloqueo de fugas')
        st.write('Ocho países: edad, sexo, hambre, agresiones, peleas, amistades, alcohol, actividad física y sedentarismo. Siete países: añade absentismo, apoyo de pares y tres preguntas parentales; excluye Guatemala por ausencia de esas preguntas.')
        st.caption('País, año, identificadores, diseño y etiquetas Y1–Y5 quedan fuera de X. Codificación de categorías y ausencias dentro de cada fold.')
        st.dataframe(load(folder,'armonizacion.csv'),hide_index=True,width='stretch')
        st.caption('Cada fila vincula una pregunta nacional con la variable común. Revise su equivalencia antes de comparar encuestas.')
    with tabs[3]:
        st.subheader('Distribución observada por país')
        eda=load(folder,'eda_por_pais.csv')
        if eda.empty:
            d=load(folder,'cohorte_gshs_8paises.csv');rows=[]
            for country,g in d.groupby('country_iso3'):
                rows.append(dict(country=country,n=len(g),events=int(g.y1_ideation.sum()),prevalence_unweighted=g.y1_ideation.mean(),prevalence_weighted=np.average(g.y1_ideation,weights=g.weight)))
            eda=pd.DataFrame(rows)
        st.dataframe(eda,hide_index=True,width='stretch')
        st.caption('n y eventos son conteos de la muestra. La prevalencia ponderada estima la proporción con Y1 entre los estudiantes representados por cada encuesta elegible.')
        if not eda.empty:chart(px.bar(eda,x='country',y='prevalence_weighted',title='Y1 observado, ponderado dentro de país'),width='stretch')
        st.caption('Descriptivo de cada encuesta, sin intervalos de diseño ni interpretación causal.')
    with tabs[6]:
        st.divider()
        st.subheader('Evaluación externa por país')
        experiment=st.selectbox('Experimento',['gshs8_core','gshs7_familiar','gshs5_core','gshs5_familiar'])
        summary=load(folder,experiment+'/resumen_macro.csv')
        st.dataframe(summary[['model','ap','roc_auc','brier','sensitivity','specificity']],hide_index=True,width='stretch')
        st.caption('Cada país pesa igual en el promedio. AP y AUC mayores indican mejor ordenamiento. Un Brier menor indica menor error cuadrático de las probabilidades.')
        st.dataframe(load(folder,experiment+'/metricas_por_pais.csv'),hide_index=True,width='stretch')
        st.caption('AP = average precision. Métricas macro por país, ponderadas dentro de país. Selección interna agrupada. Cinco familias para el núcleo común; logística para el bloque familiar. No se elige el modelo mirando su resultado externo.')
        st.write('Ajuste final seleccionado internamente:',read_json(folder/'ajuste_final.json',{}).get('selected','No disponible'))
        st.download_button('Descargar métricas agregadas',summary.to_csv(index=False).encode(),file_name=experiment+'_resumen.csv',mime='text/csv')
    with tabs[4]:
        st.subheader('Calibración y subgrupos — núcleo de ocho países')
        st.caption('Calibrador y umbral de costo FN:FP=3:1 ajustados internamente. El país externo no participa del ajuste. Solo se muestran agregados.')
        predictions=load(folder,'gshs8_core/predicciones_externas.csv')
        chosen=predictions[predictions.selected].copy()
        chosen['w']=chosen.weight/chosen.groupby('country_iso3').weight.transform('sum')
        chosen['bin']=pd.cut(chosen.p_calibrated,bins=np.linspace(0,1,11),include_lowest=True)
        points=[]
        for label,g in chosen.groupby('bin',observed=True):
            if len(g)>=20:points.append(dict(bin=str(label),n=len(g),predicted=np.average(g.p_calibrated,weights=g.w),observed=np.average(g.y1_ideation,weights=g.w)))
        chart(px.scatter(pd.DataFrame(points),x='predicted',y='observed',size='n',hover_data=['bin'],range_x=[0,1],range_y=[0,1],title='Calibración descriptiva de la selección anidada'),width='stretch')
        st.caption('Por encima de la diagonal imaginaria, el modelo subestima la proporción observada. El tamaño indica cuántos estudiantes hay en cada intervalo, sin IC. Cada país tiene igual masa total.')
        groups=[]
        for (country,sex),g in chosen.groupby(['country_iso3','sex_code']):
            y=g.y1_ideation.to_numpy();p=(g.p_calibrated>=g.threshold).to_numpy();w=g.weight.to_numpy()
            pos=w[y==1].sum();neg=w[y==0].sum()
            groups.append(dict(country=country,sex_code=sex,n=len(g),events=int(y.sum()),fnr=float(w[(y==1)&~p].sum()/pos) if pos else None,fpr=float(w[(y==0)&p].sum()/neg) if neg else None))
        st.dataframe(pd.DataFrame(groups),hide_index=True,width='stretch')
        st.caption('Sexo: 1=masculino, 2=femenino. Diferencias descriptivas; no se concluye equidad a partir de una cifra.')
    with tabs[5]:
        from protocol_views import render_protocol
        render_protocol(folder)
        with st.expander('Comparaciones pareadas de cinco países y reproducibilidad'):
            st.dataframe(load(folder,'comparaciones_pareadas.csv'),hide_index=True,width='stretch')
            st.caption('Mismos estudiantes de prueba de cinco países. B−A; 500 remuestreos de PSU dentro de estrato. IC condicionales a países y modelos fijos.')
            st.json(read_json(folder/'verificaciones.json',{}),expanded=False)
            for name,label,mime in [('INFORME_ENTRENAMIENTO.md','Descargar informe','text/markdown'),('receta_predefinida.json','Descargar receta','application/json'),('comparaciones_pareadas.csv','Descargar comparaciones','text/csv')]:
                p=folder/name
                if p.exists():st.download_button(label,p.read_bytes(),file_name=p.name,mime=mime)
        yrbs=load(folder,'yrbs_temporal/evaluacion_2025.csv')
        if not yrbs.empty:
            with st.expander('Réplica independiente YRBS — EE. UU.'):
                st.write('Desarrollo 2021, calibración 2023 y prueba 2025. No se mezcla con GSHS.')
                st.dataframe(yrbs,hide_index=True,width='stretch')
                st.caption('Las métricas resumen el desempeño en la encuesta estadounidense de 2025, posterior al ajuste. Diferencias con GSHS también pueden reflejar cambios de población o cuestionario.')

