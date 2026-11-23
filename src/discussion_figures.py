"""Aggregate figures for discussion of the GSHS results."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

COUNTRIES = dict(ARG='Argentina', BOL='Bolivia', GTM='Guatemala', HND='Honduras',
                 PAN='Panamá', PER='Perú', PRY='Paraguay', URY='Uruguay')
QUESTIONS = {
    'parental_understanding': 'Mis padres comprendieron mis problemas',
    'parental_knowledge': 'Mis padres sabían qué hacía en mi tiempo libre',
    'peer_kindness': 'Mis compañeros fueron amables y me ayudaron',
    'food_insecurity': 'Pasé hambre por falta de alimentos en casa',
}
FREQUENCY = ['Nunca', 'Rara vez', 'A veces', 'La mayoría de las veces', 'Siempre']
VIEWS = ['Aporte de los predictores', 'Familia, escuela y alimentación', 'Transporte a otro país',
         'Brechas por sexo y edad', 'Comparación de algoritmos', 'Escenario de apoyo']

def country_labels(frame):
    frame = frame.copy()
    labels = {**COUNTRIES, 'Macro · cinco países fijos': 'Media de los cinco países',
              'MACRO_paises_fijos': 'Media de los cinco países',
              'Macro · ocho países': 'Media de los ocho países'}
    frame['País'] = frame.country.map(labels).fillna(frame.country)
    return frame

def interval_figure(frame, title, axis):
    fig = go.Figure()
    for row in frame.itertuples(index=False):
        fig.add_trace(go.Scatter(x=[row.low, row.high], y=[row.group, row.group], mode='lines',
                                 line=dict(color='#608198', width=3), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=frame.value, y=frame.group, mode='markers', marker=dict(color='#005F73', size=10),
                             customdata=frame[['low','high']], showlegend=False,
                             hovertemplate='%{y}<br>Estimación %{x:.3f}<br>IC95% [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>'))
    fig.add_vline(x=0, line_dash='dot', line_color='gray')
    fig.update_layout(title=title, xaxis_title=axis, yaxis_title=None)
    return fig

@st.cache_data(show_spinner=False)
def observed(path,stamp,question,country):
    fields=['country_iso3','stratum','psu','weight','y1_ideation',question]
    frame=pd.read_csv(path,usecols=fields)
    frame=frame[frame.country_iso3==country].copy()
    rows=[]
    for code,label in enumerate(FREQUENCY,1):
        domain=frame[question].eq(code)
        n=int(domain.sum())
        events=int(frame.loc[domain,'y1_ideation'].sum())
        if min(events,n-events)<10:continue
        w=frame.weight
        denominator=w[domain].sum()
        rate=(w[domain]*frame.loc[domain,'y1_ideation']).sum()/denominator
        # Ratio linearization: retain all eligible PSUs, including out-of-domain zeros.
        influence=w*domain*(frame.y1_ideation-rate)/denominator
        clusters=influence.groupby([frame.stratum,frame.psu]).sum()
        strata=clusters.groupby(level=0)
        variance=(strata.var()*strata.size()).sum()
        se=np.sqrt(variance)
        rows.append(dict(respuesta=label,n=n,eventos=events,observada=rate,
                         low=max(0,rate-1.96*se),high=min(1,rate+1.96*se)))
    return pd.DataFrame(rows)

def comparison(run, view, choose, *, core=False):
    protocol = run / 'protocolo'
    folder = protocol / 'nucleo_8' if core else protocol
    block_label = 'núcleo común de ocho países' if core else 'bloque familiar de cinco países'
    if view==VIEWS[0]:
        options=['Bloque completo frente a edad y sexo','Añadir cinco preguntas familiares y escolares al núcleo común',
                 'Ampliar de cinco a ocho países con variables comunes','Ampliar de cinco a siete países con el bloque familiar']
        selection = 'Núcleo común frente a edad y sexo' if core else choose('Comparación pareada',options)
        if core or selection==options[0]:
            frame=pd.read_csv(folder/'H1.csv').rename(columns={'delta_AP':'value'})
        else:
            compare=pd.read_csv(run/'comparaciones_pareadas.csv')
            key=['core_vs_familiar_5paises','5_vs_8_mismas_variables_logistica','5_vs_7_familiar'][options.index(selection)-1]
            frame=compare[compare.comparison==key].rename(columns={'delta':'value','ci_low':'low','ci_high':'high'})
        frame=country_labels(frame).rename(columns={'País':'group'})
        fig=interval_figure(frame,selection,'Diferencia de AP · positivo favorece el cambio')
        count = 'ocho' if core else 'cinco'
        note=(f'Cada punto compara los mismos estudiantes de prueba. La media da igual peso a los {count} países. IC95% de 500 remuestreos de conglomerados dentro de estratos, condicionados a los modelos ajustados.')
    elif view==VIEWS[1]:
        question=choose('Pregunta de encuesta',list(QUESTIONS),format_func=QUESTIONS.get)
        countries=[c for c in COUNTRIES if c!='GTM' or question=='food_insecurity']
        country=choose('Encuesta nacional',countries,format_func=COUNTRIES.get)
        path=run/'cohorte_gshs_8paises.csv'
        frame=observed(str(path),path.stat().st_mtime_ns,question,country)
        if frame.empty:
            return None, 'Ninguna categoría reúne el mínimo de eventos y negativos.'
        frame['más']=frame.high-frame.observada
        frame['menos']=frame.observada-frame.low
        fig=px.scatter(frame,x='respuesta',y='observada',error_y='más',error_y_minus='menos',
                       hover_data=['n','eventos'],category_orders={'respuesta':FREQUENCY},
                       title=f'Ideación autorreportada según respuesta · {COUNTRIES[country]}',
                       labels={'respuesta':'Frecuencia en los últimos 30 días','observada':'Proporción ponderada con Y1'})
        fig.update_yaxes(tickformat='.0%',rangemode='tozero')
        note=('Estudiantes de 13–15 años con Y1 observado en los últimos 12 meses. Asociación descriptiva sin ajuste. IC95% aproximados por linealización dentro de la cohorte elegible. Los estratos con una sola PSU aportan cero a la varianza. Se omiten categorías con menos de diez positivos o diez negativos.')
        if question != 'food_insecurity':
            note += ' La encuesta de Guatemala carece de esta pregunta.'
    elif view==VIEWS[2]:
        metric=choose('Métrica de transporte',['AP','AUC','Brier','calibration_bias'])
        frame=country_labels(pd.read_csv(folder/'transporte.csv'))
        frame=frame[frame.metric==metric].sort_values('external_minus_development')
        long=frame.melt(id_vars=['País','external_minus_development'],value_vars=['development_oof','external'],var_name='Evaluación',value_name='Valor')
        long['Evaluación']=long['Evaluación'].map({'development_oof':'Desarrollo (OOF)','external':'País excluido'})
        fig=px.line(long,x='Evaluación',y='Valor',color='País',markers=True,hover_data=['external_minus_development'],title=f'{metric} al evaluar en otro país · {block_label}')
        note=('Cada línea une la media de los países de desarrollo con el resultado del país excluido. AP depende también de la prevalencia. Para Brier se prefieren valores menores y para el sesgo de calibración se busca cercanía a cero.')
    elif view==VIEWS[3]:
        contrast=choose('Grupos comparados',['Femenino menos masculino','14 menos 13 años','15 menos 13 años'])
        metric=choose('Medida de la brecha',['FNR','FPR','sensitivity','AP','calibration_bias'])
        field,comparison=[('sex_code',2),('age_years',14),('age_years',15)][['Femenino menos masculino','14 menos 13 años','15 menos 13 años'].index(contrast)]
        frame=country_labels(pd.read_csv(folder/'equidad.csv'))
        frame=frame[(frame.subgroup==field)&(frame.compared==comparison)&(frame.metric==metric)].rename(columns={'País':'group','difference':'value'})
        fig=interval_figure(frame,f'{contrast} · {metric} · {block_label}','Diferencia absoluta en escala de proporción (con signo)')
        note=('Cero indica igual estimación entre grupos. FNR es la proporción de positivos omitidos y FPR la de negativos señalados. Los intervalos por conglomerados reflejan incertidumbre dentro de cada país.')
    elif view==VIEWS[4]:
        frame=country_labels(pd.read_csv(protocol/'H4.csv'))
        names={'B1_random_forest':'Random Forest','B1_xgboost':'XGBoost','B1_catboost':'CatBoost','B1_mlp':'MLP','B1_logreg_C01':'Logística','ensamble_media':'Ensamble'}
        frame['Familia']=frame.model.map(names)
        matrix=frame.pivot(index='País',columns='Familia',values='delta_vs_logistica')
        limit=max(abs(matrix.to_numpy().min()),abs(matrix.to_numpy().max()),1e-6)
        fig=px.imshow(matrix,text_auto='.3f',aspect='auto',color_continuous_scale='RdBu',zmin=-limit,zmax=limit,
                      labels={'color':'ΔAP'},title='Cambio de AP frente a logística · núcleo común de ocho países')
        note=('Cada celda compara modelos en los mismos estudiantes del país excluido. Son diferencias puntuales. Las diferencias pequeñas requieren evaluar incertidumbre antes de proclamar una familia ganadora.')
    else:
        frame=country_labels(pd.read_csv(folder/'decision_exploratoria.csv'))
        country=choose('País del escenario',sorted(frame['País'].unique()))
        df=frame[frame['País']==country].rename(columns={'model':'Usar el modelo','all':'Invitar a todos','none':'No invitar'})
        fig=px.line(df,x='threshold',y=['Usar el modelo','Invitar a todos','No invitar'],markers=True,
                    labels={'threshold':'Umbral de probabilidad','value':'Beneficio neto supuesto','variable':'Estrategia'},title='Invitación voluntaria a apoyo · '+country)
        note=('Escenario exploratorio. Cada umbral fija el costo relativo de una invitación innecesaria frente a una omitida. Las curvas dependen de la prevalencia observada en la encuesta y de esos costos supuestos.')
    if view==VIEWS[0]:
        row=frame.iloc[-1]
        reading=f'{row.group}: ΔAP {row.value:+.3f} (IC95% {row.low:+.3f} a {row.high:+.3f}). Un valor positivo indica mejor precisión promedio al ordenar las respuestas positivas.'
    elif view==VIEWS[1]:
        a,b=frame.iloc[0],frame.iloc[-1]
        reading=f'Para «{QUESTIONS[question]}», la proporción con Y1 fue {a.observada:.1%} en «{a.respuesta}» y {b.observada:.1%} en «{b.respuesta}». Son grupos de estudiantes distintos.'
    elif view==VIEWS[2]:
        row=frame.loc[frame.external_minus_development.abs().idxmax()]
        reading=f'La mayor diferencia en magnitud aparece en {row["País"]}: {row.development_oof:.3f} en desarrollo y {row.external:.3f} en evaluación externa.'
    elif view==VIEWS[3]:
        row=frame.loc[frame.value.abs().idxmax()]
        reading=f'La mayor brecha en magnitud corresponde a {row.group}: {row.value*100:+.1f} puntos porcentuales ({contrast.lower()}, {metric}).'
    elif view==VIEWS[4]:
        values=frame.loc[frame.Familia!="Logística","delta_vs_logistica"]
        reading=f'Frente a la logística, las diferencias de AP van de {values.min():+.3f} a {values.max():+.3f} entre las combinaciones de país y algoritmo.'
    else:
        i=len(df)//2
        row=df.iloc[i]
        reading=f'En {country}, al umbral {row.threshold:.0%}, el beneficio neto estimado es {row["Usar el modelo"]:.3f} con el modelo, {row["Invitar a todos"]:.3f} al invitar a todos y {row["No invitar"]:.3f} sin invitaciones. No se midieron efectos de la intervención.'
    return fig, reading+' '+note


def discussion(run, *, core=False):
    from publication_charts import chart
    view=st.selectbox('Pregunta para la discusión',VIEWS,key='discussion_view')
    fig,note=comparison(run,view,st.selectbox,core=core)
    if fig is not None:
        chart(fig,width='stretch')
    st.caption(note)
