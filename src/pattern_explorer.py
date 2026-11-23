"""Module 8 of the empirical dashboard: simulated profiles and model explanation."""

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from pattern_model import (
    ARTIFACT_NAME, DOMAINS, MODEL_ID, ModelUnavailable,
    explain_profile, load_model, score_profile,
)


FREQUENCY = ('Nunca', 'Rara vez', 'A veces', 'La mayoría de las veces', 'Siempre')
TIMES = ('0 veces', '1 vez', '2 o 3 veces', '4 o 5 veces', '6 o 7 veces',
         '8 o 9 veces', '10 u 11 veces', '12 veces o más')
QUESTIONS = {
    'age_years': ('Edad', 'Años cumplidos', ('13 años', '14 años', '15 años')),
    'sex_code': ('Sexo registrado', 'Categorías de la encuesta', ('Masculino', 'Femenino')),
    'food_insecurity': ('Hambre por falta de alimentos en casa', 'Frecuencia en los últimos 30 días', FREQUENCY),
    'physical_attack': ('Agresiones físicas recibidas', 'Número de veces en los últimos 12 meses', TIMES),
    'physical_fight': ('Participación en peleas físicas', 'Número de veces en los últimos 12 meses', TIMES),
    'close_friends': ('Amistades cercanas', 'Número de amistades', ('Ninguna', '1 amistad', '2 amistades', '3 o más')),
    'alcohol_30d': ('Consumo de alcohol', 'Días con al menos una bebida alcohólica en los últimos 30 días',
                    ('0 días', '1 o 2 días', '3 a 5 días', '6 a 9 días', '10 a 19 días', '20 a 29 días', 'Los 30 días')),
    'physical_activity': ('Actividad física de al menos 60 minutos', 'Número de días en los últimos 7 días',
                          tuple('1 día' if n == 1 else f'{n} días' for n in range(8))),
    'sedentary_time': ('Tiempo sentado en actividades sedentarias', 'Horas durante un día habitual',
                       ('Menos de 1 hora', '1 a 2 horas', '3 a 4 horas', '5 a 6 horas', '7 a 8 horas', 'Más de 8 horas')),
}
EXAMPLE = dict(zip(DOMAINS, (14, 2, 3, 2, 1, 3, 2, 4, 3)))
COUNTRIES = {'ARG': 'Argentina', 'BOL': 'Bolivia', 'GTM': 'Guatemala', 'HND': 'Honduras',
             'PAN': 'Panamá', 'PER': 'Perú', 'PRY': 'Paraguay', 'URY': 'Uruguay'}


@st.cache_resource(show_spinner=False, max_entries=2)
def _cached_model(path, revision):
    return load_model(Path(path))


@st.cache_data(show_spinner=False, max_entries=4)
def _external_results(metrics_path, scores_path, revision):
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[metrics['model'].eq(MODEL_ID),
                          ['country', 'n', 'ap', 'roc_auc', 'brier', 'sensitivity', 'specificity']].copy()
    metrics['country'] = metrics['country'].map(COUNTRIES)
    scores = pd.read_csv(scores_path, usecols=[
        'country_iso3', 'sex_code', 'model', 'y1_ideation', 'weight', 'p_calibrated', 'threshold',
    ])
    scores = scores.loc[scores['model'].eq(MODEL_ID) & scores.sex_code.isin([1, 2])].copy()
    positive = scores.y1_ideation.eq(1)
    classified = scores.p_calibrated.ge(scores.threshold)
    scores['positive_weight'] = scores.weight.where(positive, 0)
    scores['negative_weight'] = scores.weight.where(~positive, 0)
    scores['missed_weight'] = scores.weight.where(positive & ~classified, 0)
    scores['flagged_weight'] = scores.weight.where(~positive & classified, 0)
    grouped = scores.groupby(['country_iso3', 'sex_code'], as_index=False)[[
        'positive_weight', 'negative_weight', 'missed_weight', 'flagged_weight',
    ]].sum()
    grouped['Falsos negativos'] = grouped.missed_weight / grouped.positive_weight.where(grouped.positive_weight.gt(0))
    grouped['Falsos positivos'] = grouped.flagged_weight / grouped.negative_weight.where(grouped.negative_weight.gt(0))
    grouped['País'] = grouped.country_iso3.map(COUNTRIES)
    grouped['Sexo registrado'] = grouped.sex_code.map({1: 'Masculino', 2: 'Femenino'})
    return metrics, grouped[['País', 'Sexo registrado', 'Falsos negativos', 'Falsos positivos']]


def _answer(key, code):
    return QUESTIONS[key][2][DOMAINS[key].index(code)]


def _set_profile(namespace, values):
    state = st.session_state
    for key in DOMAINS:
        state[f'{namespace}_{key}'] = values.get(key)
    for suffix in ('result', 'field', 'alternative'):
        state.pop(f'{namespace}_{suffix}', None)


def _clear_comparison(namespace):
    for suffix in ('field', 'alternative'):
        st.session_state.pop(f'{namespace}_{suffix}', None)


def _percentage(value):
    return f'{100 * value:.1f}'.replace('.', ',') + ' %'


def _profile_input(key, namespace):
    label, period, _ = QUESTIONS[key]
    response = st.selectbox(label, tuple(DOMAINS[key]), index=None,
                            placeholder='Selecciona una respuesta',
                            format_func=lambda code: _answer(key, code),
                            key=f'{namespace}_{key}')
    st.html(f'<p class="p07-period">{escape(period)}</p>')
    return response


def _render_score(result):
    st.html('<p class="p07-eyebrow">Perfil de referencia</p>')
    st.subheader('Puntuación del modelo')
    if result is None:
        st.html('''<div class="p07-empty">
            <span class="p07-empty-icon" aria-hidden="true">∿</span>
            <strong>Completa el perfil para ver su puntuación</strong>
            <span>También puedes cargar un ejemplo ficticio y evaluarlo.</span>
            </div>''')
        return
    score = result['score']
    st.html(f'''<div class="p07-score">
        <div class="p07-score-number">{_percentage(score)}</div>
        <div class="p07-track" aria-hidden="true"><span style="width:{100 * score:.4f}%"></span></div>
        <div class="p07-scale" aria-hidden="true"><span>0 %</span><span>100 %</span></div>
    </div>''')
    st.caption('Estimación calibrada del autorreporte de ideación en los 12 meses previos a la encuesta.')
    with st.expander('Respuestas del perfil evaluado'):
        st.table(pd.DataFrame([
            (QUESTIONS[key][0], _answer(key, code)) for key, code in result['profile'].items()
        ], columns=['Variable', 'Respuesta']))


def _render_attributions(effects):
    st.subheader('Qué influyó en la puntuación')
    st.caption('Aportes SHAP por respuesta sobre la escala interna de CatBoost. El gráfico describe la predicción del modelo.')
    maximum = max(abs(value) for value in effects.values()) or 1
    bars = []
    for key, value in sorted(effects.items(), key=lambda pair: -abs(pair[1])):
        side = 'increase' if value >= 0 else 'decrease'
        formatted = f'{value:+.3f}'.replace('.', ',').replace('-', '−')
        bars.append(f'''<li class="p07-effect">
            <span>{escape(QUESTIONS[key][0])}</span>
            <span class="p07-diverging" aria-hidden="true"><i class="{side}" style="width:{50 * abs(value) / maximum:.4f}%"></i></span>
            <span class="p07-effect-value">{formatted}</span>
        </li>''')
    st.html('<div class="p07-effect-key"><span>← Reduce la salida</span><span>Aumenta la salida →</span></div>'
            '<ul class="p07-effects" aria-label="Aportes SHAP por variable">' + ''.join(bars) + '</ul>')


def _render_comparison(namespace, artifact, result):
    st.subheader('Comparar una respuesta')
    st.caption('Cambia una respuesta del perfil de referencia para observar cómo cambia la puntuación.')
    state = st.session_state
    field = st.selectbox('Variable a cambiar', tuple(DOMAINS), index=None,
                         format_func=lambda key: QUESTIONS[key][0], key=f'{namespace}_field',
                         on_change=lambda: state.pop(f'{namespace}_alternative', None))
    if field is None:
        return
    current = result['profile'][field]
    st.caption(f'Respuesta de referencia: {_answer(field, current)}')
    alternate = st.selectbox('Respuesta alternativa', [code for code in DOMAINS[field] if code != current],
                             index=None, format_func=lambda code: _answer(field, code),
                             key=f'{namespace}_alternative')
    if alternate is None:
        return
    scenario = {**result['profile'], field: alternate}
    score = score_profile(artifact, scenario)
    difference = f'{100 * (score - result["score"]):+.1f}'.replace('.', ',').replace('-', '−')
    st.metric('Puntuación del perfil alternativo', _percentage(score), delta=f'{difference} puntos porcentuales',
              delta_color='off', border=True)
    st.caption('La diferencia compara dos salidas del modelo con las demás respuestas fijas.')


def _render_validation(folder):
    with st.expander('Ver validación por país y errores por sexo'):
        metrics_path = folder / 'gshs8_core/metricas_por_pais.csv'
        scores_path = folder / 'gshs8_core/predicciones_externas.csv'
        if not metrics_path.is_file() or not scores_path.is_file():
            st.info('La ejecución activa no tiene archivos de validación por país.')
            return
        metrics, errors = _external_results(str(metrics_path), str(scores_path),
                                           (metrics_path.stat().st_mtime_ns, scores_path.stat().st_mtime_ns))
        st.caption('Cada fila resume CatBoost evaluado en una encuesta excluida del ajuste de ese pliegue. Las métricas se ponderan dentro del país.')
        st.dataframe(metrics, hide_index=True, column_config={
            'country': 'País', 'n': 'Estudiantes', 'ap': st.column_config.NumberColumn('Precisión promedio', format='%.3f'),
            'roc_auc': st.column_config.NumberColumn('ROC-AUC', format='%.3f'),
            'brier': st.column_config.NumberColumn('Brier', format='%.3f'),
            'sensitivity': st.column_config.NumberColumn('Sensibilidad', format='percent'),
            'specificity': st.column_config.NumberColumn('Especificidad', format='percent'),
        })
        st.caption('Los errores por sexo emplean los umbrales de investigación de cada pliegue. Falsos negativos: proporción de positivos omitidos. Falsos positivos: proporción de negativos señalados.')
        st.dataframe(errors, hide_index=True, column_config={
            name: st.column_config.NumberColumn(name, format='percent')
            for name in ('Falsos negativos', 'Falsos positivos')
        })


def render_pattern_explorer(folder):
    st.html(Path(__file__).with_name('pattern_explorer.css'))
    with st.container(key='p07_explorer'):
        st.html('''<header class="p07-hero">
            <p class="p07-eyebrow">Simulación de investigación</p>
            <h1>Explorador de patrones</h1>
            <p>Explora cómo las respuestas de un perfil ficticio de 13 a 15 años influyen en la estimación de ideación suicida autorreportada.</p>
            <div class="p07-tags"><span>CatBoost</span><span>Encuestas GSHS</span><span>9 variables</span></div>
        </header>''')
        st.caption('Uso de investigación con perfiles simulados. La salida carece de validación para diagnóstico o evaluación clínica individual.')

        path = folder / ARTIFACT_NAME
        if not path.is_file():
            st.info('El modelo CatBoost todavía no está disponible en esta ejecución.')
            return
        revision = path.stat().st_mtime_ns
        namespace = 'p07_profile'
        version = (str(path), revision)
        if st.session_state.get('p07_artifact_version') != version:
            _set_profile(namespace, {})
            st.session_state['p07_artifact_version'] = version
        try:
            artifact = _cached_model(str(path), revision)
        except ModelUnavailable as error:
            st.warning(str(error))
            return
        except (OSError, ValueError, ImportError, EOFError):
            st.error('No se pudo abrir el modelo guardado de la ejecución activa.')
            return
        st.caption(f'Ejecución del modelo · {folder.name}')

        form_column, score_column = st.columns([3, 2], gap='medium')
        with form_column, st.container(key='p07_profile_card'):
            st.subheader('Perfil simulado')
            st.caption('Selecciona una respuesta en cada campo.')
            example_column, clear_column = st.columns(2)
            example_column.button('Cargar ejemplo ficticio', on_click=_set_profile, args=(namespace, EXAMPLE), width='stretch')
            clear_column.button('Limpiar', on_click=_set_profile, args=(namespace, {}), width='stretch')
            with st.form('p07_profile_form', border=False):
                profile = {}
                keys = tuple(QUESTIONS)
                for index in range(0, len(keys) - 1, 2):
                    for column, key in zip(st.columns(2), keys[index:index + 2]):
                        with column:
                            profile[key] = _profile_input(key, namespace)
                profile[keys[-1]] = _profile_input(keys[-1], namespace)
                submitted = st.form_submit_button('Evaluar perfil', type='primary', width='stretch')
            if submitted:
                _clear_comparison(namespace)
                st.session_state.pop(f'{namespace}_result', None)
                if any(code is None for code in profile.values()):
                    st.warning('Completa las nueve respuestas para obtener la puntuación.')
                else:
                    try:
                        score, effects = explain_profile(artifact, profile)
                    except (ValueError, RuntimeError):
                        st.error('No se pudo evaluar el perfil con esta versión del modelo.')
                    else:
                        st.session_state[f'{namespace}_result'] = {'profile': profile, 'score': score, 'effects': effects}

        result = st.session_state.get(f'{namespace}_result')
        with score_column, st.container(key='p07_score_card'):
            _render_score(result)
            if result is not None:
                st.divider()
                _render_comparison(namespace, artifact, result)
        if result is not None:
            with st.container(key='p07_explanation_card'):
                _render_attributions(result['effects'])
        _render_validation(folder)
