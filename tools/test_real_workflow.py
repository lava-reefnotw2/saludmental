"""Integration checks; --start launches a genuine full run through the UI button."""
from pathlib import Path
import sys,tempfile,json,os
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.venv_training'));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
os.chdir(ROOT)
import real_data_training as rt
import training_jobs as jobs
import run_real_workflow as workflow
from streamlit.testing.v1 import AppTest
from discussion_figures import COUNTRIES, VIEWS, comparison
from protocol_views import external_figures

jobs.preflight(False)
before=jobs.active_results()
with tempfile.TemporaryDirectory(dir=ROOT/'.run') as tmp:
    tmp=Path(tmp)
    with patch.object(rt,'OUT',tmp/'cohort'):
        df=rt.prepare()
        assert len(df)==50883 and df.record_key.is_unique
        assert int(df.y1_ideation.sum())==10683
        assert df.loc[df.country_iso3=='GTM',rt.FULL[9:]].isna().all().all()
        assert not set(rt.FULL)&set(c for c in df.columns if c.startswith('y'))
    with patch.object(jobs,'ROOT',tmp):
        try:jobs.preflight(False);raise AssertionError('Missing inputs accepted')
        except FileNotFoundError:pass
    fakebase=tmp/'failure';fakebase.mkdir()
    jobs.atomic_json(fakebase/'active.json',{'run_id':'previous_success'})
    (fakebase/'training.lock').touch()
    with patch.object(workflow,'BASE',fakebase),patch.object(workflow,'preflight',side_effect=ValueError('forced failure')):
        try:workflow.run('failure_case');raise AssertionError('Failure not raised')
        except ValueError:pass
    assert jobs.read_json(fakebase/'active.json')['run_id']=='previous_success'
    assert jobs.read_json(fakebase/'runs/failure_case/status.json')['state']=='failed'
    assert not (fakebase/'training.lock').exists()
os.environ.pop('SP07_TRAINING_OUT',None)
assert jobs.active_results()==before
app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run(timeout=60)
assert not app.exception,[e.message for e in app.exception]
assert len(app.tabs)==7
assert any('50,883' in str(m.value) for m in app.metric)
assert app.tabs[-1].label == '7 · Modelos'
assert [button.label for button in app.sidebar.button] == ['Entrenamiento', 'Explorador de patrones']
sources = app.tabs[0].dataframe[0].value
assert set(sources['País']) == set(COUNTRIES.values())
assert {Path(path).name for path in sources['Dataset']} == set(rt.FILES.values())
assert sources['Estudiantes elegibles'].sum() == len(df)
core = before / 'protocolo/nucleo_8'
for name in ('H1', 'transporte', 'equidad', 'curvas', 'calibracion', 'explicacion', 'metricas', 'decision_exploratoria'):
    assert set(rt.pd.read_csv(core / f'{name}.csv').country) >= set(COUNTRIES), name
for figure, _ in external_figures(core, sorted(COUNTRIES), True):
    assert {trace.name for trace in figure.data} == set(COUNTRIES.values()), figure.layout.title.text

def choose_first(label, options, **kwargs):
    return options[0]

figure, _ = comparison(before, VIEWS[0], choose_first, core=True)
assert set(figure.data[-1].y) == set(COUNTRIES.values()) | {'Media de los ocho países'}
app.button(key='p07_nav_explorer').click().run(timeout=60)
assert not app.exception,[e.message for e in app.exception]
assert not app.tabs and any(b.label == 'Cargar ejemplo ficticio' for b in app.button)
app.button(key='p07_nav_training').click().run(timeout=60)
assert not app.exception,[e.message for e in app.exception]
if '--start' in sys.argv:
    button=next(b for b in app.tabs[-1].button if 'Re-ejecutar' in b.label)
    button.click().run(timeout=60)
    assert not app.exception,[e.message for e in app.exception]
    status=jobs.job_status();assert status['state'] in ['starting','running'],status
    try:jobs.start_run(False);raise AssertionError('Duplicate run allowed')
    except RuntimeError:pass
    print('REAL_RUN_ID='+status['run_id'],flush=True)
print(json.dumps(dict(raw_cohort=True,no_target_leakage=True,missing_inputs_blocked=True,failed_run_preserves_active=True,ui_exceptions=0,tabs=7)),flush=True)
