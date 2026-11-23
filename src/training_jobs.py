"""Process lifecycle and atomic publication of empirical training runs."""
from pathlib import Path
import json, os, subprocess, sys, uuid
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'entrenamiento_real'

def read_json(path, default=None):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError): return default

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temp,path)

def alive(pid):
    if not pid:return False
    if os.name=='nt':
        import ctypes
        handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        try:
            code=ctypes.c_ulong()
            return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:ctypes.windll.kernel32.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except OSError:return False

def job_status():
    job=read_json(BASE/'job.json',{})
    if not job:return {}
    state=read_json(BASE/'runs'/job['run_id']/'status.json',job)
    if state.get('state') in ['starting','running'] and not alive(job.get('pid')):
        state.update(state='failed',message='El proceso terminó sin completar la ejecución. Se conservan los resultados anteriores.')
        atomic_json(BASE/'runs'/job['run_id']/'status.json',state)
        (BASE/'training.lock').unlink(missing_ok=True)
    return {**job,**state}

def active_results():
    active=read_json(BASE/'active.json',{})
    if active.get('run_id'):
        return BASE/'runs'/active['run_id']
    # The previously verified empirical run is the initial dataset, never data/ or artifacts/.
    if (BASE/'verificaciones.json').exists() and (BASE/'gshs8_core/resumen_macro.csv').exists():return BASE
    return None

def preflight(include_yrbs=False):
    from real_data_training import FILES, METADATA
    needed=[ROOT/'datasets'/f for f in FILES.values()]+[METADATA/f'{tag}_diccionario.csv' for tag in FILES]
    if include_yrbs:
        for year in [2021,2023,2025]:
            needed += [ROOT/f'revision_p07/datasets/tabulares/YRBS_{year}_completo.csv',ROOT/f'revision_p07/datasets/tabulares/YRBS_{year}_diccionario.csv',ROOT/f'revision_p07/datasets/originales/{year}XXH-Formats-Program.sas']
    missing=[str(p.relative_to(ROOT)) for p in needed if not p.is_file()]
    if missing:raise FileNotFoundError('Faltan archivos necesarios: '+', '.join(missing))

def start_run(include_yrbs=False):
    preflight(include_yrbs)
    BASE.mkdir(exist_ok=True)
    job_status() # Recover a stale lock only when its recorded worker is no longer alive.
    try:fd=os.open(BASE/'training.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Ya hay un entrenamiento en curso. Consulta su avance.')
    os.close(fd)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=BASE/'runs'/run_id;out.mkdir(parents=True)
    state=dict(run_id=run_id,state='starting',phase=0,message='Preparando entrenamiento real',include_yrbs=include_yrbs,started_at=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'status.json',state)
    args=[sys.executable,str(ROOT/'tools/run_real_workflow.py'),'--run-id',run_id]
    if include_yrbs:args.append('--yrbs')
    try:
        with (out/'ejecucion.log').open('w',encoding='utf-8') as log:
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1'}
            process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags,close_fds=True,env=env)
        atomic_json(BASE/'job.json',{**state,'pid':process.pid})
    except Exception:
        (BASE/'training.lock').unlink(missing_ok=True)
        raise
    return run_id
