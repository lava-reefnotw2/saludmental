"""Launch the same empirical workflow used by the Streamlit training button."""
from pathlib import Path
import sys,time,argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
if (ROOT/'.venv_training').exists():sys.path.insert(0,str(ROOT/'.venv_training'))
from training_jobs import start_run,job_status
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--yrbs',action='store_true',help='Include the independent US replication')
    parser.add_argument('--background',action='store_true')
    args=parser.parse_args()
    run_id=start_run(args.yrbs)
    print('Ejecución iniciada:',run_id,flush=True)
    if not args.background:
        last=None
        while True:
            status=job_status()
            current=(status.get('state'),status.get('phase'),status.get('message'))
            if current!=last:print(status.get('message'),flush=True);last=current
            if status.get('state')=='completed':break
            if status.get('state')=='failed':sys.exit(1)
            time.sleep(3)
