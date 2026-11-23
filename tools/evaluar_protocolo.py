from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'.venv_training'),str(root/'src')]
from training_jobs import active_results
from protocol_analysis import analyse,threadpool_limits
if __name__=='__main__':
    with threadpool_limits(limits=4):print(analyse(active_results()))
