# -*- coding: utf-8 -*-
from pathlib import Path
import json
BASE=Path(__file__).resolve().parent
p=BASE/'queue_status.json'
if not p.exists():print('Not started')
else:
 s=json.loads(p.read_text(encoding='utf-8'));print(json.dumps(s,ensure_ascii=False,indent=2))
 t=s.get('current_task')
 if t:
  run=BASE/'runs'/t['dataset']/t['config']/('seed_'+str(t['seed']))
  state=run/'progress.json'
  if state.exists():print('Worker phase:',state.read_text(encoding='utf-8'))
  logs=sorted(run.glob('run_*.log'))
  if logs:
   text=logs[-1].read_text(encoding='utf-8',errors='replace').replace('\r','\n')
   print('Latest worker log:',str(logs[-1]));print('\n'.join(text.splitlines()[-5:]))
