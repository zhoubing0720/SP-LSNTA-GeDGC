# -*- coding: utf-8 -*-
"""Serial independent queue; no model math or writes to old results."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,csv,traceback,math
BASE=Path(__file__).resolve().parent
P=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def log(s):
 with (BASE/'queue.log').open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+s+'\n')
 if sys.stdout is not None:print(s,flush=True)
def metrics_path(t):return BASE/'runs'/t['dataset']/t['config']/('seed_'+str(t['seed']))/'metrics.json'
def valid(v,t):
 assert (v['Dataset'],v['config'],v['seed'])==(t['dataset'],t['config'],t['seed'])
 assert v['fingerprints']['protocol_sha256']==digest(BASE/'protocol.json')
 assert v['fingerprints']['data_sha256']==P['datasets'][t['dataset']]['sha256']
 assert v['pretraining_performed'] and v['fresh_initialization']
 assert v['fingerprints']['execution_sha256']==P['execution_sha256']
 for k in ['ACC','NMI','ARI','Purity','TIME_full_seconds','peak_gpu_allocated_MiB','peak_gpu_reserved_MiB']:assert math.isfinite(v[k])
 assert v['peak_gpu_allocated_MiB']>0

def summarize():
 rows=[]
 for t in P['tasks']:
  p=metrics_path(t)
  if p.exists():
   v=json.loads(p.read_text(encoding='utf-8'));valid(v,t)
   rows.append({k:v[k] for k in ['Dataset','config','model','seed','ACC','NMI','ARI','Purity','TIME_full_seconds','TIME_without_loading_seconds','peak_gpu_allocated_MiB','peak_gpu_reserved_MiB','joint_epochs']})
 if rows:
  dest=BASE/'summary';dest.mkdir(exist_ok=True)
  with (dest/'per_seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  from collections import defaultdict
  import statistics
  groups=defaultdict(list)
  for r in rows:groups[(r['Dataset'],r['config'])].append(r)
  summary=[]
  for (ds,cfg),vals in groups.items():
   if sorted(v['seed'] for v in vals)!=P['seeds']:continue
   row={'Dataset':ds,'config':cfg,'n':5}
   for k in ['ACC','NMI','ARI','Purity','TIME_full_seconds','peak_gpu_allocated_MiB','peak_gpu_reserved_MiB']:
    x=[r[k] for r in vals];row[k+'_mean']=statistics.mean(x);row[k+'_sample_SD']=statistics.stdev(x)
   summary.append(row)
  if summary:
   with (dest/'complete_groups_mean_SD.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 return len(rows)

def main():
 for rel,h in P['execution_sha256'].items():assert digest(BASE/rel)==h,rel
 lock=BASE/'queue.lock'
 try:fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError:raise RuntimeError('Queue lock exists; check previous process before restarting')
 os.write(fd,str(os.getpid()).encode());os.close(fd)
 atomic(BASE/'batch_started.json',{'pid':os.getpid(),'started_unix':time.time(),'protocol_sha256':digest(BASE/'protocol.json'),'total_tasks':len(P['tasks'])})
 try:
  count=summarize()
  for index,task in enumerate(P['tasks']):
   path=metrics_path(task);path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists():valid(json.loads(path.read_text(encoding='utf-8')),task);continue
   if (path.parent/'Intermediate_data').exists():raise RuntimeError('Inspect incomplete task before fresh restart: '+str(path.parent))
   log('START '+json.dumps(task));status={'status':'running','completed':count,'total':len(P['tasks']),'task_index':index+1,'current_task':task,'scheduler_pid':os.getpid(),'time':time.time()}
   atomic(BASE/'queue_status.json',status)
   env=os.environ.copy();env.update(PYTHONHASHSEED=str(task['seed']),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',GEDGC_SEED=str(task['seed']))
   attempt=1
   while (path.parent/('run_%02d.log'%attempt)).exists():attempt+=1
   with (path.parent/('run_%02d.log'%attempt)).open('x',encoding='utf-8') as stream:
    proc=subprocess.Popen([P['python'],'-u','-B',str(BASE/'run_worker.py'),'--dataset',task['dataset'],'--config',task['config'],'--seed',str(task['seed'])],cwd=str(BASE),env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    status['worker_pid']=proc.pid;atomic(BASE/'queue_status.json',status);code=proc.wait()
   if code!=0:raise RuntimeError('Worker failed, exit '+str(code)+': '+json.dumps(task))
   valid(json.loads(path.read_text(encoding='utf-8')),task);count=summarize();log('DONE '+str(count)+'/'+str(len(P['tasks'])))
  atomic(BASE/'queue_status.json',{'status':'complete','completed':count,'total':len(P['tasks']),'time':time.time()});log('ALL COMPLETE')
 except Exception as e:
  atomic(BASE/'queue_status.json',{'status':'failed','error':str(e),'traceback':traceback.format_exc(),'completed':summarize(),'total':len(P['tasks']),'time':time.time()});log('FAILED '+str(e));raise
 finally:
  if lock.exists():lock.unlink()
if __name__=='__main__':main()
