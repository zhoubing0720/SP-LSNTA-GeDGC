# -*- coding: utf-8 -*-
"""One fresh profiled task; retained scientific functions, independent output."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time,random,inspect,gc
BASE=Path(__file__).resolve().parent
P=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'))
for key,sub in [('NUMBA_CACHE_DIR','numba'),('MPLCONFIGDIR','matplotlib')]:
 p=BASE/'cache'/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.use_deterministic_algorithms(True,warn_only=True)
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 if '.' in name:
  parent,child=name.rsplit('.',1)
  if parent in sys.modules:setattr(sys.modules[parent],child,m)
 return m
class Timing:
 def __init__(self):self.rows=[]
 def call(self,name,fn,*a,**kw):
  torch.cuda.synchronize();t=time.perf_counter();before=torch.cuda.memory_allocated()
  try:return fn(*a,**kw)
  finally:
   torch.cuda.synchronize();self.rows.append({'stage':name,'seconds':time.perf_counter()-t,'allocated_before_bytes':before,'allocated_after_bytes':torch.cuda.memory_allocated(),'peak_allocated_cumulative_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_cumulative_bytes':torch.cuda.max_memory_reserved()})
 def wrap(self,name,fn):
  def wrapper(*a,**kw):return self.call(name,fn,*a,**kw)
  return wrapper

def main(args):
 config=P['configs'][args.config];ds=P['datasets'][args.dataset]
 sp_root=BASE/'snapshot/sp';orig_root=BASE/'snapshot/gedgc';nm_root=BASE/'snapshot/nmodal'
 sys.path.insert(0,str(sp_root))
 import module,tools
 vae_module=load('module.Multi_vae',(sp_root if config['sp'] else orig_root)/'module/Multi_vae.py')
 loader=load('tools.load_data',(nm_root if ds['nmodal'] else sp_root)/'tools/load_data.py')
 data_path=Path(ds['path'])
 if ds['nmodal']:
  loader._resolve_teaseq_path=lambda:data_path
  loader._resolve_dyngen_path=lambda:data_path
 else:loader._resolve_path=lambda name:data_path
 pretrain=load('pretrain',sp_root/'pretrain.py')
 train_path=orig_root/'train.py' if args.config=='A_GeDGC' else (nm_root if ds['nmodal'] else sp_root)/'train.py'
 train=load('train',train_path)
 from tools import Form_data
 from graph_embedding.spectralnet import SpectralNet
 assert 'seed' in inspect.signature(pretrain.pretrain_opt).parameters
 if args.config!='A_GeDGC':assert 'seed' in inspect.signature(train.train_opt).parameters
 if args.validate_interfaces:
  torch.manual_seed(2026)
  v=vae_module.Multiview_VAE([[5,256,3],[7,256,3]]).eval()
  x=[torch.randn(8,5),torch.randn(8,7)];mean,logvar=v.get_latent(x);z=mean.clone();recon=v.get_recon(z)
  assert mean.shape==(8,3) and torch.isfinite(logvar).all()
  for i in range(2):
   if config['sp']:
    vm,_=v.vaes[i].get_latent(x[i]);expected=v.vaes[i].get_recon(z+.1*(vm-mean))
   else:expected=v.vaes[i].get_recon(z)
   assert torch.allclose(recon[i],expected,atol=1e-6,rtol=1e-5)
  print(json.dumps({'dataset':args.dataset,'config':args.config,'sp':config['sp'],'model_class_file':str(vae_module.__file__),'loader_file':str(loader.__file__),'joint_module':str(train.__file__),'scientific_interface_check':'passed'}));return
 assert args.seed in P['seeds'] and os.environ.get('PYTHONHASHSEED')==str(args.seed)
 assert torch.cuda.is_available(),'CUDA required by retained gen_x'
 for rel,info in P['sources'].items():
  expected=info.get('snapshot_sha256_after_seed_parameterization',info['sha256']);assert digest(BASE/rel)==expected,rel
 assert digest(data_path)==ds['sha256'],'Input file changed'
 for rel,expected in P['execution_sha256'].items():assert digest(BASE/rel)==expected,rel
 run=BASE/'runs'/args.dataset/args.config/('seed_'+str(args.seed));run.mkdir(parents=True,exist_ok=True)
 if (run/'metrics.json').exists():raise RuntimeError('Completed task is immutable')
 if (run/'Intermediate_data').exists():raise RuntimeError('Partial weights cannot be reused for a cold-start run')
 os.chdir(str(run));dataname='experiment'
 (run/'Intermediate_data'/dataname).mkdir(parents=True);(run/'result'/dataname).mkdir(parents=True)
 info={'protocol_id':P['protocol_id'],'protocol_sha256':digest(BASE/'protocol.json'),'dataset':args.dataset,'config':args.config,'flags':config,'seed':args.seed,'data_path':str(data_path),'data_sha256':ds['sha256'],'fresh_pretraining':True,'source_snapshot':P['sources'],'execution_sha256':P['execution_sha256'],'python':sys.executable,'python_version':sys.version,'torch':torch.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'determinism':'all RNG seeded; deterministic algorithms warn_only; retain warnings','data_provenance_caveat':P['data_provenance_caveats'].get(args.dataset)}
 atomic(run/'run_info.json',info);atomic(run/'progress.json',{'stage':'starting','time':time.time()})
 timer=Timing();torch.cuda.init();seed_all(args.seed);torch.cuda.reset_peak_memory_stats()
 torch.cuda.synchronize();begin=time.perf_counter()
 print('EFFECTIVE CONFIGURATION',json.dumps(config),flush=True)
 raw,labels,c=timer.call('load_preprocess',loader.load_data,args.dataset)
 assert len(labels)==ds['cells'] and c==ds['classes']
 assert [int(x.shape[1]) for x in raw]==ds['feature_dims']
 assert all(torch.isfinite(x).all() for x in raw)
 atomic(run/'progress.json',{'stage':'pretraining','time':time.time()})
 pretrain.pretrain_vae=timer.wrap('VAE_pretrain',pretrain.pretrain_vae)
 pretrain.pretrain_classifier=timer.wrap('classifier_pretrain',pretrain.pretrain_classifier)
 seed_all(args.seed);timer.call('pretraining_total_including_GMM',pretrain.pretrain_opt,raw,labels,c,dataname,seed=args.seed)
 checkpoint=run/'Intermediate_data'/dataname
 atomic(run/'pretraining_completion.json',{'complete':True,'seed':args.seed,'fresh':True,'weights_sha256':{p.name:digest(p) for p in checkpoint.glob('pretrained_*.pkl')}})
 seed_all(args.seed)
 def shared():
  v=torch.load(str(checkpoint/'pretrained_vae.pkl')).cuda().eval();parts=[]
  from torch.utils.data import DataLoader
  with torch.no_grad():
   for batch in DataLoader(loader.Cell(raw),batch_size=1024,shuffle=False):
    m,_=v.get_latent([x.cuda().float() for x in batch]);parts.append(m.detach().cpu())
  v=v.cpu();del v;gc.collect();torch.cuda.empty_cache();return torch.cat(parts).float()
 graph_input=timer.call('shared_mean_for_graph',shared) if config['graph']=='latent' else torch.cat(raw,dim=1)
 spectral=SpectralNet(n_clusters=c,should_use_ae=True,should_use_siamese=True,ae_hiddens=[512,c],siamese_hiddens=[256,c],spectral_hiddens=[256,c])
 atomic(run/'progress.json',{'stage':'spectral_graph','time':time.time()})
 timer.call('SpectralNet_fit',spectral.fit,graph_input)
 distance=timer.call('SpectralNet_predict',spectral.predict,graph_input)
 if not torch.is_tensor(distance):distance=torch.as_tensor(distance)
 distance=distance.detach().cpu().float();torch.save(distance,str(checkpoint/'SpectralNet_D.pkl'))
 neighbors=timer.call('graph_neighbors',Form_data.cal_neighbors_D,distance,3)
 del distance,graph_input,spectral;gc.collect();torch.cuda.empty_cache()
 data=timer.call('form_graph_samples',Form_data.form_data,raw,labels,neighbors)
 raw.clear();del raw,neighbors;gc.collect()
 if args.config!='A_GeDGC':
  if config['fg_mnn']:train.aggregate_feature_gated_mnn=timer.wrap('FG_MNN',train.aggregate_feature_gated_mnn)
  else:
   def disabled_fg(data,*a,**kw):print('FG-MNN DISABLED',flush=True);return data
   train.aggregate_feature_gated_mnn=disabled_fg
  if config['uot']:train.aggregate_uot_no_gate=timer.wrap('UOT',train.aggregate_uot_no_gate)
  else:
   def disabled_uot(data,*a,**kw):print('UOT DISABLED',flush=True);return data
   train.aggregate_uot_no_gate=disabled_uot
 captured={};original_model=train.train_model;original_acc=train.cluster_acc;inference_start=[None]
 def joint(*a,**kw):
  atomic(run/'progress.json',{'stage':'joint_training','time':time.time()})
  result=timer.call('joint_optimization',original_model,*a,**kw);captured['epochs']=len(result[-1]);captured['loss_history']=result[-1]
  inference_start[0]=time.perf_counter();return result
 def capture_acc(pred,true,*a,**kw):
  result=original_acc(pred,true,*a,**kw)
  if inference_start[0] is not None:captured['pred']=np.asarray(pred).copy();captured['true']=np.asarray(true).copy()
  return result
 train.train_model=joint;train.cluster_acc=capture_acc
 seed_all(args.seed);atomic(run/'progress.json',{'stage':'structure_and_joint','time':time.time()})
 kwargs={} if args.config=='A_GeDGC' else {'seed':args.seed}
 scores=timer.call('structure_joint_inference_total',train.train_opt,data,labels,dataname,**kwargs)
 torch.cuda.synchronize();inference_seconds=time.perf_counter()-inference_start[0]
 assert len(captured['pred'])==len(labels) and np.array_equal(captured['true'],labels)
 np.savez_compressed(run/'predictions.npz',y_true=labels,y_pred=captured['pred'])
 atomic(run/'loss_history.json',{'joint_epochs':captured['epochs'],'loss':list(map(float,captured['loss_history']))})
 torch.cuda.synchronize();total=time.perf_counter()-begin
 loading=sum(r['seconds'] for r in timer.rows if r['stage']=='load_preprocess')
 metrics={'Dataset':args.dataset,'config':args.config,'model':'GeDGC' if args.config=='A_GeDGC' else 'SP-LSNTA-GeDGC' if args.config=='F_full' else args.config,'seed':args.seed,'ACC':float(scores[0]),'NMI':float(scores[1]),'ARI':float(scores[2]),'Purity':float(scores[3]),'FMI':float(scores[4]),'TIME_full_seconds':total,'TIME_without_loading_seconds':total-loading,'peak_gpu_allocated_MiB':torch.cuda.max_memory_allocated()/1048576,'peak_gpu_reserved_MiB':torch.cuda.max_memory_reserved()/1048576,'inference_export_seconds':inference_seconds,'joint_epochs':captured['epochs'],'pretraining_performed':True,'fresh_initialization':True,'fingerprints':{'protocol_sha256':info['protocol_sha256'],'data_sha256':info['data_sha256'],'execution_sha256':info['execution_sha256']},'timing':timer.rows,'performance_scope':P['runtime_scope'],'GPU':info['GPU'],'torch':info['torch'],'CUDA':info['CUDA']}
 import math
 for k in ['ACC','NMI','ARI','Purity','FMI','TIME_full_seconds','peak_gpu_allocated_MiB','peak_gpu_reserved_MiB']:assert math.isfinite(metrics[k]),k
 assert metrics['peak_gpu_reserved_MiB']>=metrics['peak_gpu_allocated_MiB']>0
 atomic(run/'metrics.json',metrics);atomic(run/'progress.json',{'stage':'complete','time':time.time()})
 print('COMPLETED',json.dumps({k:metrics[k] for k in ['Dataset','config','seed','ACC','NMI','ARI','Purity','TIME_full_seconds','peak_gpu_allocated_MiB','peak_gpu_reserved_MiB']}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=list(P['datasets']),required=True);parser.add_argument('--config',choices=list(P['configs']),required=True);parser.add_argument('--seed',type=int,required=True);parser.add_argument('--validate-interfaces',action='store_true');args=parser.parse_args()
 import numpy as np,torch
 main(args)
