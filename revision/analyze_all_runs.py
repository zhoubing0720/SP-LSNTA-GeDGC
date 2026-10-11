# -*- coding: utf-8 -*-
from pathlib import Path
import argparse,json,csv,hashlib,math,itertools,sys,collections,statistics
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score,fowlkes_mallows_score

parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,default=Path('_review_round1/revision_experiments_20261010'));parser.add_argument('--out',type=Path,default=Path('_review_round1/收尾分析_20261011'));parser.add_argument('--data-dir',type=Path);parser.add_argument('--archive-mode',action='store_true');args=parser.parse_args()
BASE=args.batch.resolve();OUT=args.out.resolve();TAB=OUT/'tables';TAB.mkdir(parents=True,exist_ok=True)
P=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'));METS=['ACC','NMI','ARI','Purity','FMI'];CORE=METS[:4];SEEDS=P['seeds'];cache={}
def sha(p):
 p=Path(p).resolve()
 if p not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4194304),b''):h.update(b)
  cache[p]=h.hexdigest()
 return cache[p]
def savejson(name,v):
 (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvsave(name,rows):
 assert rows,name
 with (TAB/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def recompute(y,p):
 assert np.issubdtype(y.dtype,np.integer) and np.issubdtype(p.dtype,np.integer)
 k=int(max(y.max(),p.max()))+1;w=np.bincount(p.astype(np.int64)*k+y.astype(np.int64),minlength=k*k).reshape(k,k)
 i,j=linear_sum_assignment(w.max()-w)
 return {'ACC':float(w[i,j].sum()/len(y)),'NMI':float(normalized_mutual_info_score(y,p)),'ARI':float(adjusted_rand_score(y,p)),'Purity':float(w.max(axis=1).sum()/len(y)),'FMI':float(fowlkes_mallows_score(y,p))}
def effect(d):
 d=np.asarray(d,float);n=len(d);mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/math.sqrt(n);half=float(stats.t.ppf(.975,n-1)*se)
 if se>0:t=mean/se;p2=float(2*stats.t.sf(abs(t),n-1));p1=float(stats.t.sf(t,n-1));dz=mean/sd
 else:t=None;p2=1. if mean==0 else 0.;p1=1. if mean==0 else 0. if mean>0 else 1.;dz=None
 signed=np.array([np.mean(d*np.array(s)) for s in itertools.product([-1,1],repeat=n)])
 exact=float(np.mean(np.abs(signed)>=abs(mean)-1e-12))
 return dict(n=n,difference_pp=mean,difference_SD_pp=sd,CI95_low_pp=mean-half,CI95_high_pp=mean+half,t=t,df=n-1,p_greater_inherited_protocol=p1,p_two_sided=p2,exact_signflip_two_sided=exact,paired_dz=dz,positive_seed_differences=int((d>1e-12).sum()),negative_seed_differences=int((d<-1e-12).sum()))
def holm(rows,key,out,eligible=lambda r:True):
 selected=[r for r in rows if eligible(r)];ordered=sorted(selected,key=lambda r:r[key]);current=0.
 for rank,r in enumerate(ordered):current=max(current,(len(ordered)-rank)*r[key]);r[out]=min(1.,current)
 for r in rows:
  if out not in r:r[out]=None
 return len(selected)
protocol_sha=P.get('original_protocol_sha256',sha(BASE/'protocol.json')) if args.archive_mode else sha(BASE/'protocol.json')
if args.data_dir:
 for ds in P['datasets'].values():ds['path']=str(args.data_dir.resolve()/Path(ds['path']).name)
for rel,inf in P['sources'].items():assert sha(BASE/rel)==inf.get('snapshot_sha256_after_seed_parameterization',inf['sha256']),rel
for rel,h in P['execution_sha256'].items():assert sha(BASE/rel)==h,rel
for dataset,ds in P['datasets'].items():assert sha(ds['path'])==ds['sha256'],dataset
RUNS={};audit=[];perseed=[];reference_labels={};stages=[]
for task in P['tasks']:
 ds,cfg,seed=task['dataset'],task['config'],task['seed'];run=BASE/'runs'/ds/cfg/('seed_'+str(seed));m=json.loads((run/'metrics.json').read_text(encoding='utf-8'));info=json.loads((run/'run_info.json').read_text(encoding='utf-8'))
 assert (m['Dataset'],m['config'],m['seed'])==(ds,cfg,seed)
 assert m['fingerprints']['protocol_sha256']==protocol_sha and m['fingerprints']['data_sha256']==P['datasets'][ds]['sha256']
 assert m['fingerprints']['execution_sha256']==P['execution_sha256']
 assert info['flags']==P['configs'][cfg] and info['fresh_pretraining'] and m['fresh_initialization'] and m['pretraining_performed']
 pre=json.loads((run/'pretraining_completion.json').read_text(encoding='utf-8'));assert pre['complete'] and pre['fresh'] and pre['seed']==seed
 if not args.archive_mode:
  for filename,h in pre['weights_sha256'].items():assert sha(run/'Intermediate_data/experiment'/filename)==h
 assert json.loads((run/'progress.json').read_text(encoding='utf-8'))['stage']=='complete'
 with np.load(run/'predictions.npz',allow_pickle=False) as z:y=z['y_true'];pred=z['y_pred']
 assert y.ndim==pred.ndim==1 and len(y)==len(pred)==P['datasets'][ds]['cells']
 assert len(np.unique(y))==P['datasets'][ds]['classes']
 if ds in reference_labels:assert np.array_equal(y,reference_labels[ds])
 else:reference_labels[ds]=y.copy()
 computed=recompute(y,pred);maxerr=max(abs(m[k]-computed[k]) for k in METS);assert maxerr<1e-12,(ds,cfg,seed,maxerr)
 latent_file=run/'result/experiment'/('Trajectory_latent_GeDGC.npz' if cfg=='A_GeDGC' else 'Trajectory_latent_SP_LSNTA_GeDGC.npz')
 with np.load(latent_file,allow_pickle=False) as z:assert np.array_equal(z['y_true'],y);assert z['latent'].shape==(len(y),P['datasets'][ds]['classes']);assert np.isfinite(z['latent']).all()
 assert m['peak_gpu_reserved_MiB']>=m['peak_gpu_allocated_MiB']>0 and m['TIME_full_seconds']>0
 assert all(math.isfinite(m[k]) for k in METS+['TIME_full_seconds','peak_gpu_allocated_MiB','peak_gpu_reserved_MiB'])
 hist=json.loads((run/'loss_history.json').read_text(encoding='utf-8'));assert len(hist['loss'])==m['joint_epochs']==hist['joint_epochs']
 RUNS[(ds,cfg,seed)]=m
 perseed.append(dict(dataset=ds,config=cfg,seed=seed,role=task['role'],**{k:m[k] for k in METS},TIME_full_seconds=m['TIME_full_seconds'],peak_gpu_allocated_MiB=m['peak_gpu_allocated_MiB'],peak_gpu_reserved_MiB=m['peak_gpu_reserved_MiB'],joint_epochs=m['joint_epochs']))
 audit.append(dict(dataset=ds,config=cfg,seed=seed,metrics_sha256=sha(run/'metrics.json'),predictions_sha256=sha(run/'predictions.npz'),latent_sha256=sha(latent_file),data_sha256=P['datasets'][ds]['sha256'],checkpoint_hashes_checked=0 if args.archive_mode else len(pre['weights_sha256']),max_recomputed_metric_difference=maxerr,labels_and_latent_order_validated=True))
 for stage in m['timing']:stages.append(dict(dataset=ds,config=cfg,seed=seed,stage=stage['stage'],seconds=stage['seconds'],nested_timer=stage['stage'] in ['VAE_pretrain','classifier_pretrain','FG_MNN','UOT','joint_optimization']))
 assert len(RUNS)==len(perseed)
print('Validated all 160 predictions and frozen inputs/source. Explicit archive mode omits checkpoint rehash.',flush=True)
csvsave('per_seed_all_160.csv',perseed);csvsave('independent_numeric_audit_160.csv',audit);csvsave('stage_timings_all.csv',stages)
groups=sorted(set((ds,cfg) for ds,cfg,s in RUNS));means=[]
for ds,cfg in groups:
 vals=[RUNS[(ds,cfg,s)] for s in SEEDS]
 for metric in METS:
  x=np.array([v[metric] for v in vals]);half=stats.t.ppf(.975,4)*x.std(ddof=1)/np.sqrt(5)
  means.append(dict(dataset=ds,config=cfg,metric=metric,n=5,mean=float(x.mean()),sample_SD=float(x.std(ddof=1)),mean_CI95_low=float(x.mean()-half),mean_CI95_high=float(x.mean()+half)))
csvsave('all_32_groups_mean_SD_CI.csv',means)
main=[]
for ds in P['datasets']:
 for metric in METS:
  a=np.array([RUNS[(ds,'A_GeDGC',s)][metric] for s in SEEDS]);b=np.array([RUNS[(ds,'F_full',s)][metric] for s in SEEDS])
  main.append(dict(dataset=ds,metric=metric,GeDGC_mean=float(a.mean()),GeDGC_SD=float(a.std(ddof=1)),full_mean=float(b.mean()),full_SD=float(b.std(ddof=1)),**effect(100*(b-a))))
assert holm(main,'p_two_sided','p_Holm28_core',lambda r:r['metric'] in CORE)==28
assert holm(main,'p_two_sided','p_Holm35_with_FMI')==35
csvsave('main_7datasets_paired_effects.csv',main)
CONTRASTS=[('SP_only','B_SP_only','A_GeDGC'),('LSNTA_only','H_LSNTA_only','A_GeDGC'),('Full_vs_GeDGC','F_full','A_GeDGC'),('LSNTA_given_SP','F_full','B_SP_only'),('SP_given_LSNTA','F_full','H_LSNTA_only'),('Latent_graph_without_alignment','C_SP_latent_graph','B_SP_only'),('Latent_graph_given_alignment','F_full','G_full_raw_graph'),('FG_MNN_given_UOT','F_full','E_without_FG_MNN'),('UOT_given_FG_MNN','F_full','D_without_UOT'),('SP_LSNTA_interaction','F_full',None)]
ablation=[];seedcontrast=[]
for ds in ['Pbmc10k','Chen_high','opmultiome']:
 for name,pos,neg in CONTRASTS:
  for metric in METS:
   diffs=[]
   for s in SEEDS:
    d=RUNS[(ds,pos,s)][metric]-RUNS[(ds,neg,s)][metric] if neg else RUNS[(ds,'F_full',s)][metric]-RUNS[(ds,'B_SP_only',s)][metric]-RUNS[(ds,'H_LSNTA_only',s)][metric]+RUNS[(ds,'A_GeDGC',s)][metric]
    diffs.append(d*100);seedcontrast.append(dict(dataset=ds,contrast=name,metric=metric,seed=s,difference_pp=d*100))
   ablation.append(dict(dataset=ds,contrast=name,metric=metric,formula=pos+'-'+str(neg) if neg else 'F-B-H+A',**effect(diffs)))
assert holm(ablation,'p_two_sided','p_Holm120_core',lambda r:r['metric'] in CORE)==120
assert holm(ablation,'p_two_sided','p_Holm150_with_FMI')==150
holm(ablation,'p_two_sided','p_Holm2_requested_Pbmc_interaction',lambda r:r['dataset']=='Pbmc10k' and r['contrast']=='SP_LSNTA_interaction' and r['metric'] in ['ACC','ARI'])
csvsave('ablation_150_contrasts.csv',ablation);csvsave('ablation_contrasts_per_seed.csv',seedcontrast)
performance=[];perfeffects=[]
for ds,cfg in groups:
 vals=[RUNS[(ds,cfg,s)] for s in SEEDS]
 row=dict(dataset=ds,config=cfg,n=5,GPU=vals[0]['GPU'],torch=vals[0]['torch'],CUDA=vals[0]['CUDA'])
 for field in ['TIME_full_seconds','TIME_without_loading_seconds','peak_gpu_allocated_MiB','peak_gpu_reserved_MiB','joint_epochs']:
  x=np.array([v[field] for v in vals]);row[field+'_mean']=float(x.mean());row[field+'_SD']=float(x.std(ddof=1));row[field+'_min']=float(x.min());row[field+'_max']=float(x.max())
 performance.append(row)
for ds in P['datasets']:
 for field in ['TIME_full_seconds','peak_gpu_allocated_MiB','peak_gpu_reserved_MiB']:
  a=np.array([RUNS[(ds,'A_GeDGC',s)][field] for s in SEEDS]);b=np.array([RUNS[(ds,'F_full',s)][field] for s in SEEDS]);ratios=b/a
  perfeffects.append(dict(dataset=ds,measure=field,GeDGC_mean=float(a.mean()),full_mean=float(b.mean()),ratio_of_means=float(b.mean()/a.mean()),percent_change_of_means=float(100*(b.mean()/a.mean()-1)),paired_ratio_mean=float(ratios.mean()),paired_ratio_SD=float(ratios.std(ddof=1))))
csvsave('performance_all_groups.csv',performance);csvsave('performance_main_ratios.csv',perfeffects)
# Main-text-ready tables: four core metrics. FMI supplied separately rather than changing the frozen original endpoint set.
for groupname,datasets in [('Table2_bimodal',['Chen_high','Pbmc10k','Kidney_4753','opmultiome','multiome_high']),('Table3_nmodal',['TEA_seq','dyngen_branching'])]:
 out=[]
 for ds in datasets:
  for cfg in ['A_GeDGC','F_full']:
   row={'dataset':ds,'model':'GeDGC' if cfg=='A_GeDGC' else 'SP-LSNTA-GeDGC','n':5}
   for metric in CORE:
    m=next(x for x in means if (x['dataset'],x['config'],x['metric'])==(ds,cfg,metric));row[metric+'_mean_SD']='%.4f ± %.4f'%(m['mean'],m['sample_SD'])
   out.append(row)
 csvsave(groupname+'_mean_SD.csv',out)
abo=[]
for ds in ['Chen_high','Pbmc10k','opmultiome']:
 for cfg in P['configs']:
  row=dict(dataset=ds,config=cfg,n=5)
  for metric in CORE:
   m=next(x for x in means if (x['dataset'],x['config'],x['metric'])==(ds,cfg,metric));row[metric+'_mean_SD']='%.4f ± %.4f'%(m['mean'],m['sample_SD'])
  abo.append(row)
csvsave('Table4_all_eight_ablation_configs_mean_SD.csv',abo)
# Report frozen choices and all adverse outcomes, no invented literature SD.
report=['# 新批次统计、消融与性能验收','', '2026-10-11。此报告只处理新160次实验，原稿和旧50次记录未修改。','', '## 验收','', '- 160份预测独立重算ACC/NMI/ARI/Purity/FMI，全部与保存值一致（容差1e-12）；输入、源码及执行协议通过；完整本地审计另检查预训练权重，归档模式不重查未发布的权重。','- 32组各5次完整独立运行，细胞标签和潜变量顺序一致；没有排除seed。','- n=5表示固定数据上的训练随机性，不是5名供体，CI不代表跨供体或跨研究生物学不确定性。','', '## 主比较（差值×100；95%配对t区间）','', '|数据集|指标|差值|95%CI|双侧Holm28 p（核心指标）|精确符号翻转 p|','|---|---|---:|---|---:|---:|']
for r in main:
 if r['metric'] in CORE:report.append('|%s|%s|%+.3f|[%.3f, %.3f]|%.5g|%.5g|'%(r['dataset'],r['metric'],r['difference_pp'],r['CI95_low_pp'],r['CI95_high_pp'],r['p_Holm28_core'],r['exact_signflip_two_sided']))
report+=['','t检验和区间依赖同seed差值的相应分布假设。n=5的双侧精确符号翻转最低p=0.0625，因此不能据此获得p<0.05；参数检验结果必须与小样本限制一并解释。四项核心指标统一Holm28；FMI另列，Holm35是含FMI的敏感性分析。原协议单侧结果保留，不取代双侧保守汇总。','', '## 组件结论需依据具体条件','', '|数据集|对比|ACC差值|95%CI|双侧Holm120 p|','|---|---|---:|---|---:|']
for r in ablation:
 if r['metric']=='ACC':report.append('|%s|%s|%+.3f|[%.3f, %.3f]|%.5g|'%(r['dataset'],r['contrast'],r['difference_pp'],r['CI95_low_pp'],r['CI95_high_pp'],r['p_Holm120_core']))
report+=['','各组件差值是其他模块存在/不存在条件下的贡献，不等于孤立因果机制证明。交互按F-B-H+A计算，不用“一个显著另一个不显著”代替交互检验。','', '## 性能范围','', P['runtime_scope'], '', '- GPU为RTX4080，torch2.1.2/CUDA11.8；CUDA同步计时，各run新初始化，不复用别的run预训练。','- peak allocated/reserved为PyTorch进程记录，不含全部驱动/其它进程显存；不报告CPU内存或GPU总容量作为峰值。','- 嵌套的VAE/classifier、FG-MNN/UOT、joint timer包含在父阶段中，不能把全部阶段直接相加。','- 完整模型与基线使用各自冻结的latent/raw SpectralNet输入，显存下降反映本轮完整实现，不能推断所有模态/规模更快或去除了平方复杂度。','- FMI提供补充结果；原四指标为主表。文献七基线保留来源条件，不能为其补造SD或与本研究seed做配对检验。']
(OUT/'统计与性能完整报告.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
savejson('analysis_validation.json',{'records':len(audit),'groups':len(groups),'all_5_metrics_independently_recomputed':True,'max_metric_difference':max(r['max_recomputed_metric_difference'] for r in audit),'all_protocol_data_source_and_pretrain_hashes_validated':not args.archive_mode,'archive_mode':args.archive_mode,'checkpoint_files_present_and_rehashed':not args.archive_mode,'files_hashed':len(cache),'n_core_main_contrasts':28,'n_all_main_contrasts':35,'n_core_ablation_contrasts':120,'n_all_ablation_contrasts':150,'seed_list':SEEDS,'python':sys.version,'numpy':np.__version__,'scipy':__import__('scipy').__version__,'sklearn':__import__('sklearn').__version__,'batch_protocol_sha256':protocol_sha,'exclusions':[]})
savejson('plot_data.json',{'main':main,'ablation':ablation,'performance':performance,'performance_effects':perfeffects})
print(json.dumps({'validation':'passed','main_core_Holm_significant':[(r['dataset'],r['metric'],round(r['difference_pp'],3),r['p_Holm28_core']) for r in main if r['metric'] in CORE and r['p_Holm28_core']<.05],'Pbmc_interaction':[{k:r[k] for k in ['metric','difference_pp','CI95_low_pp','CI95_high_pp','p_Holm2_requested_Pbmc_interaction','p_Holm120_core']} for r in ablation if r['dataset']=='Pbmc10k' and r['contrast']=='SP_LSNTA_interaction' and r['metric'] in ['ACC','ARI']]},ensure_ascii=False,indent=2))
