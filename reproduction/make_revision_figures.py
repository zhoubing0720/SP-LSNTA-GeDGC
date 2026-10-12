# -*- coding: utf-8 -*-
"""Revision figures: style inherited from initial submission; all new runs retained."""
from pathlib import Path
import argparse,json,csv,sys,hashlib,os,warnings
os.environ.setdefault('NUMBA_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import transforms
import pandas as pd
import mudata as md
import umap
sys.path.insert(0,str(Path(__file__).resolve().parent/'_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment
parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,default=Path('_review_round1/revision_experiments_20261010'));parser.add_argument('--analysis',type=Path,default=Path('_review_round1/收尾分析_20261011'));parser.add_argument('--umap-only',action='store_true');parser.add_argument('--plots-only',action='store_true');args=parser.parse_args()
BASE=args.batch.resolve();OUT=args.analysis.resolve();FIG=OUT/'figures';FIG.mkdir(exist_ok=True);SRC=OUT/'figure_source';SRC.mkdir(exist_ok=True)
P=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'));D=json.loads((OUT/'plot_data.json').read_text(encoding='utf-8'))
with (OUT/'tables/all_32_groups_mean_SD_CI.csv').open(encoding='utf-8-sig') as f:MEANS=list(csv.DictReader(f))
DISP={'Chen_high':'Chen-2019','Pbmc10k':'Pbmc10k','Kidney_4753':'Kidney','opmultiome':'opmultiome','multiome_high':'multiome','TEA_seq':'TEA-seq','dyngen_branching':'dyngen (13-class simulation)'}
METRICS=['ACC','NMI','ARI','Purity'];STYLE={'A_GeDGC':('#666666','o','', 'GeDGC'),'B_SP_only':('#4682b4','s','///','GeDGC + SP'),'H_LSNTA_only':('#579957','^','\\\\','GeDGC + LSNTA'),'F_full':('#ff0000','*','xxx','SP-LSNTA-GeDGC')}
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':13,'axes.labelsize':14,'axes.titlesize':14,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':12,'axes.linewidth':.85,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
records=[]
def mean(ds,cfg,metric):
 r=next(r for r in MEANS if (r['dataset'],r['config'],r['metric'])==(ds,cfg,metric));return 100*float(r['mean']),100*float(r['sample_SD'])
def save_source(name,rows):
 with (SRC/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def label(ax,letter):
 ax.set_gid(letter);ax.text(-.13,1.035,'('+letter+')',transform=ax.transAxes,ha='left',va='bottom',fontweight='bold',fontsize=14,clip_on=False)
def export(fig,name,claim,spread,panels,exclude_axes=None):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=str(FIG/(name+'.alignment.json')),overlay_svg=str(FIG/(name+'.alignment.svg')),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,exclude_axes=exclude_axes or [])
 # Native dimensions preserve at least the initial image pixel counts; PDF/SVG keep editable text.
 fig.savefig(FIG/(name+'.pdf'),dpi=600)
 fig.savefig(FIG/(name+'.svg'),dpi=600)
 fig.savefig(FIG/(name+'.png'),dpi=600)
 fig.savefig(FIG/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 records.append(dict(figure=name,claim=claim,spread=spread,panels=panels,initial_style='Times New Roman, white background, original grey/red curves or grey/blue/green/red hatched bars, black axes and shared legend',dpi=600,pixels=[int(round(fig.get_figwidth()*600)),int(round(fig.get_figheight()*600))]))
 plt.close(fig)
 print('EXPORTED '+name,flush=True)
if not args.umap_only:
 # Figure 2 inherits the original 2x2 line-chart geometry, grey baseline circles and red full-model stars.
 datasets=['Pbmc10k','Chen_high','Kidney_4753','opmultiome','multiome_high'];fig,axs=plt.subplots(2,2,figsize=(10,7.2));fig.subplots_adjust(left=.095,right=.975,bottom=.135,top=.93,wspace=.26,hspace=.42);source=[]
 for ax,metric,letter in zip(axs.flat,METRICS,'abcd'):
  for cfg in ['A_GeDGC','F_full']:
   color,marker,_,display=STYLE[cfg];ys,errs=zip(*(mean(ds,cfg,metric) for ds in datasets))
   ax.errorbar(np.arange(5),ys,yerr=errs,color=color,marker=marker,markersize=7 if marker=='*' else 5,linewidth=1.8 if cfg=='F_full' else 1.2,elinewidth=.8,capsize=3,label=display,zorder=4 if cfg=='F_full' else 3)
   source += [dict(dataset=ds,metric=metric,model=display,n=5,mean_percent=y,sample_SD_percent=e) for ds,y,e in zip(datasets,ys,errs)]
  ax.set_xticks(range(5),[DISP[ds] for ds in datasets]);ax.set_ylabel(metric+' (%)');ax.set_xlabel('DATA');ax.grid(True,linestyle='--',color='#dddddd',linewidth=.65);ax.set_axisbelow(True);label(ax,letter)
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False)
 export(fig,'Figure2_bimodal_performance','Mean gains depend on dataset and metric; comparison limited to two same-input models.','mean ± sample SD; n=5 full training runs',list('abcd'));save_source('Figure2_bimodal_performance',source)
 # Figure 3 keeps original three-panel grouped bars, four module configurations and hatches; add SD only.
 datasets=['Chen_high','Pbmc10k','opmultiome'];configs=['A_GeDGC','B_SP_only','H_LSNTA_only','F_full'];fig,axs=plt.subplots(1,3,figsize=(8893/600,3172/600));fig.subplots_adjust(left=.06,right=.987,bottom=.23,top=.87,wspace=.28);source=[]
 for ax,ds,letter in zip(axs,datasets,'abc'):
  all_upper=[]
  for j,cfg in enumerate(configs):
   color,marker,hatch,display=STYLE[cfg];color='#e9636b' if cfg=='F_full' else '#aaaaaa' if cfg=='A_GeDGC' else color
   ys,errs=zip(*(mean(ds,cfg,m) for m in METRICS));xs=np.arange(4)+(j-1.5)*.19
   ax.bar(xs,ys,width=.18,color=color,edgecolor='#333333',linewidth=.55,hatch=hatch,label=display,yerr=errs,rasterized=True,error_kw={'elinewidth':.75,'capsize':2,'capthick':.7})
   for x,y,e in zip(xs,ys,errs):ax.text(x,y+e+.5,'%.1f'%y,ha='left',va='center',rotation=90,rotation_mode='anchor',fontsize=11)
   source += [dict(dataset=ds,metric=m,config=cfg,n=5,mean_percent=y,sample_SD_percent=e) for m,y,e in zip(METRICS,ys,errs)];all_upper.extend(np.array(ys)+np.array(errs))
  lower=min(r['mean_percent']-r['sample_SD_percent'] for r in source if r['dataset']==ds)-4
  ax.set_ylim(max(0,5*np.floor(lower/5)),5*np.ceil((max(all_upper)+8)/5));ax.set_xticks(range(4),METRICS);ax.set_ylabel('Clustering performance (%)');ax.set_title('('+letter+') '+DISP[ds],fontweight='bold',pad=12);ax.set_gid(letter);ax.grid(axis='y',color='#dddddd',linestyle='--',linewidth=.55);ax.set_yticks(np.arange(10*np.ceil(max(0,5*np.floor(lower/5))/10),5*np.ceil((max(all_upper)+8)/5)+.1,10));ax.set_axisbelow(True)
  # Place each metric's numerical labels above its entire SD envelope and clear grid strokes.
  span=(ax.get_ylim()[1]-ax.get_ylim()[0])*25/(fig.get_figheight()*72*.64)
  for index,text in enumerate(ax.texts):
   metric=METRICS[index % 4];upper=max(sum(mean(ds,cfg,metric)) for cfg in configs);y=upper+1.0
   for tick in ax.get_yticks():
    if y-.25<=tick<=y+span+.25:y=tick+.75
   text.set_position((text.get_position()[0],y))
 legend=fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.045),ncol=4,frameon=False,fontsize=13)
 for patch in legend.get_patches():patch.set_rasterized(True)
 export(fig,'Figure3_module_ablation','SP and LSNTA have condition-dependent effects; four configurations are compared without asserting synergy.','mean ± sample SD; n=5',list('abc'));save_source('Figure3_module_ablation',source)
 # Focused component effects complement the module bars; original font and colours retained.
 names=['Latent_graph_without_alignment','Latent_graph_given_alignment','FG_MNN_given_UOT','UOT_given_FG_MNN'];labels=['Shared graph: C - B','Shared graph: F - G','FG-MNN: F - E','UOT: F - D'];fig,axs=plt.subplots(1,3,figsize=(14.82,5.8));fig.subplots_adjust(left=.17,right=.98,bottom=.19,top=.86,wspace=.35);source=[]
 for ax,ds,letter in zip(axs,datasets,'abc'):
  for i,(key,display) in enumerate(zip(names,labels)):
   r=next(r for r in D['ablation'] if r['dataset']==ds and r['contrast']==key and r['metric']=='ACC');y=3-i;color='#d62728' if r['difference_pp']>=0 else '#666666'
   ax.errorbar(r['difference_pp'],y,xerr=[[r['difference_pp']-r['CI95_low_pp']],[r['CI95_high_pp']-r['difference_pp']]],fmt='o',color=color,capsize=3,elinewidth=1,markersize=5);source.append(r)
  ax.axvline(0,color='#888888',linewidth=.8,linestyle='--');ax.set_yticks(range(4));ax.set_yticklabels(labels[::-1] if ds=='Chen_high' else []);ax.set_xlabel('ACC difference (score × 100)');ax.set_title('('+letter+') '+DISP[ds],fontweight='bold',pad=12);ax.set_gid(letter);ax.set_ylim(-.5,3.5);ax.grid(axis='x',color='#dddddd',linestyle='--',linewidth=.6)
 export(fig,'FigureS1_component_effects','Conditional component contributions are uncertain and dataset dependent.','paired mean difference ± 95% t CI; n=5; df=4',list('abc'));save_source('FigureS1_component_effects',source)
 # Separate total runtime and allocated memory endpoints rather than claiming complexity reduction.
 datasets=list(P['datasets']);fig,axs=plt.subplots(1,2,figsize=(12,5.6));fig.subplots_adjust(left=.08,right=.975,bottom=.32,top=.87,wspace=.28);source=[]
 for ax,field,title,letter in zip(axs,['TIME_full_seconds','peak_gpu_allocated_MiB'],['Full-pipeline runtime (s)','Peak allocated GPU memory (MiB)'],'ab'):
  for j,cfg in enumerate(['A_GeDGC','F_full']):
   rows=[next(r for r in D['performance'] if r['dataset']==ds and r['config']==cfg) for ds in datasets];ys=[r[field+'_mean'] for r in rows];sd=[r[field+'_SD'] for r in rows];color='#aaaaaa' if j==0 else '#e9636b';hatch='' if j==0 else 'xxx'
   ax.bar(np.arange(7)+(j-.5)*.33,ys,width=.3,color=color,hatch=hatch,edgecolor='#333333',linewidth=.6,yerr=sd,error_kw={'elinewidth':.75,'capsize':2},label=STYLE[cfg][3],rasterized=True)
   source += [dict(dataset=ds,config=cfg,measure=field,n=5,mean=y,sample_SD=e) for ds,y,e in zip(datasets,ys,sd)]
  ax.set_xticks(range(7),[DISP[d].replace(' (13-class simulation)','') for d in datasets],rotation=60,ha='right',rotation_mode='anchor');ax.set_ylabel(title);ax.set_title('('+letter+') '+title,fontweight='bold',pad=12);ax.set_gid(letter);ax.grid(axis='y',color='#dddddd',linestyle='--',linewidth=.6);ax.set_axisbelow(True)
 legend=fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False)
 for patch in legend.get_patches():patch.set_rasterized(True)
 export(fig,'FigureS2_runtime_memory','Actual allocated GPU peaks are lower; runtime varies in direction, without removing quadratic operations.','mean ± sample SD; n=5',list('ab'));save_source('FigureS2_runtime_memory',source)
 # Requested interaction has paired effect and interval; no manufactured synergistic claim.
 fig,ax=plt.subplots(figsize=(7.2,4.6));fig.subplots_adjust(left=.12,right=.96,bottom=.25,top=.85);source=[]
 for i,m in enumerate(['ACC','ARI']):
  r=next(r for r in D['ablation'] if (r['dataset'],r['contrast'],r['metric'])==('Pbmc10k','SP_LSNTA_interaction',m));ax.errorbar(i,r['difference_pp'],yerr=[[r['difference_pp']-r['CI95_low_pp']],[r['CI95_high_pp']-r['difference_pp']]],fmt='o',color='#e9636b',capsize=5,markersize=6);source.append(r)
 ax.axhline(0,color='#666666',linestyle='--',linewidth=.9);ax.set_xticks([0,1],['ACC','ARI']);ax.set_ylabel('Interaction F − B − H + A (score × 100)');ax.set_title('Pbmc10k: SP × LSNTA interaction',fontweight='bold');ax.grid(axis='y',color='#dddddd',linestyle='--');ax.set_xlim(-.6,1.6)
 export(fig,'FigureS3_Pbmc_interaction','The measured interaction is close to zero and uncertain.','paired mean contrast ± 95% t CI; n=5', ['single']);save_source('FigureS3_Pbmc_interaction',source)
# Per-dataset two-model UMAPs: use all cells and fixed preselected seed 0; cache only derived coordinates.
UMAP_RECORDS=json.loads((OUT/'UMAP_provenance.json').read_text(encoding='utf-8')) if args.plots_only else []
if args.plots_only:records += [r for r in json.loads((OUT/'figure_contracts.json').read_text(encoding='utf-8')) if r['figure'].startswith('UMAP_') or 'Figure6_' in r['figure']]
for ds,dspec in ({} if args.plots_only else P['datasets']).items():
 palette=list(plt.get_cmap('tab20').colors)+list(plt.get_cmap('tab20b').colors)
 data=md.read_h5mu(dspec['path'],backed='r')
 series=data.mod['rna'].obs['cell_type'].astype('category');names=[str(v) for v in series.cat.categories];yc=series.cat.codes.to_numpy();data.file.close()
 loaded=[]
 for cfg in ['A_GeDGC','F_full']:
  run=BASE/'runs'/ds/cfg/'seed_0';path=run/'result/experiment'/('Trajectory_latent_GeDGC.npz' if cfg=='A_GeDGC' else 'Trajectory_latent_SP_LSNTA_GeDGC.npz')
  with np.load(path,allow_pickle=False) as z:latent=z['latent'];y=z['y_true']
  assert np.array_equal(y,yc) and latent.shape==(dspec['cells'],dspec['classes'])
  coord=SRC/(ds+'_'+cfg+'_seed0_UMAP.npz');latent_sha=hashlib.sha256(path.read_bytes()).hexdigest()
  if coord.exists():
   with np.load(coord,allow_pickle=False) as z:assert str(z['latent_sha256'])==latent_sha;xy=z['xy']
  else:
   print('UMAP START '+ds+' '+cfg+' all '+str(len(y))+' cells',flush=True)
   xy=umap.UMAP(n_components=2,n_neighbors=30,min_dist=.3,metric='euclidean',random_state=42,n_jobs=1).fit_transform(latent)
   np.savez_compressed(coord,xy=xy,y_true=y,latent_sha256=np.array(latent_sha))
  assert xy.shape==(len(y),2) and np.isfinite(xy).all();loaded.append((cfg,xy,y));UMAP_RECORDS.append(dict(dataset=ds,config=cfg,model_seed=0,umap_seed=42,n_neighbors=30,min_dist=.3,metric='euclidean',all_cells=len(y),latent_sha256=latent_sha,coordinates_sha256=hashlib.sha256(coord.read_bytes()).hexdigest(),fit='separate model posterior-mean fits',model_metrics_use='sampled latent classifier predictions, not this UMAP'))
 fig,axs=plt.subplots(1,2,figsize=(10,5.6));fig.subplots_adjust(left=.04,right=.97,bottom=.31,top=.86,wspace=.08)
 for ax,(cfg,xy,y),letter in zip(axs,loaded,'ab'):
  ax.scatter(xy[:,0],xy[:,1],c=np.asarray([palette[int(v)] for v in y]),s=1.1,alpha=.85,linewidths=0,edgecolors='none',rasterized=True)
  ax.set_xticks([]);ax.set_yticks([]);ax.set_title('('+letter+') '+STYLE[cfg][3],fontsize=14,fontweight='bold',pad=12);ax.set_gid(letter)
  for spine in ax.spines.values():spine.set_linewidth(.65);spine.set_visible(True)
 handles=[Line2D([0],[0],marker='o',linestyle='none',markersize=4,markerfacecolor=palette[i],markeredgecolor='none',label=name) for i,name in enumerate(names)]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.022),ncol=min(5,len(names)),fontsize=9.2,frameon=False,columnspacing=1.05,handletextpad=.25,labelspacing=.28)
 fig.suptitle(DISP[ds],fontsize=15,fontweight='bold',y=.96)
 name='Figure6_UMAP_Chen2019' if ds=='Chen_high' else 'UMAP_'+ds
 export(fig,name,'Representative same-input posterior-mean geometry; no direct performance or cross-panel coordinate inference.','single preselected seed 0, all cells; no uncertainty claim',['a','b'])
 (SRC/(ds+'_annotation_code_names.json')).write_text(json.dumps(dict(enumerate(names)),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'figure_contracts.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'UMAP_provenance.json').write_text(json.dumps(UMAP_RECORDS,ensure_ascii=False,indent=2),encoding='utf-8')
print('ALL FIGURES EXPORTED',flush=True)
