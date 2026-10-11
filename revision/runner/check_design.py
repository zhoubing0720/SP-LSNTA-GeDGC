# -*- coding: utf-8 -*-
"""Preflight controls and center-only structural-update invariants, synthetic CPU data."""
from pathlib import Path
import importlib.util,sys,json,argparse
import numpy as np,torch
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('revision_worker',str(BASE/'run_worker.py'));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);w.np=np;w.torch=torch
P=w.P
pairs=[('C_SP_latent_graph','B_SP_only','graph'),('F_full','G_full_raw_graph','graph'),('F_full','E_without_FG_MNN','fg_mnn'),('F_full','D_without_UOT','uot'),('F_full','H_LSNTA_only','sp'),('B_SP_only','A_GeDGC','sp')]
for a,b,key in pairs:
 assert {k for k in P['configs'][a] if P['configs'][a][k]!=P['configs'][b][k]}=={key}
assert len(P['tasks'])==160 and sum(t['role']=='main_comparison' for t in P['tasks'])==70
assert all(t['config'] in ['A_GeDGC','F_full'] for t in P['tasks'] if P['datasets'][t['dataset']]['nmodal'])
torch.set_num_threads(2)
checks=[]
for ds,cfg in [('Pbmc10k','H_LSNTA_only'),('dyngen_branching','F_full')]:
 w.main(argparse.Namespace(dataset=ds,config=cfg,seed=1,validate_interfaces=True))
 train=sys.modules['train'];cls=sys.modules['module.Multi_vae'].Multiview_VAE
 dims=[5,7,9,11] if ds=='dyngen_branching' else [5,7]
 torch.manual_seed(12);v=cls([[d,256,3] for d in dims]).eval()
 data=[torch.randn(20,d,4) for d in dims];original=[x.clone() for x in data]
 updated=train.aggregate_feature_gated_mnn(data,vae=v,device=torch.device('cpu'))
 assert any(not torch.equal(a[:,:,0],b[:,:,0]) for a,b in zip(original,updated))
 for a,b in zip(original,updated):assert torch.equal(a[:,:,1:],b[:,:,1:]) and torch.isfinite(b).all()
 after_fg=[x.clone() for x in updated]
 updated=train.aggregate_uot_no_gate(updated,vae=v,seed=1,device=torch.device('cpu'))
 for a,b in zip(after_fg,updated):assert torch.equal(a[:,:,1:],b[:,:,1:]) and torch.isfinite(b).all()
 checks.append({'dataset':ds,'config':cfg,'no_SP_model_supported_by_FG':cfg=='H_LSNTA_only','FG_and_UOT_leave_neighbor_slots_unchanged':True,'finite_center_updates':True})
# Seed parameterization does not alter joint loss or structural algorithms.
import ast
src=Path(P['workspace'])/'GeDGC_NModal_SP_LSNTA/train.py';snap=BASE/'snapshot/nmodal/train.py'
a=ast.parse(src.read_text(encoding='utf-8-sig'));b=ast.parse(snap.read_text(encoding='utf-8-sig'))
for name in ['train_model','aggregate_feature_gated_mnn','aggregate_uot_no_gate','_compute_feature_gate']:
 aa=next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name==name);bb=next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name==name)
 assert ast.dump(aa,include_attributes=False)==ast.dump(bb,include_attributes=False)
result={'status':'passed','synthetic_inputs_only':True,'formal_metrics_created':False,'component_contrasts_differ_in_one_flag':True,'main_tasks':70,'ablation_tasks':90,'mathematical_functions_unchanged':True,'structural_checks':checks}
(BASE/'preflight_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('PREFLIGHT PASSED')
