# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib,mudata as md,numpy as np
BASE=Path(r'C:\Users\1ce\Desktop\实验\_review_round1\审稿问题处理\聊天来源证据恢复\Kidney_注释修正版_v1')
a=md.read_h5mu(str(BASE/'Kidney_4753_annotations_v1.h5mu'))
b=md.read_h5mu(str(BASE/'构建复核'/'Kidney_4753_header_fixed_rebuilt.h5mu'))
checks={}
for mod in ['rna','atac']:
    x,y=a[mod],b[mod];diff=x.X-y.X
    checks[mod]={'matrix_difference_entries':int(diff.nnz),'cell_order_same':x.obs_names.equals(y.obs_names),'all_observation_metadata_same':x.obs.equals(y.obs),'feature_ids_same':x.var_names.equals(y.var_names),'label_codes_same':bool(np.array_equal(x.obs['cell_type'].cat.codes,y.obs['cell_type'].cat.codes))}
    assert diff.nnz==0 and all(v for k,v in checks[mod].items() if k!='matrix_difference_entries')
original=Path(r'C:\Users\1ce\Desktop\实验\GeDGC - 副本\data\Kidney_4753.h5mu')
assert hashlib.sha256(original.read_bytes()).hexdigest()=='c095bef25a489e7a9d7462827225a91a027004c9ab276e3c309f375dafc837c9'
report={'fixed_constructor_rebuild_matches_corrected_dataset':True,'checks':checks,'frozen_experiment_input_unchanged':True,'rebuilt_file_sha256':hashlib.sha256((BASE/'构建复核'/'Kidney_4753_header_fixed_rebuilt.h5mu').read_bytes()).hexdigest()}
(BASE/'原始文件重建验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
