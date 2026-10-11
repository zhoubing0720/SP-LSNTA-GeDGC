# -*- coding: utf-8 -*-
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd,mudata as md
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description='Correct only Kidney feature metadata, retaining every matrix value and cell/label')
p.add_argument('--input',default=r'C:\Users\1ce\Desktop\实验\GeDGC - 副本\data\Kidney_4753.h5mu')
p.add_argument('--raw-dir',default=r'D:\GeDGC_datasets\Kidney\mouse_kidney_raw')
p.add_argument('--output',default=str(BASE/'Kidney_4753_annotations_v1.h5mu'))
a=p.parse_args();original=Path(a.input);raw=Path(a.raw_dir);output=Path(a.output)
assert original.resolve()!=output.resolve()
if output.exists():raise FileExistsError('Refusing to overwrite '+str(output))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
input_sha=sha(original);old=md.read_h5mu(str(original));data=old.copy();summary={}
for mod,source,col in [('rna','GSM3271044_RNA_mouse_kidney_gene.txt','gene_id'),('atac','GSM3271045_ATAC_mouse_kidney_peak.txt','id')]:
    mapping=pd.read_csv(BASE/(mod+'_feature_header_offset_mapping.csv'),dtype={'stored_feature_id':str,'actual_source_feature_id':str})
    assert list(data[mod].var_names)==mapping['stored_feature_id'].tolist()
    indices=mapping['zero_based_source_matrix_column'].to_numpy(dtype=np.int64)
    frame=pd.read_csv(raw/source,dtype=str)
    selected=frame.iloc[indices].copy()
    new_ids=selected[col].astype(str).tolist()
    assert new_ids==mapping['actual_source_feature_id'].tolist() and len(set(new_ids))==len(new_ids)
    info=data[mod].var.copy();info['legacy_feature_id']=info.index.astype(str);info['source_feature_index']=indices
    for c in selected.columns:info[c]=selected[c].to_numpy()
    info.index=pd.Index(new_ids,name=info.index.name)
    data[mod].var=info
    summary[mod]={'features':len(new_ids),'changed_ids':int(sum(x!=y for x,y in zip(old[mod].var_names,new_ids))),'first_actual_id':new_ids[0]}
data.update_var()
data.uns['kidney_annotation_correction_v1']={
 'source_accession':'GSE117089','RNA_sample':'GSM3271044','ATAC_sample':'GSM3271045',
 'legacy_file_sha256':input_sha,'scope':'feature metadata only; no matrix, cell-order, or label changes',
 'cause':'legacy read_names included the CSV header; matrix indices refer to data rows only',
 'historical_sampling_seed':0,'historical_label_usage':'RNA cell_name availability filter and proportional cell_type-stratified sampling',
 'feature_selection':'top population variance of raw counts after cell subsampling',
 'historical_constructor_sha256':'238a00381c4b0472fdc116ad818be23f5bb746b892b1fd49bf24c4d6d1927817'}
data.write_h5mu(str(output));new=md.read_h5mu(str(output))
checks={}
for mod in ['rna','atac']:
    x,y=old[mod],new[mod]
    diff=x.X-y.X
    checks[mod]={
      'matrix_shape_same':x.shape==y.shape,
      'matrix_dtype_same':str(x.X.dtype)==str(y.X.dtype),
      'matrix_diff_entries':int(diff.nnz),
      'csr_data_exact':bool(np.array_equal(x.X.data,y.X.data)),
      'csr_indices_exact':bool(np.array_equal(x.X.indices,y.X.indices)),
      'csr_indptr_exact':bool(np.array_equal(x.X.indptr,y.X.indptr)),
      'cell_order_same':x.obs_names.equals(y.obs_names),
      'obs_all_columns_same':x.obs.equals(y.obs),
      'label_category_codes_same':bool(np.array_equal(x.obs['cell_type'].cat.codes,y.obs['cell_type'].cat.codes)),
      'source_feature_ids_correct':list(y.var_names)==list(y.var['gene_id' if mod=='rna' else 'id'].astype(str)),
      'var_metadata_survived_roundtrip':data[mod].var.equals(y.var)}
    assert checks[mod]['matrix_diff_entries']==0
    assert all(v for k,v in checks[mod].items() if k!='matrix_diff_entries')
assert old.obs_names.equals(new.obs_names)
assert sha(original)==input_sha
report={'original_file':str(original),'original_sha256':input_sha,'corrected_file':str(output),'corrected_sha256':sha(output),'original_input_unchanged':True,'global_cell_order_same':True,'feature_summary':summary,'checks':checks,'raw_feature_table_sha256':{n:sha(raw/n) for n in ['GSM3271044_RNA_mouse_kidney_gene.txt','GSM3271045_ATAC_mouse_kidney_peak.txt']}}
(BASE/'注释修正验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
