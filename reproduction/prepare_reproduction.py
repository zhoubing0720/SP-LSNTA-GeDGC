from pathlib import Path
import argparse,json,shutil,hashlib,sys
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description="Prepare a new independent full-pipeline benchmark without overwriting existing runs")
p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve()
if out.exists():raise FileExistsError("Choose a new empty output path: "+str(out))
config=json.loads((HERE/'protocol_template.json').read_text(encoding='utf-8'))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
for ds,info in config['datasets'].items():
 path=a.data_dir.resolve()/Path(info['path']).name
 if not path.is_file() or sha(path)!=info['sha256']:raise ValueError("Missing or different exact input: "+str(path))
 info['path']=str(path)
config['python']=sys.executable;config['workspace']=str(out);config['portable_reproduction_of']='reviewer_revision_full_pipeline_perf_v1'
out.mkdir(parents=True);shutil.copytree(HERE/'frozen_source',out/'snapshot')
for name in config['execution_sha256']:
 shutil.copy2(HERE/'runner'/name,out/name)
 config['execution_sha256'][name]=sha(out/name)
for rel,info in config['sources'].items():
 expected=info.get('snapshot_sha256_after_seed_parameterization',info['sha256'])
 if sha(out/rel)!=expected:raise ValueError("Frozen source mismatch: "+rel)
(out/'protocol.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
print("Prepared 160 new, independent tasks. No training has started.")
print("Run with the same Python environment: "+str(out/'run_queue.py'))
