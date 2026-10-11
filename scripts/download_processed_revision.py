from pathlib import Path
import argparse,json,hashlib,urllib.request,os
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description="Download versioned exact processed release files and verify SHA256")
 p.add_argument('--list',action='store_true',help='Inspect planned assets without downloading');p.add_argument('--all',action='store_true');p.add_argument('--dataset',action='append',default=[])
 a=p.parse_args();records=json.loads((ROOT/'data/revision_downloads.json').read_text(encoding='utf-8'))
 selected=records if a.all or not a.dataset else [r for r in records if r['dataset'] in a.dataset]
 if a.dataset and set(a.dataset)-{r['dataset'] for r in selected}:raise ValueError('Unknown dataset')
 if a.list:
  for r in selected:print(r['dataset'],r['target'],r['sha256'],r['url'])
  return
 for r in selected:
  target=ROOT/r['target'];target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():
   if target.stat().st_size==r['bytes'] and sha(target)==r['sha256']:print('VERIFIED existing',r['dataset']);continue
   raise ValueError('Existing file differs; it will not be overwritten: '+str(target))
  temp=target.with_name(target.name+'.download.part')
  with urllib.request.urlopen(r['url'],timeout=120) as response,temp.open('wb') as f:
   while True:
    b=response.read(4194304)
    if not b:break
    f.write(b)
  if temp.stat().st_size!=r['bytes'] or sha(temp)!=r['sha256']:raise ValueError('Downloaded file failed verification: '+str(temp))
  os.replace(temp,target);print('VERIFIED downloaded',r['dataset'])
if __name__=='__main__':main()
