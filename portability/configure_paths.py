"""Rebase only packaged path literals; never operate on the live source checkout."""
from pathlib import Path
import json,sys,re
ROOT=Path(__file__).resolve().parents[1]
state=ROOT/'portability/configured_paths.json'
old=json.loads(state.read_text()) if state.exists() else {'root':'/hy-tmp/fed','fed_python':'/usr/local/miniconda3/envs/fed/bin/python'}
fed=str(ROOT/'.venv-fed/bin/python');changed=[];seen_inodes=set()
def relocate_dataset_paths(value):
 if isinstance(value,list):return [relocate_dataset_paths(v) for v in value]
 if isinstance(value,dict):return {k:relocate_dataset_paths(v) for k,v in value.items()}
 if isinstance(value,str) and Path(value).is_absolute():
  for name in ['data_hf','data']:
   marker='/'+name+'/'
   if marker in value:
    candidate=ROOT/name/value.split(marker,1)[1]
    if candidate.exists():return str(candidate)
 return value
for base in ['attacks','constants','dataloader','defenses','experiment','features','methods','model','scripts','tools','migration_launchers','reference_splits']:
 for p in (ROOT/base).rglob('*'):
  if not p.is_file() or p.is_symlink() or p.suffix not in {'.py','.sh','.cfg','.json','.md','.yml','.yaml','.txt'}:continue
  st=p.stat();inode=(st.st_dev,st.st_ino)
  if inode in seen_inodes:continue
  seen_inodes.add(inode)
  s=p.read_text(encoding='utf-8')
  t=re.sub(re.escape(old['root'])+r'''(?=/|['"\s]|$)''',lambda m:str(ROOT),s).replace(old['fed_python'],fed)
  if base=='reference_splits' and p.suffix=='.json':
   value=json.loads(t);updated=relocate_dataset_paths(value)
   if updated!=value:t=json.dumps(updated,ensure_ascii=False,indent=2)
  if t!=s:p.write_text(t,encoding='utf-8');changed.append(p.relative_to(ROOT).as_posix())
for n in ['data','output','result','logs']:(ROOT/n).mkdir(exist_ok=True)
state.write_text(json.dumps({'root':str(ROOT),'fed_python':fed,'configured_files':changed},indent=2))
print('CONFIGURED_PACKAGE_ROOT',ROOT,'PATH_FILES_UPDATED',len(changed))
