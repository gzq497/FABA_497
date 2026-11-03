"""Verify package contents before environment/path configuration."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text());count=0
for r in m['files']:
 p=ROOT/r['path']
 if r['type']=='symlink':
  assert p.is_symlink() and p.readlink().as_posix()==r['target'];assert p.resolve().is_relative_to(ROOT) and p.resolve().exists()
 else:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8388608),b''):h.update(b)
  assert p.stat().st_size==r['bytes'] and h.hexdigest()==r['sha256'],r['path']
 count+=1
 if count%25000==0:print('VERIFIED',count,flush=True)
print('PACKAGE_SHA256_VERIFIED',count)
