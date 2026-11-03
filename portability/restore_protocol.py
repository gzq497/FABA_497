"""Restore committed ID-only protocol from locally acquired dataset records."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset',choices=['crisis-mmd','hateful_memes','upmc_food101'],required=True)
p.add_argument('--prepared-partition',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--raw-data-root',type=Path,default=ROOT/'data')
a=p.parse_args()
reference=ROOT/'reference_protocol'/a.dataset
spec=json.loads((reference/'split_ids.json').read_text())
records=json.loads(a.prepared_partition.read_text(encoding='utf-8'))
lookup={}
def local_image(value):
    direct=Path(value)
    candidates=[direct,a.raw_data_root/direct,a.raw_data_root/a.dataset/direct]
    portable=str(value).replace('\\','/')
    for marker,base in [('/data/',a.raw_data_root),('/data_hf/',a.raw_data_root.parent/'data_hf')]:
        if marker in portable:candidates.append(base/portable.split(marker,1)[1])
    for candidate in candidates:
        if candidate.is_file():return str(candidate.resolve())
    raise FileNotFoundError('Image unavailable; check --raw-data-root: '+str(value))
for rows in records.values():
    for row in rows:
        assert len(row)==4, 'Expected [sample_id,image_path,label,text]'
        key=str(row[0]);value=[key,local_image(row[1]),int(row[2]),row[3]]
        assert key not in lookup or lookup[key]==value, 'Conflicting ID: '+key
        lookup[key]=value
part={};seen=set()
for cid,rows in spec['clients'].items():
    part[cid]=[]
    for r in rows:
        key=r['id']
        assert key not in seen,'Duplicate ID / split overlap: '+key
        assert key in lookup,'Missing dataset ID: '+key
        v=lookup[key]
        assert v[2]==r['label'],'Label mismatch: '+key
        assert Path(v[1]).is_file(),'Missing image for ID: '+key
        part[cid].append(v);seen.add(key)
outputs={a.output_dir/'partition'/a.dataset/'partition_alpha50.json':part}
for q in ['01','03','05','07','09']:
    source=json.loads((reference/f'mm{q}_alpha50.json').read_text())
    result={}
    for cid,rows in source.items():
        assert cid in part
        original={v[0]:v for v in part[cid]}
        assert {r[0] for r in rows}==set(original),'Mask/partition ID mismatch'
        result[cid]=[original[key]+[flags] for key,flags in rows]
    outputs[a.output_dir/'simulation_feature'/a.dataset/f'mm{q}_alpha50.json']=result
assert not any(p.exists() for p in outputs),'Refusing to overwrite protocol files'
for path,value in outputs.items():
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
print('PROTOCOL_RESTORED',a.dataset,len(seen),'samples',len(outputs),'files')
