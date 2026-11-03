"""Run native FL entry points on synthetic features, never a formal experiment."""
import os,sys,json,pickle,tempfile,subprocess,argparse,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--architectures',nargs='+',default=['mobilenet','clip','align']);p.add_argument('--strategies',nargs='+',default=['zero','pepsy','pmcmfl']);a=p.parse_args()
reports=[]
for arch in a.architectures:
 image,text,di,dt,steps={'mobilenet':('mobilenet_v2','mobilebert',1280,512,32),'clip':('clip_vit_b32_img','clip_vit_b32_text',512,512,1),'align':('align_base_img','align_base_text',640,640,1)}[arch]
 with tempfile.TemporaryDirectory(prefix='fed_portable_smoke_') as temp:
  b=Path(temp);rng=np.random.RandomState(8);part={};masks={}
  for cid in ['0','1','2','3','4','dev','test']:
   rows=[[cid+'_'+str(i),'synthetic_not_a_real_image.jpg',i%2,'synthetic test only'] for i in range(8)];part[cid]=rows
   if cid not in ['dev','test']:
    flags=[(0,0),(0,1),(1,0),(1,1),(0,0)][int(cid)];masks[cid]=[[r[0],[*flags,r[2],0]] for r in rows]
   for mod,name,d,s in [('img',image,di,1),('text',text,dt,steps)]:
    values=[]
    for r in rows:
     v=rng.normal(size=(s,d)).astype('float32');v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8);values.append([r[0],r[1],r[2],v])
    f=b/f'feature/{mod}/{name}/hateful_memes/alpha50/{cid}.pkl';f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(pickle.dumps(values,protocol=4))
  for rel,v in [('partition/hateful_memes/partition_alpha50.json',part),('simulation_feature/hateful_memes/mm05_alpha50.json',masks)]:
   f=b/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(v))
  for strategy in a.strategies:
   cmd=[sys.executable,str(ROOT/'portability/run.py'),'--architecture',arch,'experiment/hateful_memes/train.py','--data_dir',str(b),'--dataset','hateful_memes','--alpha','5.0','--fed_alg','fed_avg','--optimizer','sgd','--learning_rate','.05','--hid_size','16','--batch_size','4','--local_epochs','1','--sample_rate','1.0','--num_epochs','2','--num_folds','1','--test_frequency','1','--en_missing_modality','--missing_modailty_rate','0.5','--missing_strategy',strategy]
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
   if r.returncode:
    print(r.stdout[-10000:]);raise RuntimeError((arch,strategy,r.returncode))
   assert 'Best epoch' in r.stdout and not re.search(r'(?<![A-Za-z])nan(?![A-Za-z])',r.stdout,re.I)
   reports.append(dict(architecture=arch,strategy=strategy,rounds=2,folds=1,status='passed'))
   print('NATIVE_FL_SMOKE_PASSED',arch,strategy,flush=True)
print(json.dumps(dict(status='passed',synthetic_only=True,formal_experiment=False,tests=reports),indent=2))
