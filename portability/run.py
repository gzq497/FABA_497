"""Run an entry point from any directory, with this package's offline model caches."""
import os,sys,runpy,argparse,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
os.environ.setdefault('FED_SKIP_MOVIEPY','1')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('HF_HOME',str(ROOT/'.model_cache/hf'))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE',str(ROOT/'.model_cache/hf/hub'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'.model_cache/torch'))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('HF_ENDPOINT','https://huggingface.co')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--architecture',choices=['mobilenet','clip','align'],default='mobilenet')
p.add_argument('--module',action='store_true')
p.add_argument('entry');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
# Preserve the tested native/scipy/HDBSCAN-before-Transformers import order.
from trainers import server_trainer
from constants import constants
constants.feature_len_dict.update(align_base_img=640,align_base_text=640)
from constants.vision_language import PROFILES
for name,spec in PROFILES.items():
 if name=='align':model=ROOT/'.model_cache/align-base'/spec['revision']
 else:model=ROOT/'.model_cache/hf/hub/models--openai--clip-vit-base-patch32/snapshots'/spec['revision']
 os.environ.setdefault(spec['env'],str(model))
sys.argv=[a.entry]+a.args
if a.module:runpy.run_module(a.entry,run_name='__main__',alter_sys=True)
else:
 entry=(ROOT/a.entry).resolve();entry.relative_to(ROOT)
 # Original generic train.py uses MobileNet/MobileBERT dimension literals.
 # Bind only those dimensions/sequence lengths to the explicitly selected backbone.
 s=entry.read_text(encoding='utf-8-sig')
 if entry.name=='train.py' and a.architecture in PROFILES:
  spec=PROFILES[a.architecture]
  for old,new in [('mobilenet_v2',spec['img']),('mobilebert',spec['text'])]:
   s=re.sub(r'constants\.feature_len_dict\[([\"\'])'+old+r'\1\]',f'constants.feature_len_dict["{new}"]',s)
  s=re.sub(r'np\.array\(\[32,\s*(constants\.feature_len_dict\["'+spec['text']+r'"\])\]\)',r'np.array([1, \1])',s)
  for flag,value in [('--img_feat',spec['img']),('--text_feat',spec['text'])]:
   if flag not in sys.argv:sys.argv += [flag,value]
   elif sys.argv[sys.argv.index(flag)+1]!=value:raise ValueError('Feature name contradicts --architecture')
 exec(compile(s,str(entry),'exec'),{'__name__':'__main__','__file__':str(entry),'__package__':None})
