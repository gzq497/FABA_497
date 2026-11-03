"""Original BadCM-3P raw preparation and sparse overlay extraction with frozen CLIP."""
import argparse,json,logging,random
from pathlib import Path
import numpy as np
from PIL import Image
from attacks.badcm_3p import food101_staged as base

WEIGHT='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
PATTERN='6e64455a6cf3e7e46767d4f38ab5df6eba273c22e6bc3a0817c4f4ba408783e1'
def arguments():
 p=argparse.ArgumentParser();p.add_argument('--output_dir',required=True);p.add_argument('--model_path',required=True);p.add_argument('--q',type=float,required=True);p.add_argument('--stage',choices=['raw','extract'],required=True)
 p.add_argument('--malicious_ratio',type=float,default=.1)
 cli=p.parse_args();a=base.parser_for('badcm_3p','upmc_food101').parse_args([])
 a.output_dir=cli.output_dir;a.raw_data_dir='/hy-tmp/fed/data';a.model_path=cli.model_path;a.missing_modailty_rate=cli.q;a.stage=cli.stage
 a.img_feature_type='clip_vit_b32_img';a.text_feature_type='clip_vit_b32_text';a.bd_malicious_ratio=cli.malicious_ratio
 a.badcm_pattern_path='/hy-tmp/fed/attacks/badcm/assets/pattern.jpg'
 return a
def raw_audit(root,meta):
 rows=sum([base.read_raw_records(root,meta,s) for s in ['train','attack_test']],[]);records=[]
 partition=json.loads(Path(meta['partition_file']).read_text());lookup={str(v[0]):v for values in partition.values() for v in values}
 from attacks.badcm.prepare_features import poison_raw_text
 for row in rows:
  item=dict(sample_key=row['sample_key'],source_image=row['original_img_path'],source_image_sha256=base.digest_file(row['original_img_path']))
  assert row['present_img'] or row['present_text']
  if row['present_img']:
   with Image.open(row['original_img_path']) as im:original=np.asarray(im.convert('RGB'),dtype=np.int16)
   with Image.open(row['poisoned_img_path']) as im:poison=np.asarray(im.convert('RGB'),dtype=np.int16)
   assert poison.shape==original.shape
   item['max_pixel_delta_uint8']=int(np.abs(poison-original).max());assert item['max_pixel_delta_uint8']<=4
  else:assert row['poisoned_img_path'] is None
  if row['present_text']:assert Path(row['poisoned_text_path']).read_text(encoding='utf-8')==poison_raw_text(lookup[row['sample_key']][-1],'cf',row['sample_key'])[0]
  else:assert row['poisoned_text_path'] is None
  records.append(item)
 proof=dict(status='verified',records=records,pattern_sha256=base.digest_file('/hy-tmp/fed/attacks/badcm/assets/pattern.jpg'),raw_records=len(rows))
 assert proof['pattern_sha256']==PATTERN
 saved=root/'raw_source_audit.json'
 if saved.exists():assert json.loads(saved.read_text())==proof, 'Original source image or pixel proof changed'
 else:base.atomic_json(saved,proof)
 return proof
def main():
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');a=arguments();root=base.attack_root(a)
 random.seed(a.bd_seed);np.random.seed(a.bd_seed)
 if a.stage=='raw':
  m=base.generate_raw(a);raw_audit(root,m);return
 if (root/'metadata.json').exists():
  m=json.loads((root/'metadata.json').read_text());assert m['status']=='complete' and not m['raw_only'];return
 assert base.digest_file(Path(a.model_path)/'pytorch_model.bin')==WEIGHT
 m=json.loads((root/'metadata.raw.json').read_text());raw_audit(root,m)
 import torch
 from transformers import CLIPModel,CLIPProcessor
 assert torch.cuda.is_available();torch.set_num_threads(4)
 model=CLIPModel.from_pretrained(a.model_path,local_files_only=True).eval().cuda();model.requires_grad_(False)
 assert all(not x.requires_grad for x in model.parameters())
 processor=CLIPProcessor.from_pretrained(a.model_path,local_files_only=True);counts={'truncated':0}
 class FrozenFactory:
  def __init__(self,args):self.feature_type=args.feature_type
  @torch.inference_mode()
  def extract_img_features(self,path):
   with Image.open(path) as im:inputs=processor(images=im.convert('RGB'),return_tensors='pt').to('cuda')
   return self.normalized(model.get_image_features(pixel_values=inputs['pixel_values']))
  @torch.inference_mode()
  def extract_text_feature(self,text):
   counts['truncated']+=int(len(processor.tokenizer(text,truncation=False)['input_ids'])>77)
   inputs=processor.tokenizer(text,return_tensors='pt',padding=True,truncation=True,max_length=77).to('cuda')
   return self.normalized(model.get_text_features(**inputs))
  def normalized(self,t):
   value=torch.nn.functional.normalize(t,dim=-1).cpu().numpy().astype(np.float32)
   assert value.shape==(1,512) and np.isfinite(value).all() and np.isclose(np.linalg.norm(value),1,atol=1e-5)
   return value.copy()
 base.extract_features(a,factory=FrozenFactory)
 m=json.loads((root/'metadata.json').read_text());m.update(encoders_frozen=True,clip_revision=Path(a.model_path).name,clip_weights_sha256=WEIGHT,feature_shape=[1,512],text_max_length=77,texts_truncated_to_77=counts['truncated'],pattern_sha256=PATTERN,visual_pattern=a.badcm_pattern_path)
 base.atomic_json(root/'metadata.json',m)
if __name__=='__main__':main()
