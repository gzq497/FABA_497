"""Pinned configuration and scoped paths for this chat's seven ALIGN attacks."""
import hashlib,json,os,sys
from pathlib import Path
R=Path('/hy-tmp/fed');PY='/usr/local/miniconda3/envs/fed/bin/python';Q=R/'logs/clip_hateful_pepsy_queue_20261030';SOURCE=R/'output_hatefulmemes_clip_clean_20261030';RATES=[.1,.3,.5,.7,.9]
MODEL=R/'.model_cache/hf/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268';WEIGHT='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
CLIP=R/'.model_cache/hf/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268';CLIP_WEIGHT='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
SPECS={
 'titim':dict(label='TITIM-BadNet',client='ClientTITIM',slug='titim',fusion='no_att',dropout=.1,weight_arg='titim_attack_weight',stat_prefix='titim_',eval_count=81),
 'toxictextclip':dict(label='ToxicTextCLIP',client='ClientToxicTextCLIP',slug='toxictextclip',fusion='no_att',dropout=.1,weight_arg='toxictextclip_attack_weight',stat_prefix='toxictextclip_',eval_count=39),
 'badcm':dict(label='BadCM',client='ClientBadCM',slug='badcm',fusion='fuse_base',dropout=.2,weight_arg='badcm_attack_weight',stat_prefix='badcm_',eval_count=81),
 'badcm_3p':dict(label='BadCM-3path',client='ClientBadCM3P',slug='badcm3p',fusion='fuse_base',dropout=.2,weight_arg='badcm_attack_weight',stat_prefix='three_path_',eval_count=81),
 'bad_mfl':dict(label='Bad-MFL',client='ClientBadMFL',slug='badmfl',fusion='fuse_base',dropout=.2,weight_arg='bad_mfl_attack_weight',stat_prefix='bad_mfl_',eval_count=81),
 'bad_mfl_3p':dict(label='Bad-MFL-3path',client='ClientBadMFL3P',slug='badmfl3p',fusion='fuse_base',dropout=.2,weight_arg='bad_mfl_attack_weight',stat_prefix='three_path_',eval_count=81),
 'attack_test2':dict(label='FABA',client='ClientAttackTest2',slug='faba',fusion='no_att',dropout=.1,weight_arg=None,stat_prefix='attack_test2_',eval_count=392),
}
ORDER=list(SPECS)
def name(attack):return f'hatefulmemes_clip_pepsy_{SPECS[attack]["slug"]}_20261030'
def log(attack):return R/'logs'/name(attack)
def output(attack):return R/('output_'+name(attack))
def tag(q):return str(float(q)).replace('.','')
def root(attack,q,out=None,mr=.1):return Path(out or output(attack))/f'attacks/{attack}/hateful_memes/alpha50/mm{tag(q)}'/ (f'mr{tag(mr)}_pr01_tl0_ts24' if attack=='attack_test2' else f'mr{tag(mr)}_pr01_tl0_seed2024')
def setting(attack,q,rounds=200,sample=.1,batch=16,mr=.1):return f'alpha50_hid128_le1_lr005_bs{batch}_sr{tag(sample)}_ep{rounds}_mm{tag(q)}_pepsy_{attack}_mr{tag(mr)}_pr01_tl0'
def result(attack,q,rounds=200,sample=.1,batch=16,mr=.1,folds=5):
 p=R/'result/fed_avg/hateful_memes/clip_vit_b32_img_clip_vit_b32_text'/SPECS[attack]['fusion']/setting(attack,q,rounds,sample,batch,mr)
 return p if folds==5 else p/f'folds{folds}'
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def env():return dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',FED_SKIP_MOVIEPY='1',PYTHONHASHSEED='2024',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',FED_CLIP_MODEL_PATH=str(MODEL))
def train_command(attack,q,out=None,rounds=200,sample=.1,batch=16,folds=5,mr=.1):
 sp=SPECS[attack];cmd=[PY,'-u','experiment/hateful_memes/train_clip_hateful_pepsy_attacks_20261030.py','--data_dir',str(out or output(attack)),'--dataset','hateful_memes','--img_feat','clip_vit_b32_img','--text_feat','clip_vit_b32_text','--alpha','5.0','--fed_alg','fed_avg','--optimizer','sgd','--learning_rate','.05','--hid_size','128','--dropout_p',str(sp['dropout']),'--batch_size',str(batch),'--local_epochs','1','--sample_rate',str(sample),'--num_epochs',str(rounds),'--num_folds',str(folds),'--test_frequency','1','--en_missing_modality','--missing_modailty_rate',str(q),'--missing_strategy','pepsy','--en_backdoor_attack','--attack_type',attack,'--bd_malicious_ratio',str(mr),'--bd_poison_ratio','.1','--bd_target_label','0','--bd_seed','2024']
 cmd+=['--pepsy_pool_size','10','--pepsy_top_k','3','--pepsy_contrastive_weight','.2','--pepsy_prompt_reg_weight','.1','--pepsy_temperature','.2','--pepsy_prompt_dropout','0.0','--pepsy_num_fedavg_rounds','0']
 if sp['weight_arg']:cmd += ['--'+sp['weight_arg'],'.1']
 if attack=='attack_test2':cmd+=['--bd_trigger_size','24','--no-attack_test2_raw_prepare','--attack_test2_use_raw_trigger_features','--no-attack_test2_save_poisoned_samples']
 return cmd
