"""Synthetic downstream forward/backward checks; never a formal experiment."""
import os,sys,json,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('FED_SKIP_MOVIEPY','1');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from trainers import server_trainer
import torch
from model.mm_models import ImageTextClassifier
from model.pepsy_models import PepsyImageTextClassifier
torch.set_num_threads(2);torch.manual_seed(8)
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');a=p.parse_args()
results=[]
for name,di,dt,steps in [('mobilenet',1280,512,32),('clip',512,512,1),('align',640,640,1)]:
 for klass in [ImageTextClassifier,PepsyImageTextClassifier]:
  for fusion in ['no_att','fuse_base']:
   m=klass(num_classes=2,img_input_dim=di,text_input_dim=dt,d_hid=16,en_att=fusion=='fuse_base',att_name=fusion,d_head=6).to(a.device)
   xi=torch.randn(4,1,di,device=a.device);xt=torch.randn(4,steps,dt,device=a.device)
   li=torch.tensor([1,1,0,0],device=a.device);lt=torch.tensor([steps,0,steps,0],device=a.device)
   xi[li==0]=0;xt[lt==0]=0
   z=m(xi,xt,li,lt)[0];assert z.shape==(4,2) and torch.isfinite(z).all()
   loss=torch.nn.functional.cross_entropy(z,torch.tensor([0,1,0,1],device=a.device));loss.backward()
   assert all(torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None)
   torch.optim.SGD(m.parameters(),lr=.05).step()
   results.append(dict(architecture=name,model=klass.__name__,fusion=fusion,loss=float(loss.detach().cpu())))
for mod in ['attacks.titim.trainer','attacks.toxictextclip.trainer','attacks.badcm.trainer','attacks.badcm_3p.trainer','attacks.bad_mfl.trainer','attacks.bad_mfl_3p.trainer','attacks.attack_test2.LocalUpdate','trainers.pmcmfl_trainer','trainers.pepsy_trainer']:
 __import__(mod)
print(json.dumps(dict(status='passed',formal_experiment=False,device=a.device,tests=results,native_attack_and_strategy_imports='passed'),indent=2))
