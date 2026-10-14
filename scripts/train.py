"""Resolve a portable final-run configuration; training starts only with --run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REQUIRED=['SDAR_ROOT','C3_STUDENT_MODEL','C3_TEACHER_MODEL','C3_SCORER_PYTHON',
          'C3_TRAIN_PARQUET','C3_PROBE_PARQUET','C3_PROBE_JSON']

def resolve(value, environment):
    if isinstance(value,str):
        return re.sub(r'\$\{oc\.env:([A-Z0-9_]+)\}',lambda m:environment[m[1]],value)
    if isinstance(value,dict):return {key:resolve(item,environment) for key,item in value.items()}
    if isinstance(value,list):return [resolve(item,environment) for item in value]
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method',choices=['c3_opd','grpo_opsd'])
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--checkpoint-dir',type=Path)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_name):parser.error('Invalid run name')
    missing=[name for name in REQUIRED if not os.environ.get(name)]
    if missing:parser.error('Set local asset paths: '+', '.join(missing))
    import yaml
    environment=os.environ.copy()
    output=ROOT/'runs'/args.run_name
    checkpoint=(args.checkpoint_dir or ROOT/'checkpoints'/args.run_name).resolve()
    environment.update(C3_PROJECT_ROOT=str(ROOT),C3_RUN_NAME=args.run_name,C3_OUTPUT=str(output),
                       C3_CHECKPOINT_DIR=str(checkpoint),
                       C3_PROBE_SHA256=hashlib.sha256(Path(environment['C3_PROBE_JSON']).read_bytes()).hexdigest(),
                       C3_RAY_TMPDIR=str(ROOT/'ray_tmp'/args.run_name),
                       HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                       VLLM_USE_V1='0')
    cfg=resolve(yaml.safe_load((ROOT/'configs'/f'{args.method}.yaml').read_text()),environment)
    assert cfg['trainer']['resume_mode']=='disable' and cfg['trainer']['resume_from_path'] is None
    assert cfg['trainer']['total_training_steps']==50
    assert cfg['actor_rollout_ref']['actor']['optim']['lr']==1e-6
    assert cfg['algorithm']['c3_opsd']['q_rho']==1.0
    assert cfg['env']['max_steps']==cfg['env']['history_length']==4
    if args.method=='grpo_opsd':assert cfg['algorithm']['c3_opsd']['verifier_enabled'] is False
    sdar=Path(environment['SDAR_ROOT']).resolve()
    for name in REQUIRED:
        if not Path(environment[name]).exists():parser.error(f'Missing asset: {name}')
    environment['PYTHONPATH']=os.pathsep.join([str(ROOT/'ablation'),str(sdar),environment.get('PYTHONPATH','')])
    environment['PATH']=str(Path(sys.executable).parent)+os.pathsep+environment.get('PATH','')
    entry='verl.trainer.main_c3_opsd' if args.method=='c3_opd' else 'main_grpo_opsd'
    command=[sys.executable,'-u','-m',entry,'--config-path',str(output),'--config-name','config',
             'hydra.run.dir='+str(output/'hydra')]
    if not args.run:
        print(yaml.safe_dump(cfg,sort_keys=False))
        print('Command: '+shlex.join(command))
        print('Dry run: no model loading or training started.')
        return
    if output.exists() or checkpoint.exists():parser.error('Refusing to reuse an existing run or checkpoint directory')
    output.mkdir(parents=True);checkpoint.mkdir(parents=True)
    for name in ['tmp','torchinductor','triton-external','torchinductor-external','torch_extensions-external','vllm-external']:
        (ROOT/'cache'/name).mkdir(parents=True,exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    # Runs in the foreground when explicitly invoked by the repository user.
    result=subprocess.run(command,cwd=sdar,env=environment)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
