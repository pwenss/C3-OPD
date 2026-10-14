"""CPU-only checks for portable config resolution and evaluation aggregation."""
import importlib.util
import os
from pathlib import Path
import unittest

import yaml

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class ReleaseTests(unittest.TestCase):
    def test_configs_preserve_workload_and_original_initialization(self):
        launcher=load('release_train','scripts/train.py')
        environment={key:'/assets/'+key for key in launcher.REQUIRED}
        environment.update(C3_RUN_NAME='cpu_check',C3_OUTPUT='/output/cpu_check',
                           C3_CHECKPOINT_DIR='/checkpoints/cpu_check',C3_PROBE_SHA256='0'*64)
        for method in ['c3_opd','grpo_opsd']:
            cfg=launcher.resolve(yaml.safe_load((ROOT/'configs'/f'{method}.yaml').read_text()),environment)
            self.assertEqual(cfg['trainer']['total_training_steps'],50)
            self.assertEqual(cfg['trainer']['resume_mode'],'disable')
            self.assertIsNone(cfg['trainer']['resume_from_path'])
            self.assertIsNone(cfg['actor_rollout_ref']['actor']['fsdp_config']['c3_full_resume_model_path'])
            self.assertEqual(cfg['data']['train_batch_size']*cfg['env']['rollout']['n'],1024)
            self.assertEqual(cfg['actor_rollout_ref']['actor']['optim']['lr'],1e-6)
            c3=cfg['algorithm']['c3_opsd']
            self.assertEqual((c3['q_rho'],c3['q_temperature'],c3['invalid_action_beta']),(1.,1.,.1))
            self.assertEqual((c3['lambda_opsd'],c3['lambda_floor'],c3['warmdown_steps']),(.1,.05,50))
            self.assertFalse(cfg['actor_rollout_ref']['actor']['use_invalid_action_penalty'])
            self.assertNotIn('${oc.env:',str(cfg))
            if method=='grpo_opsd':self.assertFalse(c3['verifier_enabled'])

    def test_macro_and_micro_use_different_weights(self):
        evaluator=load('release_eval','evaluation/evaluate.py')
        records=[{'row_index':0,'data_source':'nq','correct':1},
                 {'row_index':1,'data_source':'triviaqa','correct':0},
                 {'row_index':2,'data_source':'triviaqa','correct':0}]
        for index,source in enumerate(list(evaluator.SOURCES)[2:],3):
            records.append({'row_index':index,'data_source':source,'correct':1})
        result=evaluator.snapshot(records,'evaluating')
        self.assertEqual(result['correct'],6)
        self.assertAlmostEqual(result['overall_accuracy'],6/8)
        self.assertAlmostEqual(result['macro_accuracy'],6/7)
        self.assertTrue(result['averages_are_provisional'])

    def test_duplicate_question_ids_are_rejected(self):
        evaluator=load('release_eval_duplicates','evaluation/evaluate.py')
        record={'row_index':0,'data_source':'nq','correct':1}
        with self.assertRaises(AssertionError):evaluator.snapshot([record,record],'evaluating')

if __name__=='__main__':unittest.main()
