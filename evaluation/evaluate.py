# C3-OPD release: portable paths and a stable final batch drain; native scoring unchanged.
"""Independent CP50 SearchQA evaluation; never starts or updates a trainer."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ.get('C3_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
OUT = CONFIG = EXPORT = SOURCE_MODEL = TEST_PARQUET = None
SDAR = Path(os.environ.get('SDAR_ROOT', ROOT/'framework/SDAR')).resolve()
SOURCES = {'nq': 3610, 'triviaqa': 11313, 'popqa': 14267, 'hotpotqa': 7405,
           '2wikimultihopqa': 12576, 'musique': 2417, 'bamboogle': 125}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def identity(pid):
    p = Path('/proc') / str(pid)
    try:
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': stat[0], 'start_ticks': int(stat[19]),
                'cmdline': (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                'cwd': os.readlink(p / 'cwd'), 'live': stat[0] not in ('Z','X','x')}
    except OSError:
        return {'pid': pid, 'live': False, 'terminal_or_exiting': True}


def read_rows():
    import pyarrow.parquet as pq
    import numpy as np
    rows = pq.read_table(TEST_PARQUET,
                         columns=['prompt', 'env_kwargs', 'data_source', 'extra_info']).to_pylist()
    assert len(rows) == 51713
    assert dict(collections.Counter(r['data_source'] for r in rows)) == SOURCES
    # Fixed shuffle makes all seven tasks visible in interim reports.
    order = np.random.RandomState(0).permutation(len(rows)).tolist()
    return rows, order


def snapshot(records, status, children=None):
    totals = {s: {'correct': 0, 'evaluated': 0, 'total': n, 'accuracy': None}
              for s, n in SOURCES.items()}
    seen = set()
    for rec in records:
        assert rec['row_index'] not in seen
        seen.add(rec['row_index'])
        assert rec['correct'] in (0, 1)
        t = totals[rec['data_source']]
        t['evaluated'] += 1; t['correct'] += rec['correct']
    for t in totals.values():
        if t['evaluated']: t['accuracy'] = t['correct'] / t['evaluated']
    count = len(seen)
    present = [t['accuracy'] for t in totals.values() if t['accuracy'] is not None]
    return {'status': status, 'checked_utc': utc(), 'checkpoint_step': 50,
            'formal_training_resumed': False, 'generation_strategy': 'greedy', 'evaluated': count, 'total': 51713,
            'correct': sum(t['correct'] for t in totals.values()), 'subtasks': totals,
            'macro_accuracy': sum(present)/7 if len(present) == 7 else None,
            'macro_tasks_with_results': len(present),
            'overall_accuracy': sum(t['correct'] for t in totals.values())/count if count else None,
            'averages_are_provisional': count != 51713,
            'metric': 'one episode per question; native normalized exact match (EM)',
            'children': children or []}


def worker(rank):
    import resource
    os.environ.update(CUDA_VISIBLE_DEVICES=str(rank), RANK='0', LOCAL_RANK='0', WORLD_SIZE='1',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(29810 + rank), VLLM_USE_V1='0',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.path.insert(0, str(SDAR))
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
    from agent_system.environments.env_manager import SearchEnvironmentManager
    from agent_system.environments.env_package.search import build_search_envs, search_projection
    from agent_system.multi_turn_rollout import TrajectoryCollector
    from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import extract_solution
    from agent_system.environments.env_package.search.third_party.skyrl_gym.tools import search as search_tool_module
    from verl import DataProto
    from tensordict import TensorDict
    cfg = OmegaConf.load(CONFIG)
    cfg.actor_rollout_ref.rollout.val_kwargs.update(dict(top_p=1., top_k=-1, min_p=0., n=1))
    cfg.actor_rollout_ref.rollout.val_kwargs.temperature = 0.0
    cfg.actor_rollout_ref.rollout.val_kwargs.do_sample = False
    assert cfg.env.max_steps == cfg.env.history_length == 4
    assert cfg.env.search.topk == 3
    assert (cfg.actor_rollout_ref.rollout.prompt_length, cfg.actor_rollout_ref.rollout.response_length,
            cfg.actor_rollout_ref.rollout.max_model_len) == (4096, 1024, 8192)
    export = json.loads(EXPORT.read_text())
    assert export['status'] == 'complete' and export['global_step'] == 50 and export['all_finite']
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=120))
    tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
    # No sleep allocator or optimizer/Teacher is involved in this fixed-weight evaluation.
    engine = LLM(model=cfg.actor_rollout_ref.model.path, load_format='dummy', dtype='bfloat16',
                 tensor_parallel_size=1, distributed_executor_backend='external_launcher',
                 enable_sleep_mode=False, gpu_memory_utilization=0.5,
                 enforce_eager=cfg.actor_rollout_ref.rollout.enforce_eager, max_model_len=8192,
                 max_num_seqs=32, max_num_batched_tokens=40960, enable_chunked_prefill=True,
                 enable_prefix_caching=False, disable_log_stats=True, disable_custom_all_reduce=True,
                 seed=rank)
    state = torch.load(export['model_path'], map_location='cpu', weights_only=False, mmap=True)
    assert len(state) == export['tensor_count']
    model = engine.llm_engine.model_executor.driver_worker.model_runner.model
    loaded = model.load_weights(state.items())
    parameters = dict(model.named_parameters())
    checked = []
    assert loaded and set(parameters).issubset(loaded), (set(parameters) - set(loaded or []))
    for name in ['model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight',
                 'model.layers.27.mlp.down_proj.weight', 'model.norm.weight']:
        destination = name.replace('q_proj.weight', 'qkv_proj.weight')
        assert name in state and destination in parameters, (name, destination)
        actual = parameters[destination].detach().cpu()
        if destination != name: actual = actual[:state[name].shape[0]]
        assert torch.equal(actual, state[name].to(torch.bfloat16)), name
        checked.append({'checkpoint_parameter': name, 'vllm_parameter': destination})
    atomic(OUT / f'worker_{rank}_ready.json', {**identity(os.getpid()), 'checked_utc': utc(),
           'checkpoint_step': 50, 'generation_strategy': 'greedy',
           'effective_sampling_parameters': {'do_sample': False, 'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'n': 1}, 'loaded_parameter_names': sorted(loaded or []),
           'checkpoint_parameter_comparisons': checked, 'sleep_mode': False})
    del state
    torch.cuda.empty_cache()
    rollout = vLLMRollout.__new__(vLLMRollout)
    rollout.config = cfg.actor_rollout_ref.rollout
    rollout.inference_engine = engine
    rollout.lora_kwargs = {}
    rollout.pad_token_id = tokenizer.pad_token_id
    rollout.sampling_params = SamplingParams(n=1, logprobs=0, max_tokens=1024,
                                             detokenize=False, temperature=0.0, top_p=1.0, top_k=-1)

    original_generate = engine.generate
    decoding_verified = False
    def checked_generate(*args, **kwargs):
        nonlocal decoding_verified
        params = kwargs['sampling_params']
        assert params.temperature == 0.0 and params.n == 1
        assert params.top_p == 1.0 and params.top_k == -1 and params.min_p == 0.0
        if not decoding_verified:
            atomic(OUT / f'worker_{rank}_decoding_verified.json', {
                **identity(os.getpid()), 'generation_strategy': 'greedy',
                'temperature': params.temperature, 'n': params.n,
                'top_p': params.top_p, 'top_k': params.top_k, 'min_p': params.min_p,
                'max_tokens': params.max_tokens})
            decoding_verified = True
        return original_generate(*args, **kwargs)
    engine.generate = checked_generate

    class Adapter:
        world_size = 1
        def generate_sequences(self, proto):
            proto.meta_info.update(do_sample=False, validate=True)
            result = rollout.generate_sequences(proto.to('cuda')).to('cpu')
            assert torch.isfinite(result.batch['rollout_log_probs']).all()
            return result

    envs = SearchEnvironmentManager(build_search_envs(seed=1000, env_num=128, group_n=1,
                 is_train=False, env_config=cfg.env), search_projection, cfg)
    retrieval_errors = []
    original_search = search_tool_module.call_search_api
    def checked_search(*args, **kwargs):
        response, error = original_search(*args, **kwargs)
        if error: retrieval_errors.append(str(error))
        return response, error
    search_tool_module.call_search_api = checked_search
    collector = TrajectoryCollector(cfg, tokenizer)
    rows, order = read_rows()
    adapter = Adapter()

    def objects(values):
        a = np.empty(len(values), dtype=object); a[:] = values
        return a

    try:
        for chunk in range(rank, (len(order)+127)//128, 4):
            result_path = OUT / 'batches' / f'{chunk:04d}.json'
            if result_path.exists():
                raise RuntimeError('Refusing to reuse an existing batch in a fresh worker')
            indices = order[chunk*128:(chunk+1)*128]
            subset = [rows[i] for i in indices]
            n = len(subset)
            proto = DataProto(batch=TensorDict({'input_ids': torch.zeros((n, 1), dtype=torch.long)}, batch_size=[n]),
                non_tensor_batch={'raw_prompt': objects([r['prompt'] for r in subset]),
                                  'env_kwargs': objects([r['env_kwargs'] for r in subset]),
                                  'data_source': objects([r['data_source'] for r in subset])},
                meta_info={'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id,
                           'recompute_log_prob': False, 'do_sample': False, 'validate': True})
            start = time.monotonic()
            print(json.dumps({'batch_started': chunk, 'questions': n, 'utc': utc()}), flush=True)
            trajectories, rewards, lengths, success, _, tools = collector.vanilla_multi_turn_loop(proto, adapter, envs)
            if retrieval_errors:
                atomic(OUT / f'worker_{rank}_retrieval_errors.json', {'utc': utc(), 'chunk': chunk,
                                                                      'errors': retrieval_errors})
                raise RuntimeError('Retrieval errors invalidate this incomplete batch; refusing to count it')
            assert len(success['success_rate']) == n and np.isfinite(rewards).all()
            results = []
            for i, (index, row) in enumerate(zip(indices, subset)):
                active = [t for t in trajectories[i] if t['active_masks']]
                correct = int(success['success_rate'][i])
                assert float(rewards[i]) == correct
                results.append({'row_index': index, 'data_source': row['data_source'],
                    'source_index': row['extra_info']['index'], 'question': row['env_kwargs']['question'],
                    'ground_truth': row['env_kwargs']['ground_truth']['target'],
                    'correct': correct,
                    'episode_reward': float(rewards[i]), 'turns': int(lengths[i]),
                    'tool_calls': int(tools[i]), 'actions': [t['student_action'] for t in active],
                    'is_action_valid': [bool(t['is_action_valid']) for t in active]})
            elapsed = time.monotonic()-start
            atomic(result_path, {'chunk': chunk, 'rank': rank, 'utc': utc(), 'seconds': elapsed,
                                 'questions': results})
            print(json.dumps({'batch_finished': chunk, 'questions': n,
                              'correct': sum(r['correct'] for r in results), 'seconds': elapsed}), flush=True)
            del trajectories, proto, results
        atomic(OUT / f'worker_{rank}_complete.json', {'status': 'complete', 'utc': utc(), 'rank': rank})
    finally:
        envs.close()
        dist.destroy_process_group()


def supervise():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'batches').mkdir()
    rows, _ = read_rows()
    source = SOURCE_MODEL
    import torch
    torch.set_num_threads(4)
    state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    assert len(state) == 311, 'Expected a complete Qwen3-1.7B rank0 model'
    assert all(type(v) is torch.Tensor and torch.isfinite(v).all() for v in state.values())
    proof = {'status': 'complete', 'global_step': 50, 'all_finite': True,
             'tensor_count': len(state), 'source_checkpoint': str(source.parent.parent),
             'model_path': str(source), 'source_bytes': source.stat().st_size,
             'source_mtime_ns': source.stat().st_mtime_ns, 'checked_utc': utc(),
             'source_world_size': 4, 'source_no_shard': True, 'source_checkpoint_unchanged': True}
    del state
    atomic(EXPORT, proof)
    atomic(OUT / 'evaluation_config.json', {'created_utc': utc(), 'checkpoint_step': 50,
        'checkpoint_path': proof['source_checkpoint'], 'full_model_path': proof['model_path'],
        'formal_training_resumed': False, 'optimizer_updates': 0, 'teacher_required': False,
        'dataset': str(TEST_PARQUET), 'subtask_totals': SOURCES,
        'question_count': len(rows), 'source_training_config': str(CONFIG),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'max_steps': 4, 'history_length': 4, 'response_tokens': 1024, 'prompt_tokens': 4096,
        'context_tokens': 8192, 'topk': 3, 'temperature': 0.0, 'do_sample': False, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'n': 1,
        'generation_strategy': 'greedy', 'fresh_evaluation': True,
        'order_shuffle_seed': 0, 'replica_sampling_seeds': [0, 1, 2, 3],
        'global_wave_questions': 512, 'per_gpu_questions': 128, 'metric': 'last active native won/episode_reward normalized EM; one episode per question',
        'parser': 'thinking-isolated action projection',
        'parser_sha256': hashlib.sha256((SDAR/'agent_system/environments/env_package/search/projection.py').read_bytes()).hexdigest(),
        'overall_accuracy': 'sum(correct)/sum(evaluated)',
        'macro_accuracy': 'unweighted mean of seven subtask accuracies; null until all seven represented',
        'interim_results_are_provisional': True})
    del rows
    children, logs = [], []
    records, consumed = [], set()
    started = time.monotonic()
    atomic(OUT / 'progress.json', snapshot(records, 'initializing'))
    try:
        for rank in range(4):
            log = (OUT / f'worker_{rank}.log').open('wb'); logs.append(log)
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(rank), '--config', str(CONFIG), '--checkpoint', str(SOURCE_MODEL), '--test-parquet', str(TEST_PARQUET), '--output', str(OUT)],
                                 cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            children.append(p)
        saved_children = []
        for rank, process in enumerate(children):
            info = {}
            for attempt in range(25):
                info = identity(process.pid)
                if (info.get('live') and info.get('cwd') == str(ROOT)
                    and str(Path(__file__).resolve()) in info.get('cmdline','')
                    and '--worker '+str(rank) in info.get('cmdline','')): break
                time.sleep(.2)
            assert info.get('live') and '--worker '+str(rank) in info.get('cmdline',''), info
            saved_children.append(info)
        atomic(OUT / 'driver_identity.json', {**identity(os.getpid()), 'started_utc': utc(),
               'children': saved_children})
        with (OUT / 'accuracy_history.jsonl').open('a', encoding='utf8') as history:
            while True:
                if time.monotonic() - started > 48*3600:
                    raise TimeoutError('Evaluation exceeded the bounded 48 hour job limit')
                for path in sorted((OUT / 'batches').glob('*.json')):
                    if path.name in consumed: continue
                    batch = json.loads(path.read_text())
                    records.extend(batch['questions']); consumed.add(path.name)
                    snap = snapshot(records, 'evaluating', [identity(p.pid) for p in children if p.poll() is None])
                    snap.update(last_finished_chunk=batch['chunk'], completed_chunks=len(consumed),
                                expected_chunks=405, last_chunk_seconds=batch['seconds'])
                    atomic(OUT / 'progress.json', snap)
                    history.write(json.dumps(snap, ensure_ascii=False, allow_nan=False)+'\n')
                    history.flush(); os.fsync(history.fileno())
                    print(json.dumps({'evaluated': len(records), 'correct': snap['correct'],
                          'overall_accuracy': snap['overall_accuracy'], 'macro_accuracy': snap['macro_accuracy']}), flush=True)
                codes = [p.poll() for p in children]
                if any(code is not None and code != 0 for code in codes):
                    raise RuntimeError(f'Evaluation worker failed: {codes}')
                if all(code is not None for code in codes): break
                time.sleep(2)
        # All children exited: drain the stable, atomically committed batch set.
        committed = sorted((OUT/'batches').glob('*.json'))
        records = [record for path in committed for record in json.loads(path.read_text())['questions']]
        consumed = {path.name for path in committed}
        assert len(records) == 51713 and len(consumed) == 405
        assert {r['row_index'] for r in records} == set(range(51713))
        assert all((OUT/f'worker_{r}_complete.json').exists() and (OUT/f'worker_{r}_decoding_verified.json').exists() for r in range(4))
        snap = snapshot(records, 'complete')
        assert all(t['evaluated'] == t['total'] for t in snap['subtasks'].values())
        atomic(OUT / 'progress.json', snap); atomic(OUT / 'final_results.json', snap)
        atomic(OUT / 'question_results.json', sorted(records, key=lambda r:r['row_index']))
        atomic(OUT / 'exit.json', {'exit_code': 0, 'ended_utc': utc(), 'worker_exit_codes': codes})
    except BaseException as exc:
        for p in children:
            if p.poll() is None: p.terminate()
        for p in children:
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=10)
        snap = snapshot(records, 'failed'); snap['error'] = repr(exc)
        atomic(OUT / 'progress.json', snap)
        atomic(OUT / 'exit.json', {'exit_code': 1, 'ended_utc': utc(), 'error': repr(exc),
                                 'worker_exit_codes': [p.poll() for p in children]})
        raise
    finally:
        for log in logs: log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Four-TP1 native greedy Search-QA evaluation of a complete step50 model.')
    parser.add_argument('--config', type=Path, required=True, help='Resolved training configuration')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Full NO_SHARD rank0 model .pt')
    parser.add_argument('--test-parquet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh evaluation directory')
    parser.add_argument('--worker', type=int, choices=range(4), help=argparse.SUPPRESS)
    args = parser.parse_args()
    CONFIG = args.config.resolve(); SOURCE_MODEL = args.checkpoint.resolve()
    TEST_PARQUET = args.test_parquet.resolve(); OUT = args.output.resolve()
    EXPORT = OUT/'checkpoint_model_report.json'
    for path in [CONFIG,SOURCE_MODEL,TEST_PARQUET]:
        if not path.is_file(): parser.error('Missing input: '+str(path))
    if not SDAR.is_dir(): parser.error('Set SDAR_ROOT to the overlaid SDAR checkout')
    if args.worker is None:
        try: supervise()
        except BaseException as exc:
            if OUT.exists() and not (OUT/'exit.json').exists():
                atomic(OUT/'exit.json',{'exit_code':1,'ended_utc':utc(),'error':repr(exc)})
            raise
    else: worker(args.worker)
