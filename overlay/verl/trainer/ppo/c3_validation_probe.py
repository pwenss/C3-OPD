# C3-OPD final experimental source overlay; deployment paths made configurable.
"""Fixed question-level SearchQA probe; no reward penalty or turn reweighting."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from verl import DataProto


def _atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)


def validate_fixed_probe(trainer):
    cfg = trainer.config.algorithm.c3_opsd
    path = Path(cfg.validation_probe_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != cfg.validation_probe_sha256:
        raise ValueError("Fixed probe asset changed")
    probe = json.loads(raw)
    expected = probe["row_indices"]
    if len(expected) != 350 or len(set(expected)) != 350:
        raise ValueError("Expected the existing 350 unique-question probe")
    sampling = trainer.config.actor_rollout_ref.rollout.val_kwargs
    assert sampling.n == 1 and not sampling.do_sample and sampling.temperature == 0
    assert sampling.top_p == 1 and sampling.top_k == -1
    from agent_system.environments.env_package.search.third_party.skyrl_gym.tools import search as search_tool
    original_search = search_tool.call_search_api
    retrieval_errors = []

    def checked_search(*args, **kwargs):
        response, error = original_search(*args, **kwargs)
        if error:
            retrieval_errors.append(str(error))
        return response, error

    search_tool.call_search_api = checked_search
    records = []
    try:
        for chunk, test_data in enumerate(trainer.val_dataloader):
            batch = DataProto.from_single_dict(test_data)
            sources = list(batch.non_tensor_batch["data_source"])
            kwargs = list(batch.non_tensor_batch["env_kwargs"])
            ids = [int(item["probe_row_index"]) for item in kwargs]
            keys = ["raw_prompt_ids", "data_source"]
            for key in ("multi_modal_data", "raw_prompt", "tools_kwargs", "env_kwargs"):
                if key in batch.non_tensor_batch:
                    keys.append(key)
            generation = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                                   non_tensor_batch_keys=keys)
            generation.meta_info = {"eos_token_id": trainer.tokenizer.eos_token_id,
                                    "pad_token_id": trainer.tokenizer.pad_token_id,
                                    "recompute_log_prob": False,
                                    "do_sample": False, "validate": True}
            print(f"[C3 probe] step={trainer.global_steps} chunk={chunk} start questions={len(ids)}", flush=True)
            trajectories, rewards, lengths, success, _, tools = trainer.traj_collector.vanilla_multi_turn_loop(
                generation, trainer.actor_rollout_wg, trainer.val_envs)
            if retrieval_errors:
                raise RuntimeError("Retrieval errors invalidate probe: " + repr(retrieval_errors))
            assert len(success["success_rate"]) == len(ids) and np.isfinite(rewards).all()
            for row_id, source, reward, won, turns, calls, trajectory in zip(
                    ids, sources, rewards, success["success_rate"], lengths, tools, trajectories):
                correct = int(won)
                assert won in (0, 1) and float(reward) == correct
                active = [turn for turn in trajectory if turn["active_masks"]]
                records.append({"row_index": row_id, "data_source": str(source), "correct": correct,
                                "episode_reward": float(reward), "turns": int(turns), "tool_calls": int(calls),
                                "actions": [turn["student_action"] for turn in active],
                                "is_action_valid": [bool(turn["is_action_valid"]) for turn in active]})
            print(f"[C3 probe] step={trainer.global_steps} chunk={chunk} finished", flush=True)
    finally:
        search_tool.call_search_api = original_search
    assert [record["row_index"] for record in records] == expected
    subtasks = {}
    counts = collections.Counter(question["data_source"] for question in probe["questions"])
    assert len(counts) == 7 and all(count == 50 for count in counts.values())
    for source in counts:
        selected = [record for record in records if record["data_source"] == source]
        assert len(selected) == 50
        correct = sum(record["correct"] for record in selected)
        subtasks[source] = {"correct": correct, "evaluated": 50, "accuracy": correct / 50}
    total_correct = sum(record["correct"] for record in records)
    macro = sum(task["accuracy"] for task in subtasks.values()) / 7
    micro = total_correct / 350
    assert np.isfinite([macro, micro]).all() and abs(macro - micro) < 1e-12
    result = {"status": "complete", "global_step": int(trainer.global_steps),
              "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "generation_strategy": "greedy", "probe_sha256": cfg.validation_probe_sha256,
              "metric": "last active native won/episode_reward; normalized EM; one episode per question",
              "parser": "thinking-isolated action projection", "evaluated": 350, "correct": total_correct,
              "subtasks": subtasks, "macro_accuracy": macro, "overall_accuracy": micro, "questions": records}
    output = Path(cfg.validation_probe_result_dir) / f"global_step_{trainer.global_steps}.json"
    _atomic(output, result)
    metrics = {f"probe/{source}/accuracy": value["accuracy"] for source, value in subtasks.items()}
    metrics.update({"probe/macro_accuracy": macro, "probe/overall_accuracy": micro,
                    "probe/evaluated": 350, "probe/correct": total_correct})
    print("[C3 probe complete] " + json.dumps({"step": trainer.global_steps, **metrics}, allow_nan=False), flush=True)
    return metrics
