# C3-OPD final experimental source overlay; deployment paths made configurable.
"""C3-OPSD trainer using SDAR rollouts and fixed privileged teacher and verifier signals."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer


class C3OPSDRayTrainer(RLSDRayTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.algorithm.get("c3_opsd", {})
        self.q_temperature = float(cfg.get("q_temperature", 1.0))
        self.q_rho = float(cfg.get("q_rho", 0.25))
        self.lambda_init = float(cfg.get("lambda_opsd", 0.10))
        self.lambda_floor = float(cfg.get("lambda_floor", 0.02))
        self.warmdown_steps = int(cfg.get("warmdown_steps", 100))
        self.invalid_action_beta = float(cfg.get("invalid_action_beta", 0.0))
        if not np.isfinite(self.invalid_action_beta) or self.invalid_action_beta < 0:
            raise ValueError("C3 invalid_action_beta must be finite and nonnegative")
        if self.invalid_action_beta > 0 and self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
            raise ValueError("Direct C3 invalid penalty requires reward-level invalid penalty disabled")
        if not np.isfinite(self.q_temperature) or self.q_temperature <= 0:
            raise ValueError("C3 q_temperature must be positive and finite")
        self.verifier_batch_size = int(cfg.get("verifier_batch_size", 16))
        self._label_ids = None
        from pathlib import Path
        from verl.workers.c3_external_scorer import ExternalScorerClient
        template_path = cfg.get("verifier_templates_path",
            str(Path(__file__).resolve().parents[3] / "agent_system/prompts/c3_verifier_templates_search.json"))
        self.verifier_templates = json.loads(Path(template_path).read_text())
        self.external_scorer = ExternalScorerClient(cfg)

    def _validate(self):
        def validate():
            if self.config.algorithm.c3_opsd.get("validation_probe_path"):
                from verl.trainer.ppo.c3_validation_probe import validate_fixed_probe
                return validate_fixed_probe(self)
            return super(C3OPSDRayTrainer, self)._validate()
        if not self.config.actor_rollout_ref.rollout.get("c3_keep_eval_awake", False):
            return validate()
        # Evaluation neither updates the Student nor calls the Teacher scorer.
        if self.ref_policy_wg is not None:
            self.ref_policy_wg.release_for_external_scoring()
        self.actor_rollout_wg.begin_validation_residency()
        try:
            return validate()
        finally:
            self.actor_rollout_wg.end_validation_residency()

    def _lambda(self):
        if self.warmdown_steps <= 0:
            return self.lambda_floor
        frac = min(max(self.global_steps, 0) / self.warmdown_steps, 1.0)
        return self.lambda_init + frac * (self.lambda_floor - self.lambda_init)

    def _get_label_ids(self):
        if self._label_ids is None:
            ids_a = self.tokenizer.encode("A", add_special_tokens=False)
            ids_b = self.tokenizer.encode("B", add_special_tokens=False)
            if len(ids_a) != 1 or len(ids_b) != 1:
                raise ValueError(f"Verifier labels must be single tokens: A={ids_a}, B={ids_b}")
            self._label_ids = (ids_a[0], ids_b[0])
        return self._label_ids

    @staticmethod
    def _decode_valid(tokenizer, ids, mask):
        valid = ids[mask.bool()]
        return tokenizer.decode(valid, skip_special_tokens=True)

    @staticmethod
    def _task_text(raw):
        if isinstance(raw, dict):
            return str(raw.get("content", ""))
        if isinstance(raw, str):
            return raw
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if isinstance(raw, (list, tuple)):
            return "\n".join(C3OPSDRayTrainer._task_text(item) for item in raw)
        return ""

    def _verifier_prompts(self, batch):
        from verl.trainer.ppo.c3_scoring_inputs import build_verifier_messages
        return build_verifier_messages(self, batch, self.verifier_templates)

    def _build_verifier_batch(self, prompts):
        return {"local_messages": [item[0] for item in prompts],
                "valid_messages": [item[1] for item in prompts]}

    @staticmethod
    def _local_scoring_required(non_tensors):
        """Skip constant binary episode rewards only; their FP32 normalized advantage is exactly zero."""
        tasks=non_tensors["uid"];rewards=non_tensors["episode_rewards"]
        values={}
        for task,reward in zip(tasks,rewards):
            values.setdefault(str(task),set()).add(float(reward))
        return [len(values[str(task)])>1 or next(iter(values[str(task)])) not in (0.0,1.0) for task in tasks]

    def _compute_teacher_log_probs(self, batch):
        import time
        from verl.trainer.ppo.c3_scoring_inputs import build_external_teacher_inputs
        begun = time.monotonic()
        messages, response_ids, positions = build_external_teacher_inputs(
            batch, self.skill_provider, self.tokenizer)
        verifier_inputs = self._build_verifier_batch(self._verifier_prompts(batch))
        if self.config.algorithm.c3_opsd.get("skip_equal_reward_local",False):
            verifier_inputs["local_required"]=self._local_scoring_required(batch.non_tensor_batch)
        colocated = self.config.algorithm.c3_opsd.get("external_colocated", False)
        if colocated:
            self.actor_rollout_wg.release_for_external_scoring()
            if self.use_reference_policy and not self.ref_in_actor:
                self.ref_policy_wg.release_for_external_scoring()
            self.external_scorer.set_sleeping(False)
        try:
            result = self.external_scorer.score({
                "teacher_messages": messages,
                "teacher_response_ids": response_ids,
                **verifier_inputs,
                "temperature": float(self.config.actor_rollout_ref.rollout.temperature)})
        finally:
            if colocated:
                self.external_scorer.set_sleeping(True)
        teacher = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        for i, (indices, values) in enumerate(zip(positions, result["teacher_log_probs"])):
            if len(indices) != len(values):
                raise ValueError("External teacher returned misaligned response log-probs")
            teacher[i, indices] = torch.tensor(values, dtype=torch.float32, device=teacher.device)
        if len(result["teacher_log_probs"]) != len(positions):
            raise ValueError("External teacher returned an incorrect batch size")
        self._pending_c = torch.tensor(result["c"], dtype=torch.float32, device=teacher.device)
        self._pending_v = torch.tensor(result["v"], dtype=torch.float32, device=teacher.device)
        self._pending_local_scored = torch.tensor(result.get("local_scored",[True]*len(positions)),dtype=torch.bool,device=teacher.device)
        if self._pending_c.numel() != len(positions) or self._pending_v.numel() != len(positions):
            raise ValueError("External verifier returned an incorrect batch size")
        if not all(torch.isfinite(t).all() for t in (teacher, self._pending_c, self._pending_v)):
            raise ValueError("Non-finite external scoring result")
        print(f"[C3 external scoring] requests={len(positions)} total_s={time.monotonic()-begun:.3f} "
              f"details={result.get('timings')} replicas_batch_stats={result.get('replica_batch_stats',result.get('batch_stats'))} prompt_lengths={ {k: (min(v), max(v)) if v else (0,0) for k,v in result.get('prompt_lengths',{}).items()} }", flush=True)
        capture_path = self.config.trainer.get("speed_validation_fixture_path", None)
        if capture_path and self.global_steps == 1:
            tasks = batch.non_tensor_batch["uid"]
            rewards = batch.non_tensor_batch["episode_rewards"]
            unique_tasks = list(dict.fromkeys(str(x) for x in tasks))
            reward_range = {u: np.ptp([float(rewards[i]) for i in range(len(tasks)) if str(tasks[i]) == u]) for u in unique_tasks}
            chosen = [max(unique_tasks, key=lambda u: reward_range[u])]
            if len(unique_tasks) > 1:
                chosen.append(next((u for u in unique_tasks if u != chosen[0] and reward_range[u] == 0), next(u for u in unique_tasks if u != chosen[0])))
            selected = [i for i in range(len(tasks)) if str(tasks[i]) in chosen]
            self._speed_validation_capture = dict(path=str(capture_path),indices=selected,tasks=chosen,
                reward_range=reward_range,payload={k:[v[i] for i in selected] for k,v in dict(teacher_messages=messages,
                teacher_response_ids=response_ids,**verifier_inputs).items()},
                c=[result["c"][i] for i in selected],v=[result["v"][i] for i in selected],
                global_step=int(self.global_steps),lambda_opsd=self._lambda())
        return teacher.detach()

    def _apply_c3_advantages(self, batch, c_vals, v_vals):
        mask = batch.batch["response_mask"].float()
        seq_adv = batch.batch["advantages"]
        bs = len(mask)
        turns = batch.non_tensor_batch.get("turn_step", np.zeros(bs, dtype=object))
        trajs = batch.non_tensor_batch.get("traj_uid", np.arange(bs, dtype=object))
        rewards = batch.non_tensor_batch.get("episode_rewards", np.zeros(bs, dtype=object))
        task_uids = batch.non_tensor_batch.get("uid", np.arange(bs, dtype=object))
        groups = {}
        task_to_trajs = {}
        traj_to_reward = {}
        for i in range(bs):
            traj = str(trajs[i])
            task = str(task_uids[i])
            groups.setdefault(traj, []).append(i)
            task_to_trajs.setdefault(task, set()).add(traj)
            traj_to_reward[traj] = float(rewards[i])
        uid_adv = {}
        for traj_set in task_to_trajs.values():
            ug = sorted(traj_set)
            vals = torch.tensor([traj_to_reward[u] for u in ug], device=mask.device, dtype=torch.float32)
            norm = (vals - vals.mean()) / (vals.std(unbiased=False) + 1e-6)
            uid_adv.update(dict(zip(ug, norm)))
        q = torch.ones(bs, device=mask.device)
        gate = torch.ones(bs, device=mask.device)
        q_mean_errors = []
        if not all(torch.isfinite(values).all() for values in (c_vals, v_vals)):
            raise ValueError("Non-finite C3 verifier scores")
        for uid, idxs in groups.items():
            idxs.sort(key=lambda k: int(turns[k]))
            ai = uid_adv.get(uid)
            if ai is None:
                row_mask = mask[idxs]
                ai = ((seq_adv[idxs] * row_mask).sum() / row_mask.sum().clamp_min(1.0)).detach()
            # L_{i,j} is the actual number of response tokens, including zero
            # for DP padding rows. No prompt/thinking tokens are removed.
            weights = mask[idxs].sum(-1)
            ci = c_vals[idxs].to(mask.device)
            denom = weights.sum().clamp_min(1.0)
            cbar = (ci * weights).sum() / denom
            hi = torch.tanh(torch.sign(ai) * (ci - cbar) / self.q_temperature)
            hbar = (hi * weights).sum() / denom
            q[idxs] = 1.0 + self.q_rho * (hi - hbar)
            if weights.sum() > 0:
                q_mean_errors.append(torch.abs((q[idxs] * weights).sum() / denom - 1.0))
            prev_v = 1.0
            for idx in idxs:
                gate[idx] = prev_v
                prev_v = float(v_vals[idx].detach())
        if seq_adv.ndim == 2:
            a = torch.zeros(bs, device=mask.device, dtype=seq_adv.dtype)
            for uid, idxs in groups.items():
                ai = uid_adv.get(uid)
                if ai is None:
                    row_mask = mask[idxs]
                    ai = ((seq_adv[idxs] * row_mask).sum() / row_mask.sum().clamp_min(1.0)).detach()
                a[idxs] = ai
            a = a.unsqueeze(-1)
        else:
            a = seq_adv
        token_q = q.unsqueeze(-1)
        token_gate = gate.unsqueeze(-1)
        delta = (batch.batch["teacher_log_probs"] - batch.batch["old_log_probs"]).detach()
        if not torch.isfinite(q).all() or not (q > 0).all():
            raise ValueError("C3 turn q must be finite and strictly positive")
        q_error = torch.stack(q_mean_errors).max().item() if q_mean_errors else 0.0
        if q_error > 2e-6:
            raise ValueError(f"C3 trajectory token-weighted q mean differs from one: {q_error}")
        invalid = torch.zeros(bs, device=mask.device, dtype=torch.float32)
        if self.invalid_action_beta > 0:
            valid = np.asarray(batch.non_tensor_batch["is_action_valid"])
            if valid.size != bs or not np.isin(valid, [0, 1]).all():
                raise ValueError("Invalid or misaligned parser action validity flags")
            invalid = 1.0 - torch.as_tensor(valid.reshape(bs).astype(np.float32), device=mask.device)
        # A_i is unchanged trajectory GRPO. lambda uses global step s. The
        # independent beta penalty is applied after both terms, before masking.
        fused = (token_q * a + self._lambda() * token_gate * delta
                 - self.invalid_action_beta * invalid.unsqueeze(-1)) * mask
        if not torch.isfinite(fused).all():
            raise ValueError("Non-finite final C3 token advantage")
        batch.batch["advantages"] = fused
        metrics = {"c3/c_mean":c_vals.mean().item(), "c3/v_mean": v_vals.mean().item(),
                   "c3/q_mean": q.mean().item(), "c3/g_mean": gate.mean().item(),
                   "c3/lambda": self._lambda(), "c3/delta_mean": (delta * mask).sum().item() / mask.sum().clamp_min(1).item()}
        metrics.update({"c3/q_token_weighted_mean_max_error": q_error,
                        "c3/invalid_action_beta": self.invalid_action_beta,
                        "c3/invalid_turns": int(((invalid > 0) & (mask.sum(-1) > 0)).sum()),
                        "c3/invalid_response_tokens": float((invalid.unsqueeze(-1) * mask).sum()),
                        "c3/final_advantage_nonfinite_count": int((~torch.isfinite(fused)).sum())})
        local_scored=getattr(self,"_pending_local_scored",None)
        if local_scored is not None:
            metrics["c3/local_scored_count"]=int(local_scored.sum())
            metrics["c3/local_skipped_count"]=int((~local_scored).sum())
            if local_scored.any():metrics["c3/c_mean"]=c_vals[local_scored].mean().item()
            else:metrics.pop("c3/c_mean",None)
        metrics.update(self._credit_distribution_metrics(batch, q, gate, c_vals, v_vals))
        return metrics

    def _credit_distribution_metrics(self, batch, q, gate, c_vals, v_vals):
        """Describe actual turn weights without counting DP padding copies twice."""
        lengths = batch.batch["response_mask"].sum(-1).detach().cpu().numpy()
        n = len(lengths)
        trajs = batch.non_tensor_batch.get("traj_uid", np.arange(n, dtype=object))
        turns = batch.non_tensor_batch.get("turn_step", np.zeros(n, dtype=object))
        tasks = batch.non_tensor_batch.get("uid", np.arange(n, dtype=object))
        seen, selected = set(), []
        for i in range(n):
            key = (str(tasks[i]), str(trajs[i]), int(turns[i]))
            if lengths[i] > 0 and key not in seen:
                selected.append(i)
                seen.add(key)
        indices = np.asarray(selected, dtype=np.int64)
        first = np.zeros(len(indices), dtype=bool)
        trajectory_first = {}
        for j, i in enumerate(selected):
            key = (str(tasks[i]), str(trajs[i]))
            old = trajectory_first.get(key)
            if old is None or int(turns[i]) < int(turns[selected[old]]):
                trajectory_first[key] = j
        for j in trajectory_first.values():
            first[j] = True
        report = {"step": int(self.global_steps), "batch_rows": n,
                  "valid_rows": int((lengths > 0).sum()), "unique_turns": len(selected),
                  "trajectory_first_turns": int(first.sum())}
        metrics = {"c3/distribution_unique_turns": float(len(selected))}
        for name, values in (("q", q), ("g", gate), ("c", c_vals), ("v", v_vals)):
            observed_indices=indices
            if name=="c" and hasattr(self,"_pending_local_scored"):
                observed_indices=indices[self._pending_local_scored.detach().cpu().numpy()[indices]]
            x = values.detach().float().cpu().numpy()[observed_indices].astype(np.float64)
            finite = np.isfinite(x)
            xf = x[finite]
            stats = {"count": int(x.size), "nonfinite_count": int((~finite).sum())}
            if xf.size:
                stats.update(min=float(xf.min()), max=float(xf.max()),
                             mean=float(xf.mean()), std=float(xf.std()))
                for label, level in (("p01", .01), ("p05", .05), ("p25", .25),
                                     ("p50", .5), ("p75", .75), ("p95", .95), ("p99", .99)):
                    stats[label] = float(np.quantile(xf, level))
                if name == "q":
                    stats["fraction_near_one_1e-6"] = float((np.abs(xf - 1) <= 1e-6).mean())
                    cuts = np.asarray([.9, .99, .999, 1.001, 1.01, 1.1])
                else:
                    cuts = np.asarray([.01, .1, .5, .9, .99, .999])
                stats["bin_upper_bounds"] = cuts.tolist()
                stats["bin_counts"] = np.bincount(np.searchsorted(cuts, xf, side="right"),
                                                  minlength=len(cuts) + 1).tolist()
                if name == "g":
                    nonfirst = x[(~first) & finite]
                    stats["nonfirst_count"] = int(nonfirst.size)
                    if nonfirst.size:
                        stats["nonfirst_mean"] = float(nonfirst.mean())
                        stats["nonfirst_p05"] = float(np.quantile(nonfirst, .05))
                        stats["nonfirst_p50"] = float(np.quantile(nonfirst, .5))
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        metrics[f"c3/{name}_dist_{key}"] = float(value)
            report[name] = stats
        print("[C3 distribution] " + json.dumps(report, allow_nan=False, separators=(",", ":")), flush=True)
        return metrics

    def _postprocess_advantages(self, batch):
        if not hasattr(self, "_pending_c"):
            raise RuntimeError("Teacher/verifier signals are missing for the current rollout batch")
        metrics = self._apply_c3_advantages(batch, self._pending_c, self._pending_v)
        if hasattr(self, "_speed_validation_capture"):
            from pathlib import Path
            from omegaconf import OmegaConf
            capture = self._speed_validation_capture
            selected = capture.pop("indices")
            capture["tensors"] = {k:v[selected].detach().cpu().clone() for k,v in batch.batch.items()}
            capture["non_tensors"] = {k:v[selected] for k,v in batch.non_tensor_batch.items()}
            capture["config"] = OmegaConf.to_container(self.config,resolve=True)
            path = Path(capture.pop("path")).resolve()
            root = Path(os.environ.get("C3_PROJECT_ROOT", os.getcwd())).resolve()
            if root not in path.parents:
                raise ValueError("Speed validation fixture must stay inside training workspace")
            torch.save(capture,path)
            print(f"[C3 speed validation] saved complete Task groups: {path}, rows={len(selected)}",flush=True)
            del self._speed_validation_capture
        del self._pending_c, self._pending_v, self._pending_local_scored
        print("[C3-OPSD] " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()))
        return metrics
