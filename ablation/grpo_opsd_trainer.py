# C3-OPD final experimental source overlay; deployment paths made configurable.
"""GRPO+OPSD ablation: A + lambda*(Teacher-old Student) - beta*invalid.
Trajectory GRPO, token masks, Teacher inputs and all numerical parameters match C3.
No verifier prompts, logits, credit weights or validity gates are computed.
"""
import json
import numpy as np
import torch
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer

class GRPOOPSDRayTrainer(RLSDRayTrainer):
    def __init__(self, *args, **kwargs):
        RLSDRayTrainer.__init__(self, *args, **kwargs)
        cfg = self.config.algorithm.c3_opsd
        assert cfg.get("verifier_enabled") is False
        self.lambda_init = float(cfg.lambda_opsd)
        self.lambda_floor = float(cfg.lambda_floor)
        self.warmdown_steps = int(cfg.warmdown_steps)
        self.invalid_action_beta = float(cfg.invalid_action_beta)
        assert self.invalid_action_beta == .1
        assert not self.config.actor_rollout_ref.actor.use_invalid_action_penalty
        from c3_teacher_only_scorer import ExternalScorerClient
        self.external_scorer = ExternalScorerClient(cfg)

    def _validate(self):
        def validate():
            if self.config.algorithm.c3_opsd.get("validation_probe_path"):
                from verl.trainer.ppo.c3_validation_probe import validate_fixed_probe
                return validate_fixed_probe(self)
            return super(GRPOOPSDRayTrainer, self)._validate()
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

    def _compute_teacher_log_probs(self, batch):
        import time
        from verl.trainer.ppo.c3_scoring_inputs import build_external_teacher_inputs
        begun = time.monotonic()
        messages, response_ids, positions = build_external_teacher_inputs(
            batch, self.skill_provider, self.tokenizer)
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
        if not torch.isfinite(teacher).all():
            raise ValueError("Non-finite external Teacher log-probs")
        assert result["model"]["verifier_enabled"] is False
        assert set(result["timings"]) <= {"teacher", "dispatch_wall"}
        assert not any(k in result for k in ("c", "v", "generated_labels"))
        print(f"[C3 external scoring] requests={len(positions)} total_s={time.monotonic()-begun:.3f} "
              f"details={result.get('timings')} replicas_batch_stats={result.get('replica_batch_stats',result.get('batch_stats'))} prompt_lengths={ {k: (min(v), max(v)) if v else (0,0) for k,v in result.get('prompt_lengths',{}).items()} }", flush=True)
        return teacher.detach()

    def _apply_grpo_opsd_advantages(self, batch):
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
        q_error = 0.0
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
        metrics = {"ablation/verifier_enabled": 0., "c3/q_mean": q.mean().item(), "c3/g_mean": gate.mean().item(),
                   "c3/lambda": self._lambda(), "c3/delta_mean": (delta * mask).sum().item() / mask.sum().clamp_min(1).item()}
        metrics.update({"c3/q_token_weighted_mean_max_error": q_error,
                        "c3/invalid_action_beta": self.invalid_action_beta,
                        "c3/invalid_turns": int(((invalid > 0) & (mask.sum(-1) > 0)).sum()),
                        "c3/invalid_response_tokens": float((invalid.unsqueeze(-1) * mask).sum()),
                        "c3/final_advantage_nonfinite_count": int((~torch.isfinite(fused)).sum())})
        metrics.update(self._credit_distribution_metrics(batch, q, gate))
        return metrics

    def _credit_distribution_metrics(self, batch, q, gate):
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
        for name, values in (("q", q), ("g", gate)):
            observed_indices=indices
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
        metrics = self._apply_grpo_opsd_advantages(batch)
        print("[GRPO+OPSD no verifier] " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()), flush=True)
        return metrics
