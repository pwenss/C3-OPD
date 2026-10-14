# C3-OPD final experimental source overlay; deployment paths made configurable.
"""C3-OPSD training entry point built on SDAR's environment and PPO stack."""

import os
import sys

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        ray_kwargs = config.get("ray_init", {})
        runtime_env = OmegaConf.merge(get_ppo_ray_runtime_env(), ray_kwargs.get("runtime_env", {}))
        # Ray 2.58 builds a shell command for workers. The configured venv
        # lives below a directory containing a space, so use the venv's
        # PATH-resolved `python` name instead of the absolute executable path.
        runtime_env["working_dir"] = None
        env_vars = dict(OmegaConf.to_container(runtime_env.get("env_vars", {}), resolve=True))
        venv_bin = os.path.dirname(sys.executable)
        env_vars["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
        runtime_env["env_vars"] = env_vars
        runtime_env["py_executable"] = "python"
        ray_kwargs = OmegaConf.create({**ray_kwargs, "runtime_env": runtime_env})
        ray_kwargs["include_dashboard"] = False
        ray_kwargs["_temp_dir"] = os.environ.get("C3_RAY_TMPDIR", os.path.join(os.getcwd(), "ray_tmp"))
        raw_system_config = ray_kwargs.get("_system_config", {})
        system_config = dict(OmegaConf.to_container(raw_system_config, resolve=True)) if OmegaConf.is_config(raw_system_config) else dict(raw_system_config or {})
        system_config.setdefault(
            "agent_register_timeout_ms",
            int(os.environ.get("C3_RAY_AGENT_REGISTER_TIMEOUT_MS", "120000")),
        )
        ray_kwargs["_system_config"] = system_config
        ray.init(**OmegaConf.to_container(ray_kwargs))
    ray.get(C3OPSDTaskRunner.remote().run.remote(config))


@ray.remote(num_cpus=1)
class C3OPSDTaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        local_path = copy_to_local(config.actor_rollout_ref.model.path,
                                   use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from agent_system.environments import make_envs
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
        processor = hf_processor(local_path, trust_remote_code=config.data.get("trust_remote_code", False), use_fast=True)
        envs, val_envs = make_envs(config)

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            actor_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            actor_cls = ActorRolloutRefWorker
            worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(config.actor_rollout_ref.actor.strategy)

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        role_mapping = {Role.ActorRollout: ray.remote(actor_cls), Role.Critic: ray.remote(CriticWorker)}
        pool = {"global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: "global_pool", Role.Critic: "global_pool"}
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_mapping[Role.RefPolicy] = ray.remote(actor_cls)
            mapping[Role.RefPolicy] = "global_pool"

        from agent_system.reward_manager import EpisodeRewardManager
        reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
        from agent_system.multi_turn_rollout import TrajectoryCollector
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.ppo.rlsd_utils import SkillProvider

        collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        train_data = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_data = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        sampler = create_rl_sampler(config.data, train_data)
        c3_cfg = config.algorithm.get("c3_opsd", {})
        skill_dir = c3_cfg.get("skills_dir", "skills/alfworld")
        skill_provider = SkillProvider(skill_dir, skill_all=c3_cfg.get("skill_all", False))
        print(f"[C3-OPSD] Skill files: {list(skill_provider.skill_contents)}")

        from verl.trainer.ppo.c3_opsd_ray_trainer import C3OPSDRayTrainer
        trainer = C3OPSDRayTrainer(
            config=config, tokenizer=tokenizer, processor=processor,
            role_worker_mapping=role_mapping,
            resource_pool_manager=ResourcePoolManager(pool, mapping),
            ray_worker_group_cls=worker_group_cls,
            reward_fn=reward_fn, val_reward_fn=val_reward_fn,
            train_dataset=train_data, val_dataset=val_data, collate_fn=collate_fn,
            train_sampler=sampler, device_name=config.trainer.device,
            traj_collector=collector, envs=envs, val_envs=val_envs,
            skill_provider=skill_provider,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
