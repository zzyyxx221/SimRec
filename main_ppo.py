from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_VERL_REPO_ROOT = Path(os.environ.get("VERL_REPO_ROOT", str(_THIS_DIR.parent / "verl"))).resolve()
if str(_VERL_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERL_REPO_ROOT))

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_module

from SimRec import tool_agent_loop as _tool_agent_loop  # noqa: F401
from SimRec.self_play import tool_agent_loop as _self_play_tool_agent_loop  # noqa: F401


def _is_self_play() -> bool:
    """Whether a rollout should generate shopper turns with the actor itself."""

    mode = os.environ.get("SIMREC_USER_SIMULATOR_MODE", "external").strip().lower()
    return mode in {"self_play", "self-play", "selfplay"}


def _collect_runtime_env_overrides() -> dict[str, str]:
    forwarded_keys = {
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_P2P_LEVEL",
        "NCCL_IB_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_RAS_ENABLE",
        "GLOO_SOCKET_IFNAME",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "TORCH_NCCL_BLOCKING_WAIT",
        "VLLM_USE_V1",
        "VERL_RUN_ID",
        "VERL_ZMQ_RUN_ID",
        "VERL_ZMQ_SOCKET_DIR",
        "WANDB_MODE",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO",
        "TRAIN_DATA",
        "VAL_DATA",
    }
    forwarded_prefixes = ("SEARCH_", "SIMREC_")
    return {
        key: value
        for key, value in os.environ.items()
        if key in forwarded_keys or key.startswith(forwarded_prefixes)
    }


@hydra.main(config_path="../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, task_runner_class=None) -> None:
    with open_dict(config):
        config.actor_rollout_ref.rollout.agent.default_agent_loop = (
            "simrec_self_play_tool_agent" if _is_self_play() else "simrec_tool_agent"
        )

    if not ray.is_initialized():
        ray_address = os.environ.get("RAY_ADDRESS")
        if ray_address:
            print(f"Initializing Ray with address={ray_address}")
        else:
            print("Initializing Ray with default local settings")
        ray_init_kwargs = OmegaConf.create(config.ray_kwargs.get("ray_init", {}))
        runtime_env = OmegaConf.merge(PPO_RAY_RUNTIME_ENV, ray_init_kwargs.get("runtime_env", {}))
        runtime_env.env_vars = OmegaConf.merge(
            OmegaConf.create(runtime_env.get("env_vars", {})),
            OmegaConf.create(_collect_runtime_env_overrides()),
        )
        ray_init_payload = {**ray_init_kwargs, "runtime_env": runtime_env}
        if ray_address:
            ray_init_payload["address"] = ray_address
        ray_init_kwargs = OmegaConf.create(ray_init_payload)
        ray.init(**OmegaConf.to_container(ray_init_kwargs, resolve=True))

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        nsight_options = OmegaConf.to_container(config.global_profiler.global_tool_config.nsys.controller_nsight_options)
        runner_cls = task_runner_class or TaskRunner
        runner = runner_cls.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner_cls = task_runner_class or TaskRunner
        runner = runner_cls.remote()
    ray.get(runner.run.remote(config))


class TaskRunnerBase:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def _requires_critic(self, config) -> bool:
        return need_critic(config)

    def _requires_ref_policy(self, config) -> bool:
        return need_reference_policy(config)

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            role = Role.ActorRolloutRef if self._requires_ref_policy(config) and not ref_in_actor else Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorker as CriticWorker
            else:
                from verl.workers.fsdp_workers import CriticWorker
        elif config.critic.strategy == "megatron":
            if use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorker as CriticWorker
            else:
                from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        from verl.single_controller.ray import ResourcePoolManager

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def add_ref_policy_worker(self, config, ref_policy_cls):
        if config.trainer.get("use_legacy_worker_impl", "auto") == "disable":
            return
        if self._requires_ref_policy(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def get_trainer_cls(self):
        return RayPPOTrainer

    def run(self, config):
        from pprint import pprint

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        if _is_self_play():
            import SimRec.self_play.reward_manager  # noqa: F401
            import SimRec.self_play.tool_agent_loop  # noqa: F401
        else:
            import SimRec.reward_manager  # noqa: F401
            import SimRec.tool_agent_loop  # noqa: F401

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
        processor = hf_processor(local_path, trust_remote_code=config.data.get("trust_remote_code", False), use_fast=True)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        if self._requires_critic(config):
            self.add_critic_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = self.get_trainer_cls()(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


TaskRunner = ray.remote(num_cpus=1)(TaskRunnerBase)


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        module = load_module(data_config.custom_cls.path)
        dataset_cls = getattr(module, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError("The custom dataset class must inherit from torch.utils.data.Dataset")
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
    else:
        dataset_cls = RLHFDataset
    if dataset_cls.__name__ == "SimRecDataset":
        return dataset_cls(
            data_files=data_paths,
            tokenizer=tokenizer,
            processor=processor,
            config=data_config,
            is_train=is_train,
        )
    return dataset_cls(data_files=data_paths, tokenizer=tokenizer, processor=processor, config=data_config)


def create_rl_sampler(data_config, dataset):
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        module = load_module(data_config.sampler.class_path)
        curriculum_class = getattr(module, data_config.sampler.class_name)
        sampler = curriculum_class(data_source=dataset, data_config=data_config)
        assert isinstance(sampler, AbstractSampler)
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed", 1)
        if seed is None:
            seed = 1
        train_dataloader_generator.manual_seed(int(seed))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)
    return sampler


if __name__ == "__main__":
    main()
