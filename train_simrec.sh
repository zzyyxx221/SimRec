#!/usr/bin/env bash
# Launch a SimRec GRPO run.  All machine-specific paths belong in CONFIG_FILE.
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(dirname "$project_dir")
config_file=${CONFIG_FILE:-"$project_dir/configs/train.env"}

if [[ ! -f "$config_file" ]]; then
  echo "Missing config: $config_file (copy configs/train.env.example outside this repository)." >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$config_file"
set +a

for variable in MODEL_NAME TRAIN_DATA VAL_DATA SEARCH_INDEX_DIR SEARCH_MODEL_PATH; do
  if [[ -z "${!variable:-}" ]]; then
    echo "$variable must be set in $config_file" >&2
    exit 2
  fi
done

if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" ]]; then
  echo "TRAIN_DATA and VAL_DATA must point to existing JSONL files." >&2
  exit 2
fi

if [[ -n "${GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=$GPUS
fi
gpu_list=${CUDA_VISIBLE_DEVICES:-0}
gpu_count=$(awk -F, '{print NF}' <<<"${gpu_list// /}")
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$gpu_count}
export VERL_REPO_ROOT=${VERL_REPO_ROOT:-"$workspace_dir/verl"}
export PYTHONPATH="$workspace_dir:$VERL_REPO_ROOT:${PYTHONPATH:-}"

mode=${SIMREC_USER_SIMULATOR_MODE:-external}
case "$mode" in
  self_play|self-play|selfplay)
    dataset_path=SimRec/self_play/dataset.py
    reward_path=SimRec/self_play/reward_manager.py
    tool_config=${TOOL_CONFIG_FILE:-"$project_dir/configs/tool_config/self_play_tools.yaml"}
    interaction_config=${INTERACTION_CONFIG_FILE:-"$project_dir/configs/interaction_config/self_play_user.yaml"}
    agent_loop=simrec_self_play_tool_agent
    ;;
  external|api|llm)
    dataset_path=SimRec/dataset.py
    reward_path=SimRec/reward_manager.py
    tool_config=${TOOL_CONFIG_FILE:-"$project_dir/configs/tool_config/external_tools.yaml"}
    interaction_config=${INTERACTION_CONFIG_FILE:-"$project_dir/configs/interaction_config/external_user.yaml"}
    agent_loop=simrec_tool_agent
    ;;
  *)
    echo "SIMREC_USER_SIMULATOR_MODE must be external or self_play." >&2
    exit 2
    ;;
esac

python_bin=${PYTHON_BIN:-python3}
"$python_bin" -m SimRec.main_ppo \
  algorithm.adv_estimator=${RL_ALG:-grpo} \
  data.train_files="$TRAIN_DATA" data.val_files="$VAL_DATA" \
  data.train_batch_size=${TRAIN_BATCH_SIZE:-4} data.val_batch_size=${VAL_BATCH_SIZE:-4} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH:-2048} data.max_response_length=${MAX_RESPONSE_LENGTH:-1024} \
  data.custom_cls.path="$dataset_path" data.custom_cls.name=SimRecDataset \
  reward.reward_manager.source=importlib reward.reward_manager.name=SimRecRewardManager \
  reward.reward_manager.module.path="$reward_path" reward.reward_manager.module.name=SimRecRewardManager \
  actor_rollout_ref.model.path="$MODEL_NAME" critic.model.path="$MODEL_NAME" \
  actor_rollout_ref.actor.optim.lr=${LR:-1e-6} critic.optim.lr=${CRITIC_LR:-5e-6} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1} \
  critic.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1} \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=${ROLLOUT_N:-4} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.4} \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS:-4} \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$tool_config" \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path="$interaction_config" \
  actor_rollout_ref.rollout.agent.default_agent_loop="$agent_loop" \
  trainer.project_name=SimRec trainer.experiment_name=${RUN_NAME:-simrec} \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" trainer.nnodes=${N_NODES:-1} \
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-200} \
  trainer.save_freq=${SAVE_FREQ:-50} trainer.test_freq=${TEST_FREQ:-50} \
  trainer.logger="['console']" \
  ${HYDRA_EXTRA_OVERRIDES:-}
