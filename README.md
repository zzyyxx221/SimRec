# SimRec

Reference implementation for **SimRec: Learning to Search, Ask, and Recommend from Simulated Interaction**.

SimRec trains a complete recommendation-agent policy with GRPO. The agent can search a catalog, inspect candidates, retrieve optional preference context, ask a clarification question, and revise a recommendation after feedback. It supports two task-anchored user simulators:

- `external`: an OpenAI-compatible LLM generates feedback from the task need. Target IDs are controller-only state and are removed before any simulator request.
- `self_play`: the rollout model alternates between private shopper and recommender roles. The shopper receives the task need, then identifies what a recommendation lacks without receiving the target item ID.

The reward combines valid tool use, natural retrieval quality, grounded recommendation format, recovery after feedback, and final hit. Training-only target injection can prevent sparse-hit rollouts from providing no learning signal; it is disabled for validation by default.

## Install

Use a Python environment with a CUDA-compatible PyTorch build, then install [verl](https://github.com/volcengine/verl) and the extra dependencies:

```bash
pip install -r SimRec/requirements.txt
git clone https://github.com/volcengine/verl.git
pip install -e ./verl
export VERL_REPO_ROOT=$PWD/verl
export PYTHONPATH=$PWD:$VERL_REPO_ROOT
```

The launcher expects the repository and `verl` to be sibling directories. Set `VERL_REPO_ROOT` explicitly if your layout differs.

## Data

No datasets, checkpoints, model weights, API keys, or indexes are included. Prepare JSONL records with at least:

```json
{"qid":"example-1","target_item_id":"ITEM_ID","reference_query":"I need a compact travel charger.","reference_review":"The target item supports fast USB-C charging."}
```

`target_item_ids` may be used instead of `target_item_id` for multi-target evaluation. Optional `user_preference` or `user_profile` fields enable the preference tool.

To transform an Amazon-C4-style export, run:

```bash
python -m SimRec.prepare_amazon_c4 --source /path/to/amazon_c4.csv --output_dir /path/to/output
```

Build the catalog index separately and set `SEARCH_INDEX_DIR` and `SEARCH_MODEL_PATH` in your local configuration. Input data and indexes are intentionally ignored by Git.

## Train

Copy `configs/train.env.example` to a location outside this repository, replace all `/path/to/...` values, and keep API credentials in the shell environment:

```bash
export SIMREC_USER_LLM_API_KEY=...
CONFIG_FILE=/secure/location/simrec.env bash SimRec/train_simrec.sh
```

Set `SIMREC_USER_SIMULATOR_MODE=self_play` to use self-play. This changes the dataset, reward manager, interaction, tools, and agent loop together; no external user-simulator endpoint is needed.

For evaluation, set `SIMREC_TRAIN_INJECT_TARGET_ON_MISS=false` and `SIMREC_VALIDATION_INJECT_TARGET_ON_MISS=false`. Use the same retrieval model, index, simulator settings, and task split for all compared checkpoints.

## Repository layout

- `dataset.py`, `tool_agent_loop.py`, `tools/`: external-simulator agent path.
- `self_play/`: shared-actor shopper/recommender path.
- `simulator.py`: target-hidden user feedback client.
- `reward_manager.py`: task-aligned GRPO reward and trajectory records.
- `configs/`: portable, credential-free examples.

## Privacy and release hygiene

This repository deliberately excludes local paths, user identifiers, raw datasets, models, retrieval indexes, checkpoints, logs, evaluation records, experiment trackers, and credentials. Before publishing, run the checks in the final section below and inspect `git status --ignored`.

```bash
python -m compileall -q SimRec
rg -n -i '/home/|/users/' SimRec --glob '!README.md'
git -C SimRec status --short --ignored
```

The `rg` command should return no results. Inspect configuration files before
every release and never commit a credential.
