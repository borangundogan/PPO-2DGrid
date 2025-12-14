# MERLIN: Modular PPO Baseline for MiniGrid and Meta-RL Research

MERLIN is a clean, extensible, and research-grade reinforcement learning framework built around a strong PPO baseline on MiniGrid. The project is designed to quantify task distribution shift, policy robustness, and fast adaptation, and to serve as a foundation for Meta-RL methods such as MAML and FOMAML. At its current stage, MERLIN provides a fully reproducible PPO benchmark, multi-seed experimental infrastructure, and task-distribution analysis tools.

---

## 1. Core Features

### PPO implementation (baseline – completed)
- PPO from scratch: clipped surrogate objective, GAE, entropy regularization, gradient norm clipping
- Shared actor–critic backbone with automatic architecture selection:
  - `MLPActorCritic` for flattened observations
  - `CNNActorCritic` for image observations
- Full optimization diagnostics: policy loss, value loss, entropy, KL divergence, clip fraction, gradient norm
- TensorBoard logging: `loss/*`, `diagnostics/*`, `stats/*`, `hist/*`, `fig/*`
- Serves as a strong and stable baseline (not a final solution)

### ScenarioCreator and task abstraction
- All environments are defined in one YAML file: `src/config/scenario.yaml`
- Difficulty levels: `easy`, `medium`, `hard`, `hardest`
- Fixed grid size across tasks (MERLIN_C2 constraint) and centrally controlled observation mode (partial vs full, flatten vs CNN)
- Ensures strict comparability between tasks
- Example:
  ```yaml
  difficulties:
    easy:
      env_id: "MERLIN-Easy-16x16-v0"
    medium:
      env_id: "MERLIN-Medium-16x16-v0"
    hard:
      env_id: "MERLIN-Hard-16x16-v0"
    hardest:
      env_id: "MERLIN-Hardest-16x16-v0"

  observation:
    fully_observable: false
    flatten: true
  ```

### Custom MiniGrid task families (`src/custom_envs/`)
| Difficulty | Description                     |
| ---------- | ------------------------------- |
| Easy       | Empty 16x16 grid                |
| Medium     | Random internal walls           |
| Hard       | Structured wall split + random goal |
| Hardest    | Four-rooms layout with obstacles |

These environments introduce increasing structural variation, not just reward difficulty. All tasks are deterministic and reproducible; a medium-hard variant is available for experiments.

### Three-action wrapper
- Minimal action space for stable PPO learning: `src/wrappers/three_action_wrapper.py`

  | ID | Action      |
  | -- | ----------- |
  | 0  | Turn left   |
  | 1  | Turn right  |
  | 2  | Move forward |

Applied uniformly to all environments.

---

## 2. Training and Experiment Design

### Multi-seed training (robustness-first)
- Each run uses multiple seeds to measure robustness rather than lucky runs
- Seeds control network init, environment stochasticity, rollout sampling, and optimization noise
- Checkpoint layout:
  ```
  checkpoints/
  └── <env>_<difficulty>_<timestamp>/
      ├── seed_123/
      │   └── ppo_model.pth
      ├── seed_7777/
      │   └── ppo_model.pth
      └── seed_658/
          └── ppo_model.pth
  ```
- TensorBoard logs: `tb_logs/<experiment>/<seed>/`

### Training entrypoint
```bash
uv run python train.py \
  --difficulty medium \
  --total_steps 100000 \
  --seed 123
```

### Evaluation
- Entrypoint: `test.py`
- CNN/MLP-aware evaluation and human render mode
- Example:
  ```bash
  uv run python test.py \
    --difficulty hard \
    --model_path checkpoints/.../seed_123/ppo_model.pth
  ```

---

## 3. Quantitative Task-Distribution Analysis

- Purpose: measure task similarity and generalization limits, not just average reward
- Tool: `src/analyze_tasks.py`
- What it does: evaluates a trained policy across multiple task families and collects 100+ episodes per task
- Metrics: mean normalized difference, KL divergence (P||Q and Q||P), Jensen–Shannon divergence, Wasserstein distance
- Outputs: KDE plots, mean/std bar charts under `analysis_results/<experiment_name>/`

---

## 4. Example Results (PPO Baseline)
| Train Env | Test Env | Avg Reward |
| --------- | -------- | ---------- |
| Easy      | Easy     | ~0.95      |
| Medium    | Medium   | ~0.93      |
| Medium    | Hard     | ~0.45      |
| Medium    | Hardest  | ~0.00      |

**Interpretation**
- PPO solves Easy and Medium reliably
- Performance degrades sharply on Hard and Hardest
- Reward distributions diverge strongly; KL/JS/Wasserstein distances increase significantly
- Confirms task distribution shift and motivates Meta-RL

---

## 5. Project Structure
```
analysis_results/
checkpoints/
tb_logs/
src/
├── analyze_tasks.py
├── actor_critic.py
├── config/
│   └── scenario.yaml
├── custom_envs/
├── metrics/
│   ├── ppo_metrics.py
│   └── task_metrics.py
├── scenario_creator/
│   └── scenario_creator.py
├── wrappers/
│   └── three_action_wrapper.py
├── utils.py
├── ppo.py
├── train.py
└── test.py
```

---

## 6. Getting Started
- Install dependencies: `uv pip sync`
- Train: `uv run python train.py --difficulty medium`
- View logs: `tensorboard --logdir tb_logs`
- Evaluate: `uv run python test.py --difficulty medium`
