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

This module quantifies **how different** the evaluation tasks are from the training tasks. Instead of relying solely on scalar average rewards, we analyze the full probability distribution of returns to understand the policy's stability and failure modes.

- **Tool:** `src/analyze_tasks.py`
- **Methodology:**
  - Evaluates the trained policy on 100+ episodes for each task family.
  - Generates **High-Precision Normalized Reward Histograms** (Probability Mass Functions).
  - **Visual Update:** Unlike standard KDE plots, these histograms explicitly show the **"All-or-Nothing"** (bimodal) nature of the tasks with a strict Probability Y-axis ($\in [0, 1]$), avoiding density estimation artifacts.
- **Key Metrics:**
  - **Success Probability:** Explicit fraction of episodes where the agent reaches the goal.
  - **Wasserstein Distance:** Measures the physical "cost" to transform the reward distribution of Task A to Task B.
  - **KL Divergence:** Quantifies the information loss when the policy transfers from one environment to another.
- **Outputs:** Saved in `analysis_results/<experiment_name>/` as distribution plots and CSV reports.

---

## 4. Example Results (PPO Baseline)

The table and descriptions below illustrate the performance of a PPO baseline trained on "Medium" difficulty.

| Train Env | Test Env | Avg Reward | Stability Profile | Status |
| --------- | -------- | ---------- | ----------------- | ------ |
| Easy      | Easy     | ~0.95      | **Deterministic** (Single sharp peak at 1.0) | Solved |
| Medium    | Medium   | ~0.93      | **High** (Sharp peak) | Solved |
| Medium    | Hard     | ~0.45      | **Bimodal** (Split between 0.0 and 1.0) | Struggling |
| Medium    | Hardest  | ~0.00      | **Failure** (Peak at 0.0) | Failed |

**Visual Interpretation (Reward Distributions)**
The generated probability histograms reveal two distinct behaviors:
1.  **Deterministic Success (Easy/Medium):** The distribution is a narrow, high peak near Reward $\approx 1.0$. This indicates the agent is confident and consistent.
2.  **Stochastic Failure (Hard):** The distribution becomes **bimodal**. The agent either solves the task perfectly (Reward $\approx 0.9$) or fails completely (Reward $0.0$). This "gap" proves the policy has not learned a robust navigation strategy for unseen topologies.
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
