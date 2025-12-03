# PPO: Modular PPO Pipeline for MiniGrid and Meta-RL Research

This repository provides a clean, extensible, and research-grade PPO implementation designed for MiniGrid, curriculum learning, and Meta-RL research. The pipeline covers PPO training, cross-difficulty evaluation, YAML-driven environment creation, and quantitative task-distribution analysis.

It includes a full PPO algorithm, custom MiniGrid tasks, a YAML scenario builder, quantitative task-distribution analysis tools, TensorBoard-ready logging, and cross-difficulty generalization evaluation.

## Features
1. **PPO implementation (from scratch)**
   - Clipped surrogate objective with GAE
   - Auto-switch MLP/CNN depending on observation shape
   - Gradient clipping and shared actor–critic backbone
   - Full optimization diagnostics: KL divergence, clip fraction, entropy decay, policy loss, value loss, gradient norm
   - Logged to TensorBoard under `loss/*`, `diagnostics/*`, `stats/*`, `hist/*`, `fig/*`

2. **ScenarioCreator framework**
   - All environments come from one YAML file: `src/config/scenario.yaml`
   - Defines difficulty levels (easy, medium, hard, hardest) and observation mode (partial/full, flatten vs CNN)
   - Validates fixed-size grids across tasks
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

3. **Custom MiniGrid environments (`src/custom_envs/`)**
   - Easy: empty 16×16 grid
   - Medium: random internal walls
   - Hard: vertical wall split plus random goal
   - Hardest: four-rooms fully connected maze with random obstacles
   - Medium-hard variant available for experiments
   - All tasks are deterministic and reproducible

4. **Three-action wrapper**
   - Minimal action set for faster PPO learning (`src/wrappers/three_action_wrapper.py`)

     | ID | Action       |
     | -- | ------------ |
     | 0  | turn left    |
     | 1  | turn right   |
     | 2  | move forward |

   - Applies to all environments

5. **Modular actor–critic networks**
   - `src/actor_critic.py`
   - `MLPActorCritic` for flattened observations, `CNNActorCritic` for image observations (H, W, C)
   - Automatic uint8 to float normalization and automatic shape detection at PPO init

6. **Logging and metrics system**
   - TensorBoard (via `train.py`) logs:
     - `loss/*`: policy loss, value loss, entropy
     - `diagnostics/*`: KL, clip fraction, gradient norm
     - `stats/*`: episodic return/length averages
     - `hist/*`: histograms of returns and episode lengths
     - `fig/reward_vs_steps`: dynamic scatter figure
   - PPO metrics module: `src/metrics/ppo_metrics.py` encapsulates PPO update diagnostics into reusable aggregator functions

7. **Training pipeline**
   - Entrypoint: `train.py`
   - Timestamped run IDs
   - Checkpoints: `checkpoints/<env>_<difficulty>_<timestamp>/ppo_model.pth`
   - TensorBoard logs: `tb_logs/<run_id>/`
   - Evaluation after each PPO update and visual evaluation mode
   - Example:
     ```bash
     uv run python train.py \
       --difficulty medium \
       --total_steps 500000 \
       --batch_size 4096 \
       --minibatch_size 512 \
       --eval_episodes 5
     ```

8. **Evaluation pipeline**
   - Entrypoint: `test.py`
   - Auto-loads latest checkpoint, supports manual override
   - Cross-difficulty evaluation with CNN/MLP-aware preprocessing
   - Human render mode
   - Example:
     ```bash
     uv run python test.py --difficulty hard
     ```

   - Testing a medium-trained PPO on the hard environment:
     ```bash
     uv run python test.py \
       --difficulty hard \
       --model_path checkpoints/MERLIN-Medium-16x16-v0_medium_xxxxxx/ppo_model.pth
     ```

9. **Quantitative task-distribution analysis (new)**
   - Entrypoint: `src/analyze_tasks.py`
   - Purpose: measure task distribution shift, understand PPO generalization limits, and compare Easy/Medium/Hard/Hardest statistically
   - Features extracted: PPO logits or state embeddings
   - Metrics computed (per task-pair): mean norm difference, KL divergence (P || Q), KL divergence (Q || P), Jensen–Shannon divergence, Wasserstein distance (1-D mean over features)
   - Visualizations: KDE distribution plots and mean activation comparison plots saved under `analysis_results/<checkpoint_name>/`
   - Example:
     ```bash
     uv run python -m src.analyze_tasks \
       --model_path checkpoints/MERLIN-Medium-16x16-v0_medium_20251120_194100/ppo_model.pth \
       --num_steps 3000 \
       --difficulties easy medium hard
     ```

## Project structure
```
analysis_results/
checkpoints/
src/
├── analyze_tasks.py
├── actor_critic.py
├── config/
│   └── scenario.yaml
├── custom_envs/
│   ├── easy_env.py
│   ├── medium_env.py
│   ├── medium_hard_env.py
│   ├── hard_env.py
│   ├── hardest_env.py
│   └── register.py
├── metrics/
│   ├── ppo_metrics.py
│   └── task_metrics.py
├── rollout_buffer.py
├── scenario_creator/
│   └── scenario_creator.py
├── utils.py
├── wrappers/
│   └── three_action_wrapper.py
├── ppo.py
├── test.py
└── train.py
```

## Example results
| Train Env | Test Env | Avg Reward |
| --------- | -------- | ---------- |
| Easy      | Easy     | 0.94–0.96  |
| Medium    | Medium   | 0.92–0.95  |
| Medium    | Hard     | ~0.45      |
| Medium    | Hardest  | Fails      |

**Interpretation**
- PPO learns Easy and Medium extremely well.
- Hard and Hardest show distribution shift.
- Quantitative metrics (KL, JS, Wasserstein) rise sharply, and mean activations diverge.
- This gap is where Meta-RL approaches (MAML, PEARL, RL², etc.) become necessary.

## Getting started
- Install dependencies: `uv pip sync`
- Train: `uv run python train.py --difficulty medium`
- View logs: `tensorboard --logdir tb_logs`
- Evaluate: `uv run python test.py --difficulty medium`

## Future extensions
- MAML / FOMAML Meta-RL
- RL² recurrent policies
- Task-embedding networks
- Curriculum learning pipelines
