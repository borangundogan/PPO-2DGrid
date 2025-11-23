# PPO: Modular PPO Pipeline for MiniGrid & Meta-RL Research

This repository provides a clean, extensible, and research-grade PPO implementation designed for MiniGrid and task-distribution evaluation. It is built to support curriculum learning, Meta-RL experiments, and quantitative cross-task analysis.

## Features
1. **PPO Implementation (from scratch)**
   - Clipped surrogate objective
   - GAE advantage estimator
   - Shared actor–critic backbone
   - CNN/MLP auto-switch based on observation shape
   - Gradient norm clipping
   - Full diagnostics: KL divergence, clip fraction, entropy decay, value loss, policy loss, gradient norm
   - Recorded directly into TensorBoard under `loss/` and `diagnostics/`

2. **ScenarioCreator Framework**
   - All environments are created from a single YAML file: `src/config/scenario.yaml`
   - Defines task difficulties (easy, medium, hard, hardest)
   - Defines observation mode: partial or full observation, flatten vs CNN
   - All environments validated to fixed-size grids
   - Train/test environments follow the same settings
   - Example:
     ```yaml
     difficulties:
       easy:
         env_id: "Grid-Easy-16x16-v0"
       medium:
         env_id: "Grid-Medium-16x16-v0"
       hard:
         env_id: "Grid-Hard-16x16-v0"
       hardest:
         env_id: "Grid-Hardest-16x16-v0"

     observation:
       fully_observable: false
       flatten: true
     ```

3. **Custom MiniGrid Environments**
   - Implemented under `src/custom_envs/`
   - Current difficulty set:
     - Easy: empty 16×16 grid
     - Medium: random internal walls
     - Hard: vertical wall split with a passage and random goal
     - Hardest: four-rooms maze, fully connected, random obstacles and random start/goal
   - All are deterministic and reproducible

4. **Three-Action Wrapper**
   - `src/wrappers/three_action_wrapper.py`
   - Normal MiniGrid has many actions; PPO learns faster with a minimal action space:

     | ID | Action      |
     | -- | ----------- |
     | 0  | turn left   |
     | 1  | turn right  |
     | 2  | move forward|

   - Applies to all environments

5. **Modular Actor-Critic Networks**
   - `src/wrappers/actor_critic.py`
   - `MLPActorCritic` for flattened input
   - `CNNActorCritic` for image input (HxWxC)
   - Automatic normalization (uint8 → [0,1])
   - Automatic shape detection during PPO initialization

6. **Logging & Metrics System**
   - TensorBoard logs include:
     - Scalars: `loss/policy_loss`, `loss/value_loss`, `loss/entropy`, `diagnostics/kl`, `diagnostics/clipfrac`, `diagnostics/gradnorm`, `stats/episode_return_mean`, `stats/episode_length_mean`
     - Histograms: `hist/episode_rewards`, `hist/episode_lengths`
     - Figures: scatter plot Reward vs Episode Length logged as `fig/reward_vs_steps`
   - PPO update metrics refactored in `src/metrics/ppo_metrics.py` for clean aggregation and reuse

7. **Complete Training Pipeline**
   - `train.py`
   - Timestamped Run-IDs
   - Automatic checkpoint saving: `checkpoints/<env>_<difficulty>_<timestamp>/ppo_model.pth`
   - TensorBoard directory: `tb_logs/<env>_<difficulty>_<timestamp>/`
   - Supports per-step logging, per-N-steps printing (`--print_interval`), live evaluation after each PPO update, and visual evaluation mode
   - Example:
     ```bash
     uv run python train.py \
       --difficulty medium \
       --total_steps 500000 \
       --batch_size 4096 \
       --minibatch_size 512 \
       --eval_episodes 5
     ```

8. **Evaluation Pipeline**
   - `test.py`
   - Loads latest checkpoint automatically
   - Supports manual model path override
   - Cross-difficulty generalization tests
   - Uses consistent CNN/MLP preprocessing
   - Supports human rendering
   - Example:
     ```bash
     uv run python test.py --difficulty hard
     ```

   - Or, test a model trained on medium in the hard environment:
     ```bash
     uv run python test.py \
       --difficulty hard \
       --model_path checkpoints/Grid-Medium-16x16-v0_medium_20251120_143903/ppo_model.pth
     ```

## Project Structure
```
src/
├── config/
│   └── scenario.yaml
│
├── custom_envs/
│   ├── easy_env.py
│   ├── medium_env.py
│   ├── hard_env.py
│   └── hardest_env.py
│
├── scenario_creator/
│   └── scenario_creator.py
│
├── wrappers/
│   ├── actor_critic.py
│   ├── rollout_buffer.py
│   ├── three_action_wrapper.py
│   └── utils.py
│
├── metrics/
│   └── ppo_metrics.py
│
├── ppo.py
├── train.py
└── test.py
```

## Results Summary (Example Run)
| Train Env | Test Env | Avg Reward |
| --------- | -------- | ---------- |
| Easy      | Easy     | 0.94–0.96  |
| Medium    | Medium   | 0.92–0.95  |
| Medium    | Hard     | ~0.45      |
| Medium    | Hardest  | Fails      |

**Interpretation**
- PPO learns Empty and Random-Spawn tasks reliably.
- Generalization to Hard and Hardest is non-trivial—expected.
- This gap is what Meta-RL will address (MAML / PEARL / RL²).

## Getting Started
- Install dependencies: `uv pip sync`
- Train: `uv run python train.py --difficulty medium`
- View logs: `tensorboard --logdir tb_logs`
- Test: `uv run python test.py --difficulty medium`

## Future Extensions
- MAML / FOMAML-based Meta-RL
- RL² recurrent policies
- Task-embedding networks
- Distribution shift analysis (KL, JS, Wasserstein)
- Curriculum learning pipelines
