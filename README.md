# PPO: Proximal Policy Optimization for MiniGrid Tasks

PPO is a clean, modular, and research-grade PPO pipeline focused on **MiniGrid** environments. It combines scenario generation, observation preprocessing, reduced action-spaces, logging, and cross-task evaluation to serve as a foundation for MiniGrid-based RL and Meta-RL research.

## Main Features
1. **PPO Implementation (from scratch)**
   - Clipped surrogate loss with GAE advantages and shared actor–critic trunk
   - KL divergence, clip fraction, entropy, and gradient-norm diagnostics
   - Full TensorBoard integration for scalar/histogram/scatter summaries

2. **ScenarioCreator Framework**
   - YAML-driven environment builder (`src/config/scenario.yaml`) with auto train/test render modes
   - Supports fixed-size validation sets, partial/fully observable pipelines, and flattening
   - Example:
     ```yaml
     difficulties:
       easy: MiniGrid-Empty-6x6-v0
       medium: MiniGrid-Empty-Random-6x6-v0
       hard: MiniGrid-Dynamic-Obstacles-6x6-v0

     observation:
       fully_observable: false
       flatten: true
     ```

3. **Three-Action Wrapper**
   - Remaps PPO actions to the minimal set required by most MiniGrid tasks:

     | Index | Action      |
     | ----- | ----------- |
     | 0     | turn left   |
     | 1     | turn right  |
     | 2     | move forward|

   - Simplifies policy learning and improves cross-difficulty generalization.

4. **Actor-Critic Architectures**
   - `MLPActorCritic` for flattened observations
   - `CNNActorCritic` with dynamic convolutional heads for image-based PPO

5. **Training Pipeline**
   - Timestamped checkpoints, automatic run-ID creation, frequent progress prints
   - TensorBoard scalars: policy loss, value loss, entropy, KL, clip fraction, episodic returns/lengths
   - Histogram + scatter tracking for deeper optimization insight

6. **Evaluation Pipeline**
   - Auto-loads latest checkpoint per difficulty/run
   - Cross-difficulty evaluation with normalized forward pass
   - Human render mode and compatibility with the three-action wrapper

## Project Structure
```
src/
├── actor_critic.py            # MLP + CNN policy/value networks
├── ppo.py                     # PPO algorithm (updates, rollouts, advantages)
├── utils.py                   # Device helpers, normalization, rollout buffers
├── config/
│   └── scenario.yaml          # ScenarioCreator configuration
├── scenario_creator/
│   └── scenario_creator.py    # YAML-driven environment builder
├── wrappers/
│   └── three_action_wrapper.py# Reduced action-space wrapper
├── train.py                   # Training entrypoint
└── test.py                    # Evaluation / cross-task testing
```

## Example Usage
### Train
```bash
uv run python train.py \
    --difficulty easy \
    --total_steps 300000 \
    --batch_size 4096 \
    --minibatch_size 512 \
    --update_epochs 6 \
    --lr 2.5e-4 \
    --ent_coef 0.005 \
    --vf_coef 0.5 \
    --device cpu
```

### Test (auto-load latest checkpoint)
```bash
uv run python test.py --difficulty easy
```

### Cross-scenario testing
```bash
uv run python test.py \
    --difficulty medium \
    --model_path checkpoints/MiniGrid-Empty-6x6-v0_easy_xxxxxx/ppo_model.pth
```

## Results Summary
| Train Env      | Test Env                                 | Average Reward |
| -------------- | ---------------------------------------- | -------------- |
| Easy (Empty)   | Easy                                     | 0.937          |
| Easy           | Medium (Empty-Random)                    | 0.967          |
| Easy           | Hard (Dynamic Obstacles)                 | -0.604         |

**Interpretation**
- PPO masters 6×6 Empty navigation, including random spawn generalization.
- Curriculum or harder training distributions are needed for Dynamic Obstacles.

## Getting Started
1. Install dependencies with `uv pip sync` (or your preferred environment manager).
2. Adjust `src/config/scenario.yaml` to design new task suites.
3. Run training/evaluation commands above and monitor TensorBoard in `tb_logs`.
