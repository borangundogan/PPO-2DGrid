# MERLIN: Modular PPO & Meta-RL Framework for MiniGrid

**MERLIN** (Meta-RL Environment & Robust Learning Infrastructure) is a research-grade reinforcement learning framework designed to bridge the gap between **Standard RL (PPO)** and **Meta-RL (FOMAML)**.

Built on a shared **Actor-Critic Backbone**, MERLIN quantifies two distinct types of intelligence:
1.  **Generalization (PPO):** How well a fixed policy performs on unseen tasks (Zero-Shot).
2.  **Adaptation (Meta-RL):** How quickly a policy can "learn to learn" when exposed to a new environment for just a few steps (Few-Shot).

The framework provides a rigorous pipeline for measuring **Task Distribution Shift** using advanced statistical metrics (Wasserstein Distance) and **Adaptation Deltas** (Pre-Update vs. Post-Update rewards).

---

## Current Research Status

| Component | Status | Description |
| :--- | :--- | :--- |
| **PPO Baseline** | **Stable** | Converges on 'Medium' tasks. Fails on 'Hard' (Wall-Banging). |
| **Meta-RL (FOMAML)** | **Active** | **Verified Adaptation.** Agent learns to navigate obstacles it previously failed on after just 50 steps of practice. |
| **Architecture** | **Unified** | Auto-switches between **CNN** (Pixels) and **MLP** (Flat) for both PPO and Meta-RL. |
| **Evaluation** | ** rigorous** | Standardized on **Deterministic (Argmax)** evaluation to eliminate stochastic noise from metrics. |
| **Metric Pipeline** | **Active** | Measures "Adaptation Delta" (Pre vs. Post Reward) and Distribution Shift (Wasserstein). |

---

## 1. Core Features

### Dual-Algorithm Support
- **PPO (Proximal Policy Optimization):** The baseline agent. Optimized for stability with GAE, Entropy Regularization, and Orthogonal Initialization. Used to measure pure generalization limits.
- **FOMAML (First-Order MAML):** The meta-learner. Implements a "Two-Loop" optimization process (Support Set $\to$ Inner Update $\to$ Query Set) to find a weight initialization that is highly adaptable.

### ScenarioCreator & Task Hierarchy
All environments are centralized in `src/config/scenario.yaml`. The system supports **Task Sampling** (generating unique seeds per meta-batch).

| Difficulty | Topology Description | Meta-RL Challenge |
| :--- | :--- | :--- |
| **Easy** | Empty grid. Fixed goal. | Basic Motor Control |
| **Medium** | Random agent/goal placement. | Visual Search Strategy |
| **MediumHard**| Scattered pillars (obstacles). | **Adaptation Test:** Can the agent learn to avoid a pillar it just hit? |
| **Hard** | Wall split with a single gap. | **Path Planning:** Requires memory of the barrier structure. |

### Quantitative Analysis Module (`src/evaluate_meta.py`)
MERLIN moves beyond simple rewards by analyzing the **Adaptation Gap**:
- **Pre-Update (Zero-Shot):** Performance of the initialized policy on a new map.
- **Post-Update (Few-Shot):** Performance after $K=50$ steps of gradient descent on the specific map.
- **Delta:** The quantitative proof of "learning to learn" (Positive Delta = Successful Adaptation).

---

## 2. Experimental Results (The Generalization Gap)

We trained a PPO agent on **Medium** difficulty (random start/goal, no walls) until convergence and evaluated it across other topologies.

### Quantitative Metrics
| Train Env | Test Env | Mean Reward | Wasserstein Dist. | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Medium** | **Easy** | **0.96 +/- 0.02** | **0.019** (Low) | **Perfect Generalization.** The agent treats the empty room as a subset of the medium task. |
| **Medium** | **Medium** | **0.97 +/- 0.01** | Reference | Baseline performance. |
| **Medium** | **Hard** | **0.48 +/- 0.49** | **0.492** (High) | **Generalization Failure.** Performance collapses to a coin flip (see below). |

### Visual Analysis: The "All-or-Nothing" Failure
The reward distribution histograms reveal the root cause of the failure in the **Hard** environment:
1.  **Medium Task:** A sharp, unimodal peak near Reward=1.0. The agent is confident and consistent.
2.  **Hard Task:** A **Bimodal Distribution**. The agent either solves the task perfectly (Reward $\approx$ 1.0) when spawned favorably, or fails completely (Reward 0.0) when separated by a wall, proving it lacks true path planning capabilities.

## 2. Experimental Results (Meta-Learning Validation)

We trained FOMAML on **Medium** tasks and evaluated adaptation on **MediumHard** (obstacles).

**Key Finding:** The agent demonstrates **Active Adaptation**.
* On tasks where the initial policy failed (Reward 0.0), the "Fast Weights" (after 1 update) achieved near-perfect performance (Reward > 0.98).
* This proves the agent learned a "Search Heuristic" rather than a fixed path.

**Sample Evaluation Log (Iter 900):**
```text
Task Seed | Pre-Reward | Post-Reward | Delta
----------------------------------------------
1002      | 0.000      | 0.995       | +0.995  (SUCCESS: Adaptation)
1009      | 0.000      | 0.986       | +0.986  (SUCCESS: Adaptation)
1006      | 0.993      | 0.000       | -0.993  (FAILURE: Instability)
```

---


## 3. Project Structure

```
MERLIN/
|- analysis_results/       # Generated histograms and adaptation plots
|- checkpoints/            # Saved models (PPO and FOMAML)
|- src/
|  |- actor_critic.py     # Shared backbone (CNN & MLP)
|  |- fomaml.py           # Meta-RL Implementation (Inner/Outer Loop logic)
|  |- ppo.py              # Baseline Implementation
|  |- scenario_creator/   # Environment & Task Sampling logic
|  |- analyze_tasks.py    # PPO Distribution Shift Analysis
|  |- evaluate_meta.py    # Meta-RL Adaptation Analysis
|- train.py                # PPO Training Script
|- train_fomaml.py         # Meta-RL Training Script
|- README.md
```
---

## 4. Getting Started

### Installation
Recommended to use `uv` for fast dependency management, but standard pip works too.

```bash
# Using uv (recommended)
uv pip sync
# Or using pip
pip install -r requirements.txt

```

### Training

Train the baseline on the Medium environment. The system automatically saves the "Best Model" (based on evaluation rewards) to `checkpoints/`.

```bash
uv run python train.py \
  --difficulty medium \
  --total_steps 300000 \
  --seed 42 \
  --device auto

```

### Meta-Training (FOMAML)
Train the Meta-Learner to find an adaptable initialization.

```bash
uv run python train_fomaml.py \
  --difficulty medium \
  --iterations 2000 \
  --seed 42 \
  --device auto
```

### Meta-Evaluation
Test how well the trained model adapts to unseen tasks (e.g., 20 new maps).

```bash
uv run python src/evaluate_meta.py \
  --model_path checkpoints/fomaml/medium_seed42/fomaml_iter_2000.pth \
  --difficulty medium \
  --num_tasks 20
```

### Monitoring (TensorBoard)

Track Loss, Entropy, KL Divergence, and Reward curves in real-time.

```bash
tensorboard --logdir tb_logs
```

### Evaluation (Qualitative)

Watch the trained agent play in real-time (`render_mode="human"`). This uses the deterministic `policy.act()` method.

```bash
uv run python test.py \
  --difficulty hard \
  --model_path checkpoints/<Experiment_ID>/seed_42/best_model.pth

```

### Analysis (Quantitative)

Run the full distribution shift analysis pipeline to generate histograms and compute Wasserstein distances.

```bash
uv run python -m src.analyze_tasks \
  --model_path checkpoints/<Experiment_ID>/seed_42/best_model.pth \
  --difficulties easy medium hard \
  --episodes 100

```

---

## 5. Future Roadmap

* [x] Phase 1: Baseline Stability (Solved PPO Checkpointing & Architecture)

* [x] Phase 2: Quantifying Failure (Proven Generalization Gap on Hard tasks)

* [x] Phase 3: Meta-RL Implementation (FOMAML working with verified adaptation)

* [ ] Phase 4: Robustness Tuning (Reduce negative adaptation delta via Lower Inner LR)

* [ ] Phase 5: Hard Task Mastery (Scale Meta-RL to solve the Wall/Deceptive Reward problem)