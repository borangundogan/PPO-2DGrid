# MERLIN: Modular PPO Baseline for MiniGrid and Meta-RL Research

**MERLIN** (Meta-RL Environment & Robust Learning Infrastructure) is a research-grade reinforcement learning framework designed to quantify **task distribution shift**, **policy robustness**, and **generalization gaps** in grid-world navigation tasks.

Built on a robust **PPO (Proximal Policy Optimization)** baseline with automatic **CNN/MLP architecture switching**, MERLIN serves as the foundational benchmark for developing advanced Meta-RL algorithms (e.g., MAML, FOMAML). It provides a rigorous pipeline for measuring how well a standard RL agent can adapt to unseen topological structures using advanced statistical metrics like **Wasserstein Distance** and **Jensen-Shannon Divergence**.

---

## Current Research Status (Baseline Analysis)

| Component | Status | Description |
| :--- | :--- | :--- |
| **PPO Backbone** | Stable | Robust convergence on 'Medium' tasks (Reward > 0.97). |
| **Architecture** | Solved | Auto-detection for **CNN** (Image) vs. **MLP** (State) based on config. |
| **Checkpointing** | Solved | Implemented "Save Best Model" to prevent performance collapse. |
| **Generalization** | **Gap Identified** | Strong downward transfer (Medium $\to$ Easy), but **catastrophic failure** on upward transfer (Medium $\to$ Hard). |
| **Metric Pipeline** | Active | Wasserstein & KDE analysis fully implemented. |

---

## 1. Core Features

### Adaptive Actor-Critic Architecture
- **PPO from Scratch:** Implements clipped surrogate objective, GAE (Generalized Advantage Estimation), entropy regularization, and gradient norm clipping.
- **Input Agnostic:** Automatically initializes `CNNActorCritic` for pixel-based observations (3D tensors) or `MLPActorCritic` for flattened states.
- **Stability:** Features normalized observation spaces, orthogonal weight initialization, and minimal action space wrappers.

### ScenarioCreator & Task Hierarchy
All environments are defined centrally in `src/config/scenario.yaml` to ensure strict comparability (fixed grid size $16 \times 16$).

| Difficulty | Topology Description | Cognitive Requirement |
| :--- | :--- | :--- |
| **Easy** | Empty grid. Fixed goal. | Basic Motor Control |
| **Medium** | Random agent/goal placement. No obstacles. | Visual Goal Recognition |
| **MediumHard**| Scattered pillars (random obstacles). | Steering & Maneuvering |
| **Hard** | Wall split with a single gap. | Path Planning (Deceptive Reward) |
| **Hardest** | Four-Rooms layout with debris. | Long-horizon Navigation |

### Quantitative Analysis Module (`src/analyze_tasks.py`)
Moving beyond simple scalar rewards, MERLIN analyzes the **Probability Distribution of Returns** to diagnose failure modes:
- **Wasserstein Distance:** Quantifies the "physical cost" to transform the reward distribution of the training task to the test task. High values indicate a Distribution Shift.
- **Jensen-Shannon Divergence:** Measures the similarity between policy behaviors across tasks.
- **Reward Histograms:** Visualizes whether the policy is robust (unimodal peak at 1.0) or reliant on luck (bimodal distribution at 0.0 and 1.0).

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

---

## 3. Project Structure

```
MERLIN/
|- analysis_results/       # Generated histograms and metric reports
|- checkpoints/            # Saved models (Last and Best)
|- src/
|  |- actor_critic.py     # Shared backbone (CNN & MLP)
|  |- analyze_tasks.py    # Quantitative analysis pipeline (Wasserstein/KDE)
|  |- config/             # Centralized environment configuration
|  |- custom_envs/        # MiniGrid extensions (Easy, Medium, Hard, etc.)
|  |- metrics/            # Mathematical metrics (KL, JSD, Wasserstein)
|  |- ppo.py              # PPO Algorithm implementation
|  |- train.py            # Main training loop with TensorBoard
|  |- test.py             # Visualization and qualitative testing
|- tb_logs/                # TensorBoard logs
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

* [x] **Phase 1: Baseline Stability** (Solved Checkpointing & Architecture)
* [x] **Phase 2: Quantifying Failure** (Wasserstein Analysis Complete)
* [ ] **Phase 3: Curriculum Learning** (Train on `MediumHard` to bridge the gap)
* [ ] **Phase 4: Meta-RL Integration** (Implement Recurrent/Memory-based architectures to solve Hard/Hardest tasks via fast adaptation)
