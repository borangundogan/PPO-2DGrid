# MERLIN: Modular PPO & Meta-RL Framework for MiniGrid

**MERLIN** (Meta-RL Environment & Robust Learning Infrastructure) is a research-grade reinforcement learning framework designed to bridge the gap between **Standard RL (PPO)** and **Meta-RL (FOMAML)**.

Built on a shared **Actor-Critic Backbone** with dynamic CNN/MLP switching, MERLIN focuses on solving the **"Catastrophic Forgetting"** problem, tackling strict **POMDP** (Partially Observable Markov Decision Process) constraints, and achieving **Zero-Shot Scale Generalization**.

---

## Current Research Status (Phase: Head-to-Head 5M Step Benchmark)

| Component | Status | Description |
| --- | --- | --- |
| **POMDP Architecture** | **LOCKED** | Agent vision strictly blocked by walls (`see_through_walls=False`). Forces true exploration under uncertainty rather than A* style routing. |
| **Meta-RL Stability** | **SOLVED** | Implemented **"Success-Gated Adaptation" (Safety Lock)** to prevent negative transfer on already-solved tasks. |
| **PPO Baseline** | **VALIDATED** | PPO trained for 5M steps reached asymptotic plateau (max Zero-Shot capability). Confirmed structural inability to solve unseen complex topologies. |
| **Sample Efficiency** | **ALIGNED** | FOMAML hyperparameter "Golden Ratio" established (1000 iters x 10 tasks x 512 steps = 5.12M steps) for 1:1 fair comparison with PPO. |
| **Academic Evaluation** | **INTEGRATED** | Multi-metric overlapping histograms (Reward, Steps, Validation Loss) for Zero/One/Ten-Shot analysis implemented. |

---

## 1. Core Technical Breakthroughs

We address critical failure modes in standard Meta-RL and RL baselines:

### 1. The POMDP Wall (Memoryless Baseline Failure)

We proved that a standard reactive PPO policy, even when trained to optimal heuristics (e.g., reaching 0.990 reward on open maps), consistently fails (0.000 reward) on unseen complex topologies (U-turns, hidden goals) due to lack of recurrent memory and fast adaptation.

### 2. Success-Gated Adaptation ("The Safety Lock")

In standard FOMAML, the agent updates its weights even if it performs perfectly pre-update, leading to Catastrophic Forgetting.

* **Our Solution:** A conditional update mechanism (`if Pre_Reward > Threshold: Skip_Update()`).
* **Result:** Eliminated negative transfer. The agent maintains expertise on known tasks while adapting only to failure cases.

### 3. Validation Loss as a Meta-Objective Metric

Instead of relying solely on environment rewards, we integrated **Validation Loss Distributions** into our evaluation suite. By measuring the Value Network (Critic) MSE on unseen query sets, we directly quantify how well FOMAML minimizes "surprise" compared to the baseline PPO.

---

## 2. Project Structure

```text
MERLIN/
|- checkpoints/               # Saved models (PPO and FOMAML)
|- eval_results/              # Output directory for evaluation histograms
|- src/
|  |- wrappers/               # Custom Envs (StuckPenalty, ExplorationBonus)
|  |- scenario_creator/       # Scenario Logic (MediumHard, Hard, sizes)
|  |- actor_critic.py         # Dynamic CNN/MLP Backbone
|  |- fomaml.py               # FOMAML Core Logic (Inner/Outer loop)
|  |- ppo.py                  # PPO Baseline Logic
|  |- sweep_checkpoints.py    # Zero-Shot Leaderboard Generator for PPO
|  |- distribution_over_tasks.py # Zero/One/Ten-Shot Academic Plotting Suite
|- ppo/train.py               # PPO 5M Step Training Script
|- ppo/train.sh               # Automated Batch Execution Script
|- fomaml/fomaml_train.py     # Meta-RL Training Script
|- fomaml/train_fomaml.sh     # Automated Batch Execution Script
|- fomaml/fomaml_visualization.py       # Live PyGame rendering of the agent
|- fomaml/analyze_fomaml_distribution.py # Cross-difficulty adaptation analysis

```

---

## 3. Getting Started & Execution Pipeline

### Installation

Dependency management via `uv` for high-speed synchronization.

```bash
uv pip sync
```

---

### Step 1: Training the PPO Baseline

Train the baseline on the `mediumhard` (16x16) environment for 5 Million steps to establish the memoryless performance ceiling.

**Make script executable:**

```bash
chmod +x ppo/train_ppo.sh
./ppo/train_ppo.sh
```

**Manual Python Execution:**

```bash
uv run python ppo/ppo_train.py \
  --difficulty mediumhard \
  --seed 777 \
  --total_steps 5000000
```

---

### Step 2: Finding the True Best Baseline (Checkpoint Sweeping)

During a 5M step training, the "best" model might be overfitted to its local epoch seeds. Use the sweeper to evaluate all saved `.pth` files against a fixed set of 100 *unseen* tasks and generate a leaderboard.

```bash
uv run python src/sweep_checkpoints.py \
    --difficulty mediumhard \
    --tasks 100 \
    --model_dir "checkpoints/MERLIN-MediumHard.../seed_777"
```

*Note: Use the Rank #1 model path from this output for the final head-to-head evaluation.*

---

### Step 3: Meta-Training (FOMAML)

Run FOMAML using the mathematically aligned "Golden Ratio" (5.12M total environment steps) to ensure a 1:1 fair comparison with PPO. Use the provided bash script for automated execution.

**Make script executable:**

```bash
chmod +x fomaml/train_fomaml.sh
./fomaml/train_fomaml.sh
```

**Manual Python Execution:**

```bash
uv run python fomaml/fomaml_train.py \
    --difficulty mediumhard \
    --seed 777 \
    --iterations 1000 \
    --tasks_per_batch 32 \
    --k_steps 256
```

---

### Step 4: PPO Baseline Visualization & Testing

Evaluate and visually inspect the trained PPO baseline agent. This script runs the agent through a specified number of episodes, rendering the environment in real-time (via PyGame) to observe its raw navigation heuristics, and outputs a terminal summary of steps and rewards.

```bash
uv run python ppo/ppo_visualization.py \
  --model_path "path/to/ppo/best_ppo_from_sweep.pth" \
  --difficulty mediumhard \
  --episodes 10 \
  --seed 123
```

### Step 5: PPO Cross-Difficulty Generalization Analysis (OOD)

Analyze the Out-of-Distribution (OOD) performance of the trained PPO baseline. This script evaluates the zero-shot generalization capabilities of the reactive model across a full spectrum of map difficulties (`easy` through `hardest`). It generates macro-level bar charts to visualize performance degradation at scale, alongside pairwise distribution histograms and statistical distance metrics (e.g., Wasserstein, KL-Divergence) to quantify the exact distributional shift between tasks.

```bash
uv run python ppo/analyze_ppo_distribution.py \
  --model_path "checkpoints/MERLIN-MediumHard-v0_16x16_mediumhard_20260313_150304/seed_777/ppo_model_1800k.pth" \
  --difficulties easy medium mediumhard hard hardest \
  --num_tasks 50 \
  --base_seed 300000
```

*Outputs: Generalization bar charts (`ppo_reward_generalization.png`), pairwise distribution histograms (`dist_shift_X_vs_Y.png`), and terminal distance metrics will be saved to the respective `analysis_results/` directory.*


*(Note: Append the `--no-render` flag if you want to bypass the graphical interface and rapidly compute metrics in the terminal).*

---

### Step 6: Visualizing the Agent (Live Playback)

Watch the trained FOMAML agent navigate the MiniGrid environment in real-time. This script opens a PyGame window allowing you to observe the agent's behavior and navigation efficiency on specific seeds.

```bash
uv run python fomaml/fomaml_visualization.py \
  --model_path "checkpoints/MERLIN-Mediumhard-16x16-v0_FOMAML_20260221_224759/seed_888/best_model.pth" \
  --difficulty mediumhard \
  --seed 999
```

---

### Step 7: Cross-Difficulty Generalization Analysis

Test how well an agent trained exclusively on one difficulty (e.g., `mediumhard`) adapts and scales its learned knowledge across the entire difficulty spectrum (`easy` through `hardest`). This generates robust metrics showing scale-invariance and adaptation flexibility.

```bash
uv run python fomaml/analyze_fomaml_distribution.py \
  --model_path "checkpoints/MERLIN-Mediumhard-16x16-v0_FOMAML_20260222_001215/seed_888/best_model.pth" \
  --difficulties easy medium mediumhard hard hardest \
  --num_tasks 50 \
  --k_support 100 \
  --lr_inner 0.001 \
  --seed 4382
```

---

### Step 8: Pre vs. Post Adaptation Analysis (FOMAML)

Isolate and quantify the exact impact of the inner-loop adaptation mechanism. This script evaluates the FOMAML agent on a set of unseen tasks, recording its performance *before* (Zero-Shot) and *after* (Few-Shot) taking `k` gradient steps. It generates adaptation scatter plots and performance bar charts to visually demonstrate the learning delta and verify the effectiveness of the adaptation phase.

```bash
uv run python fomaml/fomaml_evaluate.py \
  --model_path "path/to/fomaml/best_model.pth" \
  --difficulty mediumhard \
  --num_tasks 50 \
  --k_support 40 \
  --lr_inner 0.001 \
  --seed 1000
```

*Outputs: `adaptation_scatter.png` and `adaptation_bar_chart.png` (detailing the exact performance delta) will be automatically saved in the `analysis_results/meta_eval/` directory.*

---

### Step 9: Academic Evaluation (Zero / One / Ten-Shot)

Evaluate both models on 500 completely unseen tasks. This script generates overlapping histograms for **Reward**, **Steps**, and **Validation Loss** directly formatted for research papers.


**Zero-Shot Evaluation (No Inner-Loop Updates):**

```bash
uv run python -m src.distribution_over_tasks \
    --difficulty mediumhard \
    --num_tasks 500 \
    --adapt_steps 0 \
    --ppo_model "path/to/ppo/best_ppo_from_sweep.pth" \
    --fomaml_model "path/to/fomaml/best_model_from_sweep.pth"
```

**Ten-Shot Evaluation (10 Gradient Step Adaptation):**

```bash
uv run python -m src.distribution_over_tasks \
    --difficulty mediumhard \
    --num_tasks 500 \
    --adapt_steps 10 \
    --ppo_model "path/to/ppo/best_ppo_from_sweep.pth" \
    --fomaml_model "path/to/fomaml/best_model_from_sweep.pth"
```

*Outputs will be saved in the `eval_results/` directory as high-resolution PNGs.*
