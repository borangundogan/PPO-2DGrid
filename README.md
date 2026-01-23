# MERLIN: Modular PPO & Meta-RL Framework for MiniGrid

**MERLIN** (Meta-RL Environment & Robust Learning Infrastructure) is a research-grade reinforcement learning framework designed to bridge the gap between **Standard RL (PPO)** and **Meta-RL (FOMAML)**.

Built on a shared **Actor-Critic Backbone**, MERLIN focuses on solving the **"Catastrophic Forgetting"** problem in Meta-Learning and achieving **Zero-Shot Scale Generalization** (transferring knowledge from 8x8 to 16x16 maps).

---

## Current Research Status (As of Jan 2026)

| Component | Status | Description |
| :--- | :--- | :--- |
| **Meta-RL Stability** | **SOLVED** | Implemented **"Success-Gated Adaptation" (Safety Lock)**. Improvement delta increased from **-0.54** (Naive) to **+1.06** (Robust). |
| **Scale Transfer** | **ACHIEVED** | Model trained on **8x8 MediumHard** successfully navigates **16x16** environments without retraining. |
| **Hyperparameters** | **OPTIMIZED** | Identified the "Sweet Spot" for adaptation: **$k=40$, $lr=0.001$**. Proven that $k=100$ causes overfitting to noise. |
| **Navigation** | **MASTERED** | Solved the "Looping" problem using a **Hybrid Exploration Wrapper** (Stuck Penalty + Novelty Bonus). |

---

## 1. Core Technical Breakthroughs

We addressed three critical failure modes in standard Meta-RL:

### 1. Success-Gated Adaptation ("The Safety Lock")
In standard FOMAML, the agent is forced to update its weights even if it performs perfectly on the pre-update task. This leads to **Catastrophic Forgetting** (overwriting optimal weights with noisy gradient updates).
* **Our Solution:** We implemented a conditional update mechanism.
* **Logic:** `if Pre_Reward > Threshold (0.50): Skip_Update()`
* **Result:** Eliminated negative transfer. The agent maintains expertise on known tasks while adapting only to failure cases.

### 2. The "Sweet Spot" (Ablation Study)
We conducted a sensitivity analysis on Inner Loop parameters ($k$ steps and Learning Rate).
* **Finding:** "Less is More".
    * **$k=100$:** Agent over-explores, collecting noisy data (banging into walls after finding the goal). Result: Lower performance.
    * **$k=40$:** Optimal duration. Agent finds the goal and stops. Result: Clean gradients.
    * **$lr=0.1$:** "Sledgehammer" effect. Destroys pre-trained weights.
    * **$lr=0.001$:** "Scalpel" effect. Fine-tunes navigation without breaking the policy.

### 3. Scale Generalization (8x8 $\to$ 16x16)
Instead of retraining on expensive large maps, we trained on **8x8 MediumHard** (Concept Learning) and tested on **16x16**.
* **Result:** The agent learned **local navigation rules** (wall following, gap finding) rather than coordinate memorization. It successfully navigates 16x16 mazes Zero-Shot.

---

## 2. Experimental Results

### A. Adaptation Analysis (The Impact of Safety Lock)
Comparison of Naive FOMAML vs. MERLIN's Robust FOMAML on 50 unseen tasks.

| Method | Avg Improvement (Delta) | Interpretation |
| :--- | :--- | :--- |
| **Naive FOMAML** ($k=40$) | **-0.5468** | **Negative Transfer.** Forced updates broke successful policies. |
| **High-LR FOMAML** ($lr=0.1$) | **-10.375** (Worst Case) | **Policy Collapse.** Aggressive updates erased memory. |
| **MERLIN (Robust)** | **+1.0649** | **Constructive Adaptation.** Only updates when necessary. Saved "Miracle Cases" (e.g., Seed 4400: -9.4 $\to$ +0.58). |

### B. Training Convergence
The model achieves high stability on 8x8 MediumHard tasks, minimizing "Stuck Events" (Wall Banging) to near zero.

*(See `analysis_results/` for full training curves)*
!Training Curves

---

## 3. Project Structure

```text
MERLIN/
|- analysis_results/   # Adaptation Scatter Plots & Bar Charts
|- checkpoints/        # Saved Best Models (e.g., seed_4382)
|- src/
|  |- wrappers/          # Custom Envs (StuckPenalty, ExplorationBonus)
|  |- scenario_creator/  # Scenario Logic (8x8, 16x16, MediumHard, Hard)
|  |- fomaml.py          # Meta-Learner with Safety Lock Logic
|  |- ppo.py             # Baseline Policy
|- train_fomaml.py     # Meta-RL Training Script
|- evaluate_meta.py    # Meta-Evaluation Script (with Success Gating)
```

---

## 5. Getting Started

### Installation
Recommended to use `uv` for fast dependency management, but standard pip works too.

```bash
# Using uv (recommended)
uv pip sync
# Or using pip
pip install -r requirements.txt
```

### Training (PPO Baseline)
Train the baseline on the Medium environment. The system automatically saves the "Best Model" (based on evaluation rewards) to `checkpoints/`.

```bash
uv run python ppo/train.py \
  --difficulty medium \
  --total_steps 300000 \
  --seed 42 \
  --device auto
```

### Meta-Training (FOMAML)
Train the Meta-Learner to find an adaptable initialization.

```bash
uv run python fomaml/fomaml_train.py \
  --difficulty mediumhard \
  --iterations 2000 \
  --seed 42 \
  --device auto
```

### Meta-Evaluation
Test how well the trained model adapts to unseen tasks (e.g., 50 new maps).

```bash
uv run python fomaml/fomaml_evaluate.py \
  --model_path checkpoints/fomaml/mediumhard_seed42/best_model.pth \
  --difficulty hard \
  --num_tasks 50
```

### Monitoring (TensorBoard)
Track Loss, Entropy, KL Divergence, and Reward curves in real-time.

```bash
tensorboard --logdir tb_logs
```

## 6. Future Roadmap
[x] Phase 1: Architecture Stabilization (PPO + FOMAML Integration) 

[x] Phase 2: The "Safety Lock" (Success-Gated Adaptation)

[x] Phase 3: Realism (Solved "Blind" Navigation & Wall Stuck problem)

[x] Phase 4: Ablation Study (Hyperparameter Tuning)

[x] Phase 5: Mastery (Achieved >0.97 Reward on Hard Tasks)

[x] Phase 6: Scale Transfer (8x8 $\to$ 16x16)

[x] Phase 7: Transfer Learning (Test on 16x16 grids to measure resolution invariance)