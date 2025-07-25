# 🚦 VSL-RL: Reinforcement Learning for Variable Speed Limits in Traffic Simulation

This repository implements a reinforcement learning framework for optimizing **Variable Speed Limits (VSL)** using a simulation based on the **MetaNet** traffic model. It combines traffic dynamics modeling with advanced actor-critic agents like **Soft Actor-Critic (SAC)** and **Worst-Case SAC (WCSAC)**.

---

## 🗂️ Project Structure

| File / Folder      | Description |
|--------------------|-------------|
| `train_sum.py`         | **Main script**: Runs training loop, agent interaction, evaluation, and plots. |
| `sim_env_full.py`  | **Simulation environment** using MetaNet and VSL logic, including real traffic structure. |
| `agent_critic.py`  | Actor-Critic agent with SAC + optional WCSAC extensions (safety critic, CVaR, etc.). |
| `network.py`       | Contains neural network architectures for policy and value functions. |
| `utils.py`         | Plotting functions and helper utilities. |
| `data_demands.py`  | Functions to generate demand inputs (synthetic sinusoidal or real traffic-derived). |
| `sim_env_eg.py`, `sim_env_baseline.py`, etc. | Alternate or earlier versions of the simulation environment. |

---

## 🚀 How to Run

### 1. Install Dependencies

Use the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- `tensorflow`
- `casadi`
- `numpy`
- `matplotlib`
- `pandas`
- `scipy`

### 2. Run Training

```bash
python train_sum.py
```

You can edit `train_sum.py` to switch environments or agents (e.g., default vs custom).

---

## 🧠 Agent Highlights

- **Soft Actor-Critic (SAC):** Optimizes a stochastic policy with entropy regularization.
- **WCSAC (Optional):**
  - Safety critic to penalize risky speed drops.
  - CVaR-based loss for conservative decision-making under uncertainty.
  - Adaptive λ to balance reward vs safety.
- **Reward Function:** Minimizes Total Time Spent (TTS) on main road and ramp queues.

---

## 🛣️ Environment Details

- Based on MetaNet (via `sym-metanet`) with real-segment modeling.
- Accepts dynamic demand from real or synthetic sources.
- Simulates queues, speeds, and densities across multiple road segments.
- Predicts future speed as part of the state for better control.

---

## 📊 Outputs

- Episode rewards and evaluation scores.
- Speed profiles across segments per episode.
- Optional plots and model checkpointing.

---

## 📝 Notes

- All versions of the environment are modular—only import the one you need.
- Real-world speed data (e.g., 1-min intervals) is processed and interpolated to match simulation step frequency.

---

## 📎 License

This project is for educational/research purposes.

---

## 🙋‍♀️ Author

Anushka Narsima Amarnath
Final Year, Computer Systems Engineering  
AI Projects | https://www.linkedin.com/in/anushka-narsima-a-00729124b/