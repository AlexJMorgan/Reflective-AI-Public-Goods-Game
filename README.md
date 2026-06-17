# Reflective Governance in Multi-Agent Reinforcement Learning

## Overview

This project investigates whether adding a **reflective governance layer** to reinforcement learning agents can help them overcome the "Tragedy of the Commons" – a social dilemma where individual self-interest leads to the destruction of a shared resource.

The code implements a **Public Goods Game** environment where multiple agents decide whether to:
- **Contribute** (pay a cost to help maintain a shared resource)
- **Free ride** (take the benefit without paying)

Standard Q-learning agents tend to learn to free-ride, causing the shared resource to collapse. **Reflective agents** have an extra "governor" layer that overrides selfish behaviour when the resource becomes unsustainable.

---

## Project Structure

```
├── publicgoodsenv.py      # Environment: the Public Goods Game
├── agents.py              # Q-learning agents (baseline + reflective)
└── difficultyanalysis.py  # Main experiment runner & visualisation
```

---

## Files Explained

### `publicgoodsenv.py`

Defines the Public Goods Game environment. Multiple agents share a resource (public good) that:
- **Decays** over time
- **Replenishes** when agents contribute
- **Provides benefits** to all agents (whether they contributed or not)

**Key parameters:**

| Parameter | What it does |
|-----------|---------------|
| `nAgents` | Number of players |
| `contributionCost` | How much an agent loses when they contribute |
| `decayRate` | How fast the public good shrinks each round |
| `replenishmentRate` | How much each contribution adds to the public good |
| `benefitMultiplier` | Scales how much benefit agents receive from the public good |
| `sustainabilityThreshold` | The minimum level needed to keep the resource healthy |

### `agents.py`

Contains two agent classes:

**`QLearningAgent` (Baseline)**
- Standard Q-learning with epsilon-greedy exploration
- Maximises individual wealth using the Bellman equation
- No reflection capability

**`ReflectiveAgent` (Experimental)**
- Inherits from `QLearningAgent`
- Adds a **reflective governor** that monitors the public good
- Governor decision rule: if the public good falls below a threshold AND the agent tries to free-ride, the governor forces cooperation instead
- Tracks intervention rate to measure how often the governor overrides

### `difficultyanalysis.py`

Runs experiments and generates graphs. Tests three difficulty levels (Easy, Medium, Hard) with different environment parameters:
- Decay rate (how fast resource shrinks)
- Replenishment rate (how much contributions help)
- Benefit multiplier (how much agents gain from the resource)
- Sustainability threshold (the target level to maintain)

---

## Requirements

```bash
pip install numpy matplotlib
```

---

## How to Run

Run all experiments and generate graphs:

```bash
python difficultyanalysis.py
```

This will:
1. Run multiple replications for each difficulty level (Easy, Medium, Hard)
2. Test both baseline and reflective agents
3. Generate three graphs:
   - `difficulty_cooperation.png` – Cooperation rate over time
   - `difficulty_public_good.png` – Public good level over time (with threshold line)
   - `difficulty_intervention.png` – Governor intervention rate (reflective agents only)

---

## What the Graphs Show

| Graph | What to look for |
|-------|------------------|
| **Cooperation Rate** | Baseline agents drop to low cooperation. Reflective agents maintain significantly higher cooperation. |
| **Public Good Level** | Baseline agents let the resource collapse below threshold. Reflective agents keep the resource near or above the threshold. |
| **Intervention Rate** | Governor intervenes more frequently in harder difficulties when the resource is under greater threat. |

---

## Key Findings

- **Baseline agents** reliably converge to free-riding behaviour regardless of difficulty
- **Reflective agents** achieve substantially higher cooperation rates than baseline
- **Governor intervention** increases with environmental difficulty, showing the system responds appropriately to challenge
- **Public good sustainability** is maintained or partially maintained by reflective agents, while baseline agents cause complete collapse

---

## Customising Experiments

To change the number of agents, edit `createDifficultyConfigurations()` in `difficultyanalysis.py`:

```python
base_config = {
    'nAgents': 3,  # Change this
    # ... other parameters
}
```

To change the number of episodes or replications:

```python
EPISODES       = 500   # How many rounds per run
N_REPLICATIONS = 500   # How many times to repeat
```

---

## Key Equations

**Reward calculation:**
```
reward = change in wealth = newWealth - oldWealth
```

**Public good update:**
```
publicGood = (decayRate × publicGood) + (replenishmentRate × numContributors)
```

**Governor decision rule:**
```
IF publicGood < threshold AND proposed_action == free_ride:
    OVERRIDE → force contribute
ELSE:
    APPROVE → execute proposed action
```

**Q-learning update (Bellman equation):**
```
Q(s,a) ← Q(s,a) + α[r + γ × max Q(s',a') - Q(s,a)]
```

---
