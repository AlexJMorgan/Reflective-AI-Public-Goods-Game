import numpy as np
import matplotlib.pyplot as plt
import random
from publicgoodsenv import PublicGoodsEnv
from agents import createAgents

'''
Run experiment with varying difficulty levels.
Difficulty is calibrated by adjusting environment parameters to create different sustainability thresholds and decay dynamics.
'''



# Colour palette for consistent plotting
COLOUR_BASELINE   = '#e74c3c'
COLOUR_REFLECTIVE = '#27ae60'
COLOUR_THRESHOLD  = '#f0c419'

DIFFICULTIES = ['Easy', 'Medium', 'Hard']



# Single experiment
def runSingleExperiment(agentType, envParams, episodes=300, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Initialize environment and agents
    env = PublicGoodsEnv(**envParams)

    agentParams = {
        'alpha':        0.1,
        'gamma':        0.95,
        'epsilon':      1.0,
        'epsilonMin':   0.01,
        'epsilonDecay': 0.995,
    }

    # Override parameters for reflective agents
    if agentType == 'reflective':
        agentParams['alpha']        = 0.2
        agentParams['epsilonDecay'] = 0.990
        agentParams['sustainabilityThreshold'] = envParams['sustainabilityThreshold']
        agentParams['interventionProbability'] = 1.0

    # Create agents
    agents = createAgents(agentType, envParams['nAgents'], **agentParams)

    # Initialize tracked variables
    cooperationRates  = []
    publicGoods       = []
    interventionRates = [] if agentType == 'reflective' else None

    for episode in range(episodes):
        state = env.reset()
        done  = False
        episodeActions = []

        # Run one episode
        while not done:
            actions = []
            for agent in agents:
                if agentType == 'reflective':
                    # Get action and whether it was overridden by the governor
                    action, _ = agent.getActionWithReflection(state, env.getPublicGood())
                else:
                    action = agent.getAction(state)
                actions.append(action)

            nextState, rewards, done, _ = env.step(actions)

            # Update each agents Q-values
            for i, agent in enumerate(agents):
                agent.update(state, actions[i], rewards[i], nextState)

            state = nextState
            episodeActions.append(actions)
    
        # Decay epsilon after each episode
        for agent in agents:
            agent.decayEpsilon()

        cooperationRates.append(np.mean(episodeActions))
        publicGoods.append(env.getPublicGood())

        # Track intervention rate for reflective agents
        if agentType == 'reflective':
            interventionRates.append(
                np.mean([agent.getInterventionRate() for agent in agents])
            )

    return {
        'cooperation':   cooperationRates,
        'public_goods':  publicGoods,
        'interventions': interventionRates,
        'agents':        agents,
    }



# Multiple replications
def runMultipleReplications(agentType, envParams, episodes=300, nReplications=1000):
    all_cooperation   = []
    all_public_goods  = []
    all_interventions = [] if agentType == 'reflective' else None

    print(f"    Running {nReplications} replications...", end='', flush=True)

    for rep in range(nReplications):
        if (rep + 1) % 100 == 0:
            print(f" {rep + 1}", end='', flush=True)

        results = runSingleExperiment(agentType, envParams, episodes, seed=rep)
        all_cooperation.append(results['cooperation'])
        all_public_goods.append(results['public_goods'])

        if agentType == 'reflective':
            all_interventions.append(results['interventions'])

    print(" Done!")

    return {
        'cooperation':    np.array(all_cooperation),
        'public_goods':   np.array(all_public_goods),
        'interventions':  np.array(all_interventions) if agentType == 'reflective' else None,
        'n_replications': nReplications,
    }


# Difficulty configurations
def createDifficultyConfigurations():
    base_config = {
        'nAgents':          10,
        'initialWealth':    100.0,
        'contributionCost': 10.0,
        'maxSteps':         200,
    }

    return {
        # Easy configuration: slow decay, low threshold, modest replenishment.
        'Easy': {
            **base_config,
            'initialPublicGood':       40.0,
            'decayRate':               0.97,
            'replenishmentRate':       1.4,
            'benefitMultiplier':       0.15,
            'sustainabilityThreshold': 40.0,
        },

        # Medium configuration: moderate decay, higher threshold, better replenishment.
        'Medium': {
            **base_config,
            'initialPublicGood':       50.0,
            'decayRate':               0.92,
            'replenishmentRate':       2.5,
            'benefitMultiplier':       0.12,
            'sustainabilityThreshold': 50.0,
        },

        # Hard configuration: fast decay, high threshold, weak replenishment.
        'Hard': {
            **base_config,
            'initialPublicGood':       60.0,
            'decayRate':               0.85,
            'replenishmentRate':       4.5,
            'benefitMultiplier':       0.10,
            'sustainabilityThreshold': 60.0,
        },
    }


#  Drawing functions for analysis
def _drawPanel(ax, x, mean_b, std_b, mean_r, std_r,
               col, difficulty, ylabel, threshold=None,
               intervention_mode=False):
    # Baseline
    if intervention_mode:
        ax.plot(x, np.zeros(len(x)), color=COLOUR_BASELINE,
                linewidth=1.5, label='Baseline')
    else:
        ax.plot(x, mean_b, color=COLOUR_BASELINE, linewidth=1.5, label='Baseline')
        ax.fill_between(x, mean_b - std_b, mean_b + std_b,
                        color=COLOUR_BASELINE, alpha=0.2)

    # Reflective
    if mean_r is not None:
        ax.plot(x, mean_r, color=COLOUR_REFLECTIVE, linewidth=1.5, label='Reflective')
        ax.fill_between(x, mean_r - std_r, mean_r + std_r,
                        color=COLOUR_REFLECTIVE, alpha=0.2)

    # Threshold line (only for public good plot)
    if threshold is not None:
        ax.axhline(threshold, color=COLOUR_THRESHOLD, linestyle='--',
                   linewidth=1.5, alpha=0.9, label='Sustainability Threshold')

    ax.set_title(difficulty, fontweight='bold', fontsize=12)
    ax.set_xlabel('Episode', fontsize=10)
    if col == 0:
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# Figure 1 — Cooperation Rate
def plotCooperationRate(results, savePath='difficulty_cooperation.png'):
    n_reps = results['Easy']['baseline']['n_replications']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.text(0.5, 1.01,
             f'Each point averaged across {n_reps} replications (shaded area = ±1 std)',
             ha='center', va='bottom', fontsize=9, style='italic')
    fig.suptitle('Cooperation Rate', fontweight='bold', fontsize=13, y=1.05)

    for col, difficulty in enumerate(DIFFICULTIES):
        b = results[difficulty]['baseline']['cooperation']
        r = results[difficulty]['reflective']['cooperation']
        x = np.arange(b.shape[1])
        _drawPanel(axes[col], x, b.mean(axis=0), b.std(axis=0),
                   r.mean(axis=0), r.std(axis=0),
                   col, difficulty, 'Cooperation Rate')

    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved → {savePath}")
    plt.close(fig)


# Figure 2 — Public Good Level
def plotPublicGood(results, savePath='difficulty_public_good.png'):
    n_reps = results['Easy']['baseline']['n_replications']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.text(0.5, 1.01,
             f'Each point averaged across {n_reps} replications (shaded area = ±1 std)',
             ha='center', va='bottom', fontsize=9, style='italic')
    fig.suptitle('Public Good Level', fontweight='bold', fontsize=13, y=1.05)

    for col, difficulty in enumerate(DIFFICULTIES):
        b         = results[difficulty]['baseline']['public_goods']
        r         = results[difficulty]['reflective']['public_goods']
        threshold = results[difficulty]['config']['sustainabilityThreshold']
        x         = np.arange(b.shape[1])
        _drawPanel(axes[col], x, b.mean(axis=0), b.std(axis=0),
                   r.mean(axis=0), r.std(axis=0),
                   col, difficulty, 'Public Good Level', threshold=threshold)

    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved → {savePath}")
    plt.close(fig)


# Figure 3 — Intervention Rate (Reflective Agents Only)
def plotInterventionRate(results, savePath='difficulty_intervention.png'):
    n_reps = results['Easy']['baseline']['n_replications']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.text(0.5, 1.01,
             f'Each point averaged across {n_reps} replications (shaded area = ±1 std)',
             ha='center', va='bottom', fontsize=9, style='italic')
    fig.suptitle('Intervention Rate', fontweight='bold', fontsize=13, y=1.05)

    for col, difficulty in enumerate(DIFFICULTIES):
        b_shape = results[difficulty]['baseline']['cooperation']
        r       = results[difficulty]['reflective']['interventions']
        x       = np.arange(b_shape.shape[1])
        _drawPanel(axes[col], x, None, None,
                   r.mean(axis=0), r.std(axis=0),
                   col, difficulty, 'Intervention Rate',
                   intervention_mode=True)

    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved → {savePath}")
    plt.close(fig)


# Wrapper to plot all figures
def plotAllFigures(results):
    plotCooperationRate(results,  savePath='difficulty_cooperation.png')
    plotPublicGood(results,       savePath='difficulty_public_good.png')
    plotInterventionRate(results, savePath='difficulty_intervention.png')


# Main function to run all experiments and generate figures
def main():
    configs = createDifficultyConfigurations()
    results = {}

    EPISODES       = 300
    N_REPLICATIONS = 1000

    for difficulty, config in configs.items():
        print(f"\n[{difficulty.upper()}]")
        baseline   = runMultipleReplications('baseline',   config, EPISODES, N_REPLICATIONS)
        reflective = runMultipleReplications('reflective', config, EPISODES, N_REPLICATIONS)

        results[difficulty] = {
            'baseline':   baseline,
            'reflective': reflective,
            'config':     config,
        }

    plotAllFigures(results)
    return results


if __name__ == "__main__":
    results = main()