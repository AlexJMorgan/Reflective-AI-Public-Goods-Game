from typing import Literal
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from reflective_ai_public_goods_game.publicgoodsenv import PublicGoodsEnv
from reflective_ai_public_goods_game.agents import createAgents, QLearningAgent

'''
Run experiment with varying difficulty levels.
Difficulty is calibrated by adjusting environment parameters to create different sustainability thresholds and decay dynamics.
'''



# Colour palette for consistent plotting
COLOUR_BASELINE   = '#e74c3c'
COLOUR_REFLECTIVE = '#27ae60'
COLOUR_THRESHOLD  = '#f0c419'

DIFFICULTIES = ['Easy', 'Medium', 'Hard']

# Single episode
def runSingleEpisode(
        env: PublicGoodsEnv,
        agents: list[QLearningAgent],
        agentType: Literal["baseline", "reflective"],
        tracking_stats: bool = False
    ):
    state = env.reset()
    for agent in agents:
        agent.startNewEpisode()
    done = False
    if tracking_stats is True:
        running_agent_utilities: list[list[float]] = []
        running_total_utility: list[float] = []
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
        if tracking_stats is True:
            running_agent_utilities.append([agent.getTotalReward() for agent in agents])
            running_total_utility.append(env.getTotalUtility())
    if tracking_stats is True:
        running_agent_utilities_arr = np.array(running_agent_utilities).transpose()
        return running_agent_utilities_arr, np.array(running_total_utility)


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
    totalUtilities  = []
    publicGoods       = []
    interventionRates = [] if agentType == 'reflective' else None

    for _episode in range(episodes):
        # Run one episode
        runSingleEpisode(env=env, agents=agents, agentType=agentType)
    
        # Decay epsilon after each episode
        for agent in agents:
            agent.decayEpsilon()

        totalUtilities.append(env.getTotalUtility())
        publicGoods.append(env.getPublicGood())

        # Track intervention rate for reflective agents
        if agentType == 'reflective':
            interventionRates.append(
                np.mean([agent.getInterventionRate() for agent in agents])
            )

    return {
        'total_utility': totalUtilities,
        'public_goods':  publicGoods,
        'interventions': interventionRates,
        'agents':        agents,
        'env':           env,
    }



# Multiple replications
def runMultipleReplications(agentType, envParams, episodes=300, nReplications=1000):
    all_total_utilities   = []
    all_public_goods  = []
    all_interventions = [] if agentType == 'reflective' else None

    print(f"    Running {nReplications} replications...", end='', flush=True)

    for rep in range(nReplications):
        if (rep + 1) % 100 == 0:
            print(f" {rep + 1}", end='', flush=True)

        results = runSingleExperiment(agentType, envParams, episodes, seed=rep)
        all_total_utilities.append(results['total_utility'])
        all_public_goods.append(results['public_goods'])

        if agentType == 'reflective':
            all_interventions.append(results['interventions'])

        if rep == nReplications-1:
            episode_agent_utility, episode_total_utility = runSingleEpisode(
                env=results['env'], agents=results['agents'], agentType=agentType, tracking_stats=True
            )

    print(" Done!")

    return {
        'total_utility':         np.array(all_total_utilities),
        'public_goods':          np.array(all_public_goods),
        'interventions':         np.array(all_interventions) if agentType == 'reflective' else None,
        'episode_agent_utility': episode_agent_utility,
        'episode_total_utility': episode_total_utility,
        'n_replications':        nReplications,
    }


# Difficulty configurations
def createDifficultyConfigurations():
    base_config = {
        'nAgents':        2,
        'minimumUsage':   0.0,
        'moderateUsage':  2.0,
        'greedyUsage':    8.0,
        'maxSteps':       200,
    }

    return {
        # Easy configuration: low greedy usage, higher replenishment, low threshold.
        'Easy': base_config | {
            'initialPublicGood':       40.0,
            'replenishmentRate':       4.0,
            'greedyUsage':             4.0,
            'sustainabilityThreshold': 20.0,
        },

        # Medium configuration: average replenishment, average threshold.
        'Medium': base_config | {
            'initialPublicGood':       50.0,
            'replenishmentRate':       2.5,
            'sustainabilityThreshold': 30.0,
        },

        # Hard configuration: weak replenishment, high threshold.
        'Hard': {
            **base_config,
            'initialPublicGood':       60.0,
            'replenishmentRate':       1.8,
            'sustainabilityThreshold': 40.0,
        },
    }


#  Drawing functions for analysis
def _drawPanel(ax, x, mean_b, std_b, mean_r, std_r,
               col, difficulty, ylabel, xlabel='Episode', threshold=None,
               intervention_mode=False):
    # Baseline
    if intervention_mode:
        ax.plot(x, np.zeros(len(x)), color=COLOUR_BASELINE,
                linewidth=1.5, label='Baseline')
    else:
        ax.plot(x, mean_b, color=COLOUR_BASELINE, linewidth=1.5, label='Baseline')
        if std_b is not None:
            ax.fill_between(x, mean_b - std_b, mean_b + std_b,
                            color=COLOUR_BASELINE, alpha=0.2)

    # Reflective
    if mean_r is not None:
        ax.plot(x, mean_r, color=COLOUR_REFLECTIVE, linewidth=1.5, label='Reflective')
        if std_r is not None:
            ax.fill_between(x, mean_r - std_r, mean_r + std_r,
                            color=COLOUR_REFLECTIVE, alpha=0.2)

    # Threshold line (only for public good plot)
    if threshold is not None:
        ax.axhline(threshold, color=COLOUR_THRESHOLD, linestyle='--',
                   linewidth=1.5, alpha=0.9, label='Sustainability Threshold')

    ax.set_title(difficulty, fontweight='bold', fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    if col == 0:
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# Figure 1 — Total Utility
def plotTotalUtility(results, savePath='difficulty_utility.png'):
    n_reps = results['Easy']['baseline']['n_replications']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.text(0.5, 1.01,
             f'Each point averaged across {n_reps} replications (shaded area = ±1 std)',
             ha='center', va='bottom', fontsize=9, style='italic')
    fig.suptitle('Total Utility', fontweight='bold', fontsize=13, y=1.05)

    for col, difficulty in enumerate(DIFFICULTIES):
        b = results[difficulty]['baseline']['total_utility']
        r = results[difficulty]['reflective']['total_utility']
        x = np.arange(b.shape[1])
        _drawPanel(axes[col], x, b.mean(axis=0), b.std(axis=0),
                   r.mean(axis=0), r.std(axis=0),
                   col, difficulty, 'Total Utility')

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
        b_shape = results[difficulty]['baseline']['total_utility']
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

def plotSingleEpisodeAgentUtility(results, savePath='difficulty_episode_agent_utility.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Single Episode Agent Utility', fontweight='bold', fontsize=13, y=1.05)

    for col, difficulty in enumerate(DIFFICULTIES):
        ax = axes[col]

        baseline_results = results[difficulty]['baseline']['episode_agent_utility']
        reflective_results = results[difficulty]['reflective']['episode_agent_utility']
        longest_episode = max(baseline_results.shape[1], reflective_results.shape[1])
        padded_baseline_results = np.pad(baseline_results, ((0,0),(0,longest_episode-baseline_results.shape[1])), 'edge')
        padded_reflective_results = np.pad(reflective_results, ((0,0),(0,longest_episode-reflective_results.shape[1])), 'edge')

        for i, agent_results in enumerate(padded_baseline_results):
            ax.plot(agent_results, linestyle='dashed', label=f'Baseline agent {i}')
        for i, agent_results in enumerate(padded_reflective_results):
            ax.plot(agent_results, linestyle='solid', label=f'Reflective agent {i}')
        ax.set_title(difficulty, fontweight='bold', fontsize=12)
        ax.set_xlabel('Timestep', fontsize=10)
        if col == 0:
            ax.set_ylabel('Utility', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved → {savePath}")
    plt.close(fig)

def plotSingleEpisodeTotalUtility(results, savePath='difficuly_episode_total_utility.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Single Episode Total Utility', fontweight='bold', fontsize=13, y=1.05)

    for col, difficulty in enumerate(DIFFICULTIES):
        b = results[difficulty]['baseline']['episode_total_utility']
        r = results[difficulty]['reflective']['episode_total_utility']
        x = np.arange(r.shape[0])
        b = np.pad(b, (0, r.shape[0] - b.shape[0]), mode="edge")
        _drawPanel(axes[col], x, b, None, r, None, col, difficulty, ylabel='Total Utility', xlabel='Timestep')

    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    print(f"Saved → {savePath}")
    plt.close(fig)
    

# Wrapper to plot all figures
def plotAllFigures(results):
    plotTotalUtility(results,              savePath='difficulty_utility.png')
    plotPublicGood(results,                savePath='difficulty_public_good.png')
    plotInterventionRate(results,          savePath='difficulty_intervention.png')
    plotSingleEpisodeTotalUtility(results, savePath='difficulty_episode_total_utility.png')
    plotSingleEpisodeAgentUtility(results, savePath='difficulty_episode_agent_utility.png')

# Main function to run all experiments and generate figures
def main():
    configs = createDifficultyConfigurations()
    results = {}

    EPISODES       = 50
    N_REPLICATIONS = 100

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