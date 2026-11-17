import random
from collections import defaultdict

import numpy as np

from reflective_ai_public_goods_game.agentactions import AgentActions

class QLearningAgent:
    '''
    Standard Q-Learning Agent (baseline control group)
    
    This agent uses epsilon-greedy Q-learning to maximise individual wealth without reflective governance
    '''
    
    def __init__(
        self,
        agentId = 0,
        # Learning Rate: how strongly new information updates Q-values
        alpha = 0.1,
        # Discount factor: importance of future rewards
        gamma = 0.95,
        # Exploration rate for epsilon-greedy policy
        epsilon = 1.0,
        epsilonMin = 0.01,
        epsilonDecay = 0.995
    ):
        self.agentId = agentId
        self.nActions = len(AgentActions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        
        # Q-table mapping states to action value array
        # deafultdict ensures unseen states are initialised with zero values
        self.qTable = defaultdict(lambda: np.zeros(self.nActions))
        
        # Track agent behaviour and rewards over time
        self.actionHistory = []
        self.rewardHistory = []
        
    def getAction(self, state) -> AgentActions:
        '''
        Epsilon-greedy action selection
        
        Returns: action (0 = minimal consumption, 1 = moderate consumption, 2 = greedy consumption)
        '''
        
        # Convert environment state to hashable key
        stateKey = self._stateToKey(state)
        
        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            action = random.randint(0, self.nActions - 1)
        else:
            action = int(np.argmax(self.qTable[stateKey]))
        
        # Record actions for later analysis
        self.actionHistory.append(action)
        return AgentActions(action)
    
    def update(self, state, action: AgentActions, reward, nextState):
        '''
        Q-learning update using Bellman equation
        
        Q(s,a) = Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        '''
        
        # Convert states to Q-table keys
        stateKey = self._stateToKey(state)
        nextStateKey = self._stateToKey(nextState)
        actionKey = action.value

        # Estimate optimal future value
        bestNextQ = np.max(self.qTable[nextStateKey])
        
        # Compute TD target and TD error
        tdTarget = reward + self.gamma * bestNextQ
        tdError = tdTarget - self.qTable[stateKey][actionKey]
        
        # Update Q-value
        self.qTable[stateKey][actionKey] += self.alpha * tdError
        
        # Record reward
        self.rewardHistory.append(reward)
        
    def decayEpsilon(self):
        '''
        Reduce exploration rate over time
        '''
        self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)
        
    def _stateToKey(self, state):
        '''
        Convert state to a hashable key for Q-table
        '''
        return tuple(np.round(state, 2))
    
    def getExploitationRate(self, window=100):
        '''
        Calculate exploitation rate over recent actions
        '''
        if len(self.actionHistory) == 0:
            return 0.0
        recent = self.actionHistory[-window:]
        return np.mean(recent)
    
    def getStats(self):
        '''
        Return agent statistics
        '''
        return {
            'agent_id': self.agentId,
            'total_actions': len(self.actionHistory),
            'consumption_rate': self.getConsumptionRate(),
            'avg_reward': np.mean(self.rewardHistory) if self.rewardHistory else 0.0,
            'epsilon': self.epsilon,
            'q_table_size': len(self.qTable)
        }
        
class ReflectiveAgent(QLearningAgent):
    '''
    Reflective Q-Learning Agent with dual-layer architecture
    
    Architecture:
    - Base Layer: Standard Q-learning for individual wealth maximization
    - Reflective Governor: Monitors environment and overrides when needed
    
    Governor decision rule:
    a_executed = {
        a_proposed if G(s, a_proposed, g, c) = approve
        a_alternate if G(s, a_proposed, g, c) = override
    }
    
    The governor implements threshold-based intervention:
    if G_t < θ and the base policy proposes to free ride, force cooperation instead
    '''
    
    def __init__(
        self,
        agentId = 0,
        alpha = 0.1,
        gamma = 0.95,
        epsilon = 1.0,
        epsilonMin = 0.01,
        epsilonDecay = 0.995,
        sustainabilityThreshold = 50.0,
        interventionProbability = 1.0,
    ):
        super().__init__(
            agentId,
            alpha,
            gamma,
            epsilon,
            epsilonMin,
            epsilonDecay
        )
        
        # Minimum acceptable level of the public good
        self.sustainabilityThreshold = sustainabilityThreshold
        
        # Probability that the governor actually intervenes when triggered
        self.interventionProbability = interventionProbability
        
        # Track governance behaviour
        self.interventionHistory = []
        self.overrideCount = 0
        
    def getActionWithReflection(self, state, publicGood):
        '''
        Two-stage action selection with reflective governor
        
        1. Base policy proposes action
        2. Governor evaluates and potentially overrides
        
        Returns: (finalAction, wasOverridden)
        '''
        
        # Base policy proposes an action
        proposedAction = self.getAction(state)
        
        # Governor evaluates the proposed action
        shouldOverride = self._governorDecision(state, proposedAction, publicGood)
        
        # Apply override if triggered
        if shouldOverride and random.random() < self.interventionProbability:
            finalAction = AgentActions.MINIMAL
            self.overrideCount += 1
            overridden = True
        else:
            finalAction = proposedAction
            overridden = False
        
        # Log intervention decision    
        self.interventionHistory.append(overridden)
        
        # Correct action history if overriden
        if overridden:
            self.actionHistory[-1] = finalAction.value
            
        return finalAction, overridden
    
    def _governorDecision(self, state, proposedAction: AgentActions, publicGood):
        '''
        Governor decision rule
        
        Override if:
        1. Public good is below sustainability threshold
        2. Proposed action is to consume more than minimal amount (not 0)
        
        Implements the hard constraint on selfish behaviour when the resource is near collapse
        
        Returns: True if override, False to approve
        '''
        
        isUnsustainable = publicGood < self.sustainabilityThreshold
        
        proposedDefection = (proposedAction != AgentActions.MINIMAL)
        
        shouldOverride = isUnsustainable and proposedDefection
        
        return shouldOverride

    def getInterventionRate(self, window = 100):
        '''
        Calculate the rate of governor interventions
        '''
        
        if len(self.interventionHistory) == 0:
            return 0.0
        recent = self.interventionHistory[-window:]
        return np.mean(recent)
    
    def getStats(self):
        '''
        Return extended statistics including governance metrics
        '''
        baseStats = super().getStats()
        baseStats.update({
            'intervention_count': self.overrideCount,
            'intervention_rate': self.getInterventionRate(),
            'total_decisions': len(self.interventionHistory)
        })
        return baseStats

def createAgents(agentType, nAgents, **kwargs):
    '''
    Function to create multiple agents of specified type (baseline or reflective)
        
    Returns: list of agent instances
    '''
        
    if agentType.lower() == 'baseline':
        return [QLearningAgent(agentId=i, **kwargs) for i in range(nAgents)]
    elif agentType.lower() == 'reflective':
        return [ReflectiveAgent(agentId=i, **kwargs) for i in range(nAgents)]
    else:
        raise ValueError(f"Unknown agent type: {agentType}")