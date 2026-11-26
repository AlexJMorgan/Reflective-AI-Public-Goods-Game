import random
from collections import defaultdict
from copy import copy
from typing import override

import numpy as np

from reflective_ai_public_goods_game.agentactions import AgentActions
from reflective_ai_public_goods_game.publicgoodsenv import PublicGoodsState

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
        
    def getAction(self, state: PublicGoodsState, record_history: bool = True) -> AgentActions:
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
        if record_history is True:
            self.actionHistory.append(action)
        return AgentActions(action)
    
    def update(self, state: PublicGoodsState, action: AgentActions, reward, nextState: PublicGoodsState):
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
        
    def _stateToKey(self, state: PublicGoodsState):
        '''
        Convert state to a hashable key for Q-table
        '''
        stateArray = np.array([state.goodsLevel, state.averageConsumptions])
        return tuple(np.round(stateArray, 2))
    
    def getExploitationRate(self, window=100):
        '''
        Calculate exploitation rate over recent actions
        '''
        if len(self.actionHistory) == 0:
            return 0.0
        recent = self.actionHistory[-window:]
        return np.mean(recent)

    def getTotalReward(self) -> float:
        return np.sum(self.rewardHistory) if self.rewardHistory is not None else 0.0
    
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

    def startNewEpisode(self):
        self.actionHistory = []
        self.rewardHistory = []
        
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
        opponentTrackingPoints: list[float],
        agentId: int = 0,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilonMin: float = 0.01,
        epsilonDecay: float = 0.995,
        sustainabilityThreshold: float = 50.0,
        interventionProbability: float = 1.0,
        trackingStep: float = 0.5,
        confidenceStep: float = 0.2
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

        # ToM level 0 model of opponents
        self.opponentTrackingPoints = {k:0.0 for k in sorted(opponentTrackingPoints)}
        self.trackingStep = trackingStep

        # ToM level 1 model of opponents
        self.selfTrackingPoints = {k:0 for k in sorted(opponentTrackingPoints)}
        self.toM1Confidence = 0
        self.lastToM0Prediction: float | None = None
        self.lastToM1Prediction: float | None = None
        self.confidenceStep = confidenceStep
        
    def getActionWithReflection(self, state: PublicGoodsState, publicGood):
        '''
        Two-stage action selection with reflective governor
        
        1. Base policy proposes action
        2. Governor evaluates and potentially overrides
        
        Returns: (finalAction, wasOverridden)
        '''
        
        # Base policy proposes an action
        proposedAction = self.getAction(state)
        
        # Governor evaluates the proposed action
        shouldOverride = self._governorDecision(state, proposedAction)
        
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
    
    def _governorDecision(self, state: PublicGoodsState, proposedAction: AgentActions):
        '''
        Governor decision rule
        
        Override if:
        1. Proposed action and predicted actions of others would take public goods below sustainable threshold
        2. Proposed action is to consume more than minimal amount (not 0)
        
        Implements the hard constraint on selfish behaviour when the resource is near collapse
        
        Returns: True if override, False to approve
        '''

        opponent_prediction = self._predictToM1Action(
            selfTrackingPoints=self.selfTrackingPoints,
            oppTrackingPoints=self.opponentTrackingPoints,
            state=state,
            confidence=self.toM1Confidence
        )
        
        isUnsustainable = state.goodsLevel - (state.usageRates[proposedAction] + opponent_prediction) < self.sustainabilityThreshold
        
        proposedDefection = (proposedAction != AgentActions.MINIMAL)
        
        shouldOverride = isUnsustainable and proposedDefection
        
        return shouldOverride


    def _predictToM0Action(self, trackingPoints: dict[float: float], goodsLevel: float) -> float:
        '''
        Given trackingPoints as a subintentional model of an agents actions,
        predict the consumption of that agent this round using level 0 ToM
        '''
        # Find which two tracked points the goodsLevel falls between
        trackedPoints = sorted(trackingPoints)

        if trackedPoints[0] > goodsLevel:
            return trackingPoints[trackedPoints[0]] # If goods level below all tracked, return lowest tracked value
        else:
            point_idx = 0
            while (point_idx := point_idx + 1) < len(trackedPoints):
                if trackedPoints[point_idx] > goodsLevel:
                    lower_point = trackedPoints[point_idx-1]
                    upper_point = trackedPoints[point_idx]
                    break
            else: # If the while loop finished without breaking
                return trackingPoints[trackedPoints[-1]] # If goods level above all tracked, return highest tracked value

        lower_action, upper_action = trackingPoints[lower_point], trackingPoints[upper_point]
        weightingToLow = (goodsLevel - lower_point)/(upper_point - lower_point)
        return trackingPoints[lower_point] + weightingToLow*(upper_action-lower_action)


    def _predictToM1Action(self, selfTrackingPoints: dict[float, float], oppTrackingPoints: dict[float, float], state: PublicGoodsState, confidence: float) -> float:
        toM0PredictionOfOpp = self._predictToM0Action(oppTrackingPoints, state.goodsLevel)
        toM0PredictionOfSelf = self._predictToM0Action(selfTrackingPoints, state.goodsLevel)
        mockToM1State = copy(state)
        mockToM1State.goodsLevel -= toM0PredictionOfSelf
        toM1PredictionOfOppAction = self.getAction(mockToM1State, record_history=False)
        toM1PredictionOfOppTotal = state.usageRates[toM1PredictionOfOppAction] * (state.nAgents-1)
        weightedToM1Prediction = toM0PredictionOfOpp + confidence*(toM1PredictionOfOppTotal - toM0PredictionOfOpp)
        return weightedToM1Prediction


    @override
    def update(self, state: PublicGoodsState, action: AgentActions, reward, nextState: PublicGoodsState):
        super().update(state, action, reward, nextState)
        self._updateOpponentPrediction(state, action, nextState)


    def _updateOpponentPrediction(self, state: PublicGoodsState, action: AgentActions, new_state: PublicGoodsState):
        total_usage = new_state.lastConsumption
        previous_level = state.goodsLevel
        self_usage = state.usageRates[action]
        opponent_usage = total_usage - self_usage

        self._updateSubIntentionalPredictionModel(
            trackingPoints=self.opponentTrackingPoints,
            previous_level=previous_level,
            usage=opponent_usage,
            trackingStep=self.trackingStep
        )
        self._updateSubIntentionalPredictionModel(
            trackingPoints=self.selfTrackingPoints,
            previous_level=previous_level,
            usage=self_usage,
            trackingStep = self.trackingStep
        )
        self._updateToMConfidence(opponentUsage=opponent_usage)


    def _updateSubIntentionalPredictionModel(
            self,
            trackingPoints: dict[float,float],
            previous_level: float,
            usage: float,
            trackingStep: float
        ):
        # Identify which tracked points to update
        tracked_points = sorted(trackingPoints)
        extreme = False
        if tracked_points[0] > previous_level:
            extreme = True
            extreme_point = tracked_points[0]
        else:
            point_idx = 0
            while (point_idx := point_idx + 1) < len(tracked_points):
                if tracked_points[point_idx] > previous_level:
                    lower_point = tracked_points[point_idx-1]
                    upper_point = tracked_points[point_idx]
                    break
            else: # If the while loop finished without breaking
                extreme = True
                extreme_point = tracked_points[-1]

        # Update tracked points
        if extreme is False:
            lower_action, upper_action = trackingPoints[lower_point], trackingPoints[upper_point]
            lower_error = ((previous_level - lower_point)/(upper_point - lower_point)) * (usage - lower_action)
            upper_error = ((upper_point - previous_level)/(upper_point - lower_point)) * (usage - upper_action)
            trackingPoints[lower_point] += trackingStep * lower_error
            trackingPoints[upper_point] += trackingStep * upper_error
        else:
            extreme_action = trackingPoints[extreme_point]
            extreme_error = usage - extreme_action
            trackingPoints[extreme_point] += trackingStep * extreme_error


    def _updateToMConfidence(self, opponentUsage: float):
        if (tom0 := self.lastToM0Prediction) is not None and (tom1 := self.lastToM1Prediction) is not None:
            if tom0 != tom1:
                target = (opponentUsage - tom0)/(tom1 - tom0)
                if target > 1:
                    target = 1
                elif target < 0:
                    target = 0
                self.toM1Confidence += self.confidenceStep * (target - self.toM1Confidence)


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

    @staticmethod
    def getOpponentTrackingPointsFromUsageCosts(usageCosts: dict[AgentActions, float], sustainability_threshold: float) -> list[float]:
        '''
        Given the usage costs of an environment, return recommended pool levels to track opponent actions at
        Returns levels ordered in a list, without duplicates
        '''
        return sorted(list({
            sustainability_threshold -       usageCosts[AgentActions.MINIMAL],
            sustainability_threshold - 0.5 * usageCosts[AgentActions.MINIMAL],
            sustainability_threshold,
            sustainability_threshold + 0.5 * usageCosts[AgentActions.MINIMAL],
            sustainability_threshold +       usageCosts[AgentActions.MINIMAL],
            sustainability_threshold + 2   * usageCosts[AgentActions.MINIMAL],
            sustainability_threshold +       usageCosts[AgentActions.MODERATE],
            sustainability_threshold + 2   * usageCosts[AgentActions.MODERATE],
            sustainability_threshold +       usageCosts[AgentActions.GREEDY],
            sustainability_threshold + 2   * usageCosts[AgentActions.GREEDY]
        }))

def createAgents(agentType, nAgents, **kwargs):
    '''
    Function to create multiple agents of specified type (baseline or reflective)
        
    Returns: list of agent instances
    '''
        
    if agentType.lower() == 'baseline':
        return [QLearningAgent(agentId=i, **kwargs) for i in range(nAgents)]
    elif agentType.lower() == 'reflective':
        if "opponentTrackingPoints" not in kwargs:
            raise KeyError("Must supply an 'opponentTrackingPoints' list in the kwargs to create reflective agents")
        return [ReflectiveAgent(agentId=i, **kwargs) for i in range(nAgents)]
    else:
        raise ValueError(f"Unknown agent type: {agentType}")