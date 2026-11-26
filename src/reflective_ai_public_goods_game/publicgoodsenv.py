import numpy as np

from reflective_ai_public_goods_game.agentactions import AgentActions


class PublicGoodsState:
    def __init__(
            self,
            goodsLevel: float,
            averageConsumptions: float,
            lastConsumption: float,
            usageRates: dict[AgentActions, float],
            nAgents: int
        ):
        self.goodsLevel = goodsLevel
        self.averageConsumptions = averageConsumptions
        self.lastConsumption = lastConsumption
        self.usageRates = usageRates
        self.nAgents = nAgents


class PublicGoodsEnv:
    '''
    Public Goods Game Environment
    Wealth all contained in the replenishing pool, agents must decide how much to use each timestep
    '''
    
    def __init__(
        self,
        nAgents = 3,
        minimumUsage = 0.0, # Smallest amount an agent can use in a time step
        moderateUsage = 2.0, # Medium amount for an agent to use in a time step
        greedyUsage = 8.0, # Large amount for an agent to use in a time step
        initialPublicGood = 100.0, # Initial amount of public good in the shared pool
        replenishmentRate = 8.0, # Amount added to the public good pool each time step
        sustainabilityThreshold = 50.0, # Threshold to keep the public good total above
        maxSteps = 200 # Maximum number of steps per episode
    ):
        self.nAgents = nAgents
        self.usageRates = {
            AgentActions.MINIMAL: minimumUsage,
            AgentActions.MODERATE: moderateUsage,
            AgentActions.GREEDY: greedyUsage
        }
        self.initialPublicGood = initialPublicGood
        self.replenishmentRate = replenishmentRate
        self.sustainabilityThreshold = sustainabilityThreshold
        self.maxSteps = maxSteps
        
        # Dynamic environment state variables
        self.utility = np.zeros(nAgents)
        self.publicGood = 0.0
        self.stepCount = 0
        self.lastActions = np.zeros(nAgents)
        
        # History tracking for analysis
        self.publicGoodHistory = []
        self.utilityHistory = []
        self.consumptionHistory = []
        
    def reset(self):
        '''
        Start a new episode
        '''
        self.stepCount = 0
        self.utility = np.zeros(self.nAgents)
        self.publicGood = self.initialPublicGood 
        self.lastActions = np.zeros(self.nAgents)
        
        self.publicGoodHistory = [self.publicGood]
        self.utilityHistory = [self.utility.copy()]
        self.consumptionHistory = []
        
        return self._getState()
    
    def step(self, actions) -> tuple[PublicGoodsState, np.ndarray, bool, dict]:
        '''
        Execute one time step of the environment
        
        Actions: array of size nAgents, where each element is an AgentActions value
        Returns: nextState, rewards, done, info
        '''
        
        self.stepCount += 1
        actions = np.array(actions)
        self.lastActions = actions
        
        # Count how much resource was used this step
        usageCosts = actions.copy()
        for action in self.usageRates:
            usageCosts[usageCosts == action] = self.usageRates[action]
        usageCosts = np.astype(usageCosts, np.float64)
        totalUsage = np.sum(usageCosts)
        self.consumptionHistory.append(totalUsage)
        
        # Update public good: replenishment and consumption
        self.publicGood = self.publicGood + self.replenishmentRate - totalUsage
        
        # # Health factor scales benefits when resource is degraded- Disabled
        # healthFactor = min(1.0, self.publicGood / self.sustainabilityThreshold)
        
        # # Shared benefit received by all agents
        # benefit = self.benefitMultiplier * healthFactor * self.publicGood
        
        rewards = np.zeros(self.nAgents)
        
        # Update individual agent wealth and compute rewards
        self.utility += usageCosts
        rewards = usageCosts
    
        # Record histories for analysis
        self.publicGoodHistory.append(self.publicGood)
        self.utilityHistory.append(self.utility.copy())
        
        # Check for collapse condition
        collapsed = self.publicGood < (self.sustainabilityThreshold * 0.2)
        
        # Episode ends if resource collapses or max steps reached
        done = collapsed or (self.stepCount >= self.maxSteps)
        
        nextState = self._getState()
        
        info = {
            'publicGood': self.publicGood,
            'totalUsage': totalUsage,
            'avgUtility': np.mean(self.utility),
            'totalUtility': np.sum(self.utility),
            'isSustainable': self.publicGood >= self.sustainabilityThreshold,
            'collapsed': collapsed,
        }
        return nextState, rewards, done, info
    
    def _getState(self) -> PublicGoodsState:
        '''
        State representation for agents
        
        Returns: PublicGoodsState
        '''
        
        # Normalise public good relative to initial value
        normalizedPublicGood = self.publicGood / self.initialPublicGood
        
        # Compute average recent consumption rate and get last total consumption
        if len(self.consumptionHistory) > 0:
            recentWindow = min(10, len(self.consumptionHistory))
            avgConsumptions = np.mean(self.consumptionHistory[-recentWindow:]) / self.nAgents
            lastConsumption = self.consumptionHistory[-1]
        else:
            avgConsumptions = 0.0
            lastConsumption = 0.0
        
        return PublicGoodsState(
            goodsLevel=self.publicGood,
            averageConsumptions=avgConsumptions,
            lastConsumption=lastConsumption,
            usageRates=self.usageRates,
            nAgents=self.nAgents
        )
    
    def getAgentState(self, agentId):
        '''
        Get individual agent's state representation
        Shared state and agent-specific wealth
        '''
        
        normalisedPublicGood, avgConsumptions = self._getState()
    
        return np.array([normalisedPublicGood, avgConsumptions, self.utility[agentId]])
    
    def isSustainable(self):
        '''
        Check if public good is above the sustainability threshold
        '''
        return self.publicGood >= self.sustainabilityThreshold
    
    def getPublicGood(self):
        '''
        Get current public good value
        '''
        return self.publicGood

    def getTotalUtility(self):
        '''
        Get current total utility of agents
        '''
        return np.sum(self.utility)
    
    def getMetrics(self):
        '''
        Return metric summary
        '''
        return {
            'final_public_good': self.publicGood,
            'avg_final_utiltiy': np.mean(self.utility),
            'total_utiltiy': np.sum(self.utiltiy),
            'steps_survived': self.stepCount,
            'avg_consumption': np.mean(self.consumptionHistory) if self.consumptionHistory else 0.0,
            'public_good_history': self.publicGoodHistory,
            'utiltiy_history': self.utilityHistory,
            'consumption_history': self.consumptionHistory
        }