import numpy as np

class PublicGoodsEnv:
    '''
    Public Goods Game Environment
    Agents start with initial wealth (tokens) and must decide whether to contribute to a public good or free ride
    '''
    
    def __init__(
        self,
        nAgents = 3,
        initialWealth = 100.0,
        contributionCost = 10.0,
        initialPublicGood = 100.0,
        decayRate = 0.98, # Proportion of the public good retained at each step
        replenishmentRate = 8.0, # Amount added to the public good per contributing agent
        benefitMultiplier = 0.15, # Determines how much the public good benefits agents
        sustainabilityThreshold = 50.0,
        maxSteps = 200 # Maximum number of steps per episode
    ):
        self.nAgents = nAgents
        self.initialWealth = initialWealth
        self.contributionCost = contributionCost
        self.initialPublicGood = initialPublicGood
        self.decayRate = decayRate
        self.replenishmentRate = replenishmentRate
        self.benefitMultiplier = benefitMultiplier
        self.sustainabilityThreshold = sustainabilityThreshold
        self.maxSteps = maxSteps
        
        # Dynamic environment state variables
        self.wealth = np.zeros(nAgents)
        self.publicGood = 0.0
        self.stepCount = 0
        self.lastActions = np.zeros(nAgents)
        
        # History tracking for analysis
        self.publicGoodHistory = []
        self.wealthHistory = []
        self.contributionHistory = []
        
    def reset(self):
        '''
        Start a new episode
        '''
        self.stepCount = 0
        self.wealth = np.full(self.nAgents, self.initialWealth)
        self.publicGood = self.initialPublicGood
        self.lastActions = np.zeros(self.nAgents)
        
        self.publicGoodHistory = [self.publicGood]
        self.wealthHistory = [self.wealth.copy()]
        self.contributionHistory = []
        
        return self._getState()
    
    def step(self, actions):
        '''
        Execute one time step of the environment
        
        Actions: array of size nAgents, where each element is 1 (contribute) or 0 (free ride)
        Returns: nextState, rewards, done, info
        '''
        
        self.stepCount += 1
        actions = np.array(actions)
        self.lastActions = actions
        
        # Counting how many agents contributed this step
        numContributing = np.sum(actions)
        self.contributionHistory.append(numContributing)
        
        # Update public good: decay and replenishment from contributions
        self.publicGood = (self.decayRate * self.publicGood) + (self.replenishmentRate * numContributing)
        
        # Health factor scales benefits when resource is degraded
        healthFactor = min(1.0, self.publicGood / self.sustainabilityThreshold)
        
        # Shared benefit received by all agents
        benefit = self.benefitMultiplier * healthFactor * self.publicGood
        
        rewards = np.zeros(self.nAgents)
        
        # Update individual agent wealth and compute rewards
        for i in range(self.nAgents):
            oldWealth = self.wealth[i]
            
            if actions[i] == 1:
                # Contributors pay a cost but receive the shared benefit
                self.wealth[i] = self.wealth[i] - self.contributionCost + benefit
            else:
                # Free riders receive benefit without paying cost
                self.wealth[i] = self.wealth[i] + benefit
            
            # Reward is change in wealth
            rewards[i] = self.wealth[i] - oldWealth
            # Wealth is lower-bounded at zero
            self.wealth[i] = max(0.0, self.wealth[i])
    
        # Record histories for analysis
        self.publicGoodHistory.append(self.publicGood)
        self.wealthHistory.append(self.wealth.copy())
        
        # Check for collapse condition
        collapsed = self.publicGood < (self.sustainabilityThreshold * 0.2)
        
        # Episode ends if resource collapses or max steps reached
        done = collapsed or (self.stepCount >= self.maxSteps)
        
        nextState = self._getState()
        
        info = {
            'publicGood': self.publicGood,
            'numContributing': numContributing,
            'avgWealth': np.mean(self.wealth),
            'totalWealth': np.sum(self.wealth),
            'isSustainable': self.publicGood >= self.sustainabilityThreshold,
            'collapsed': collapsed,
            'benefit': benefit,
            'healthFactor': healthFactor
        }
        return nextState, rewards, done, info
    
    def _getState(self):
        '''
        State representation for agents
        
        Returns: [normalizedPublicGood, avgContributions]
        '''
        
        # Normalise public good relative to initial value
        normalizedPublicGood = self.publicGood / self.initialPublicGood
        
        # Compute average recent contribution rate
        if len(self.contributionHistory) > 0:
            recentWindow = min(10, len(self.contributionHistory))
            avgContributions = np.mean(self.contributionHistory[-recentWindow:]) / self.nAgents
        else:
            avgContributions = 0.0
        
        return np.array([normalizedPublicGood, avgContributions])
    
    def getAgentState(self, agentId):
        '''
        Get individual agent's state representation
        Shared state and agent-specific wealth
        '''
        
        sharedState = self._getState()
        
        normalizedWealth = self.wealth[agentId] / self.initialWealth
        
        return np.array([sharedState[0], sharedState[1], normalizedWealth])
    
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
    
    def getMetrics(self):
        '''
        Return metric summary
        '''
        return {
            'final_public_good': self.publicGood,
            'avg_final_wealth': np.mean(self.wealth),
            'total_wealth': np.sum(self.wealth),
            'steps_survived': self.stepCount,
            'avg_contributions': np.mean(self.contributionHistory) if self.contributionHistory else 0.0,
            'public_good_history': self.publicGoodHistory,
            'wealth_history': self.wealthHistory,
            'contribution_history': self.contributionHistory
        }