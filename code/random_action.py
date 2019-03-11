import numpy as np
from multiagent_algorithm import MultiAgentAlgorithm

class RandomAction(MultiAgentAlgorithm):
    """Interacts with multi-agent environment using random actions."""

    def __init__(self, action_size, n_agents, seed):
        """Initializes a RandomAgent object.
        
        Params
        ======
            action_size (int): number of possible actions
            n_agents (int): number of agents
            seed (int): random seed
        """
        super().__init__(action_size, n_agents, seed)

    def step(self, state, action, reward, next_state, done):
        """Learns from interaction with environment.
        
        Params
        ======
            states (array_like): current state of each agent
            actions (array_like): action taken at the current state of each agent 
            rewards (array_like): reward from an action taken by each agent
            next_states (array_like): next state of environment of each agent
            dones (array_like): for each agent, true if the next state is the final one, false otherwise
        """
        pass

    def act(self, state):
        """Finds the best action for a given state of each agent.
        
        Params
        ======
            states (array_like): current state of each agent
            
        Returns
        =======
            actions (array_like): action that should be taken at the current state of each agent
        """
        actions = np.random.randn(self.n_agents, self.action_size)
        actions = np.clip(actions, -1, 1)   
        return actions
        
    


