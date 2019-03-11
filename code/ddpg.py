import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from multiagent_algorithm import MultiAgentAlgorithm
from model import Actor, Critic

# hyperparameters of DDPG algorithm
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size was 128
GAMMA = 0.99            # discount factor was 0.99
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor was 1e-4 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay was 0
UPDATE_EVERY = 20       # number of episodes between every round of updates
N_UPDATES = 10          # number of batches in a single round of updates

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG(MultiAgentAlgorithm):
    """Interacts with and learns from the environment using a multi-agent version of DDPG algorithm."""

    def __init__(self, state_size, action_size, n_agents, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            n_agents (int): number of agents
            seed (int): random seed
        """

        super().__init__(action_size, n_agents, seed)
        self.state_size = state_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                           lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, n_agents, seed)
        self.noise_scale = 1

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def save_model(self, model_file):
        """Save networks and all other model parameters
        
        Params
        ======
            model_file (string): name of the file that will store the model
        """
        checkpoint = {'noise_scale': self.noise_scale,
                      'actor_local': self.actor_local.state_dict(),
                      'critic_local': self.critic_local.state_dict(),
                      'actor_target': self.actor_target.state_dict(),
                      'critic_target': self.critic_target.state_dict()}
        
        torch.save(checkpoint, model_file)
    
    def load_model(self, model_file):
        """Load networks and all other model parameters
        
        Params
        ======
            model_file (string): name of the file that stores the model
        """
        checkpoint = torch.load(model_file)
        self.noise_scale = checkpoint['noise_scale']
        self.actor_local.load_state_dict(checkpoint['actor_local'])
        self.critic_local.load_state_dict(checkpoint['critic_local'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
            
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn.

        Params
        ======
            states (array_like): current state (for each agent)
            actions (array_like): action taken at the current state (for each agent) 
            rewards (array_like): reward from an action (for each agent)
            next_states (array_like): next state of environment (for each agent)
            dones (array_like): true if the next state is the final one, false otherwise (for each agent)
        """

        # Save experience / reward
        for experience in zip(states, actions, rewards, next_states, dones):
            self.memory.add(*experience)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:    

            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for i in range(N_UPDATES):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, states):
        """Returns actions for given state as per current policy.

        Params
        ======
            states (array_like): current state of each agent
        """

        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()

        # add exploration noise to the actions
        actions += self.noise_scale * self.noise.sample()

        return np.clip(actions, -1, 1)

    def learn(self, experiences, gamma):
        """Update actor and critic networks using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #

        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, n_agents, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.
        
        Params
        ======
            size (int): dimension of noise value
            n_agents (int): number of agents
            seed (int): random seed
            mu (float): mean value of Ornstein-Uhlenbeck process
            theta (float): growth rate of Ornstein-Uhlenbeck process
            sigma (float): standard deviation of Ornstein-Uhlenbeck process
        """
        self.mu = mu * np.ones((n_agents,size))
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.random_sample((x.shape[0],x.shape[1]))
        self.state = x + dx

        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        
        Params
        ======
            state (array_like): current of an agent
            action (float): action taken at the current state
            reward (float): reward from an action
            next_state (array_like): next state of an agent
            dones (boolean): true if the next state is the final one, false otherwise (for each agent)

        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
