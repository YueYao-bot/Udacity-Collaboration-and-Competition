import numpy as np
import random
from collections import namedtuple, deque
import copy
from DDPG_Net import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 5e-2  # for soft update of target parameters
LR = 5e-4  # learning rate
EPSILON = 1.0           # epsilon noise parameter
EPSILON_DECAY = 1e-6    # decay parameter of epsilon
UPDATE_EVERY = 2  # how often to update the network
UPDATE_TIMES = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Initialize Networks
        self.actor_net_local = Actor(state_size, action_size, seed).to(device)
        self.actor_net_target = Actor(state_size, action_size, seed).to(device)
        self.critic_net_local = Critic(state_size, action_size, seed).to(device)
        self.critic_net_target = Critic(state_size, action_size, seed).to(device)

        self.optimizer_actor = optim.Adam(self.actor_net_local.parameters(), lr=LR)
        self.optimizer_critic = optim.Adam(self.critic_net_local.parameters(), lr=LR)

        self.noise = OUNoise(action_size, seed)
        self.epsilon = EPSILON

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.hard_update(self.actor_net_local, self.actor_net_target)
        self.hard_update(self.critic_net_local, self.critic_net_target)

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        #for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # If enough samples are available in memory, get random subset and learn
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(UPDATE_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_net_local.eval()
        with torch.no_grad():
            action = self.actor_net_local(state).cpu().data.numpy()
        self.actor_net_local.train()
        # return best action according to actor
        return np.clip(action + self.epsilon * self.noise.sample(), -1, 1)

    def target_act(self, state):
        """Returns actions for given state as per current target policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_net_target.eval()
        with torch.no_grad():
            action = self.actor_net_target(state).cpu().data.numpy()
        self.actor_net_target.train()
        # return best action according to actor
        return np.clip(action + self.epsilon * self.noise.sample(), -1, 1)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # -------------------- update critic -------------------- #
        actions_next = self.actor_net_target(next_states)
        Q_targets_next = self.critic_net_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_net_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net_local.parameters(), 1)
        self.optimizer_critic.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        action_pred = self.actor_net_local(states)
        actor_loss = -self.critic_net_local(states, action_pred).mean()

        # Minimize the loss
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_net_local, self.critic_net_target, TAU)
        self.soft_update(self.actor_net_local, self.actor_net_target, TAU)

        # ---------------------------- update noise ---------------------------- #
        self.epsilon -= EPSILON_DECAY
        self.noise.reset()


    def soft_update(self, local_net, target_net, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_net (PyTorch model): weights will be copied from
            target_net (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, local_net, target_net):
        """hard update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_net (PyTorch model): weights will be copied from
            target_net (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(local_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
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
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)