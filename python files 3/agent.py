# agent.py
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical

# PolicyNetwork class remains the same...
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=1296, hidden1=256, hidden2=128, num_actions=35):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        # state: Tensor of shape [batch_size, input_dim]
        return self.model(state)


class PGAgent:
    def __init__(self, num_states=1296, num_actions=35, lr=1e-3, gamma=0.99):
        # --- ADD THESE LINES ---
        self.num_states = num_states
        self.num_actions = num_actions # Also store num_actions, might be useful later
        # -----------------------

        self.gamma = gamma
        self.policy_net = PolicyNetwork(input_dim=num_states,
                                        hidden1=256,
                                        hidden2=128,
                                        num_actions=num_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def choose_action(self, state):
        """
        state: 1D array or list of length num_states (one-hot)
        returns: action index, log probability of that action
        """
        # Check if state length matches num_states if needed (optional validation)
        # if len(state) != self.num_states:
        #    raise ValueError(f"Input state length {len(state)} does not match agent's num_states {self.num_states}")
        
        s = torch.FloatTensor(state).unsqueeze(0)    # → [1, num_states]
        probs = self.policy_net(s).squeeze(0)        # → [num_actions]
        prob = np.random.random()
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards):
        """
        Compute discounted returns for a list of rewards.
        """
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        # Normalize returns (optional but often helpful)
        returns_tensor = torch.FloatTensor(returns)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9) # Add epsilon for stability
        return returns_tensor # Return the normalized tensor


    def update_policy(self, trajectory):
        """
        trajectory: list of (state, action, reward, log_prob) tuples for one episode
        """
        states, actions, rewards, log_probs = zip(*trajectory)

        # Convert states list of lists/arrays into a tensor if needed by network
        # state_tensor = torch.FloatTensor(states) # If network expects batch

        returns = self.compute_returns(rewards)        # [T] (Now normalized)
        log_probs = torch.stack(log_probs)             # [T]

        # Ensure returns and log_probs have the same shape
        if returns.shape != log_probs.shape:
             # This might happen if compute_returns or log_prob collection is inconsistent.
             # Add print statements here to debug shapes if the error persists after normalization
             print(f"Shape mismatch: returns {returns.shape}, log_probs {log_probs.shape}")
             # Handle error appropriately, e.g., raise ValueError or attempt reshaping if logical

        loss = - (log_probs * returns).sum()           # policy gradient loss (using normalized returns)
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()