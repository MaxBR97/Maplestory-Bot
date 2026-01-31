import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pydirectinput
import time

# --- ACTION DEFINITIONS ---
DISCRETE_ACTIONS = [
    'left', 'right', 'up', 'down',
    'alt', 'space',
    ['up', 'alt'], ['down', 'alt'],
    ['left', 'alt'], ['right', 'alt'],
    'idle'
]

def _action_key(action):
    if isinstance(action, list):
        return tuple(action)
    return action

ACTION_TO_IDX = {_action_key(action): idx for idx, action in enumerate(DISCRETE_ACTIONS)}
IDX_TO_ACTION = {idx: action for action, idx in ACTION_TO_IDX.items()}
NUM_ACTIONS = len(DISCRETE_ACTIONS)

ATTACK_KEYS = {'space'}
POTION_KEYS = {}
MOVEMENT_KEYS = {'left', 'right', 'up', 'down'}
COMPOSITE_DELAY_SECONDS = 0.02

KEY_ALIASES = {
    'control': 'ctrl',
    'ctrl': 'ctrl',
    'alt': 'alt',
    'left alt': 'alt',
    'right alt': 'alt',
    'alt gr': 'alt',
    'altgr': 'alt',
    'left menu': 'alt',
    'option': 'alt',
    'space': 'space',
    'home': 'Home',
    'end': 'End',
    'left': 'left',
    'right': 'right',
    'up': 'up',
    'down': 'down',
}


class PolicyNetwork(nn.Module):
    """
    Neural network that takes continuous state space and outputs action probabilities.
    Uses a policy gradient approach (Actor-Critic style).
    """
    def __init__(self, state_dim=12, hidden_dim=128, num_actions=NUM_ACTIONS):
        super(PolicyNetwork, self).__init__()
        
        # Actor (Policy) network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Critic (Value) network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """Returns action logits and state value"""
        action_logits = self.actor(state)
        state_value = self.critic(state)
        return action_logits, state_value


class RLAgent:
    """
    Reinforcement Learning Agent using Policy Gradient (Actor-Critic).
    Continuous state space, discrete action space.
    """
    def __init__(self, state_dim=12, hidden_dim=128, lr=0.001, gamma=0.99, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PolicyNetwork(state_dim, hidden_dim, NUM_ACTIONS).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma  # Discount factor
        self.entropy_coef = entropy_coef  # Entropy bonus for exploration
        
        # Experience buffer for computing returns
        self.memory = {
            'states': deque(maxlen=2048),
            'actions': deque(maxlen=2048),
            'rewards': deque(maxlen=2048),
            'values': deque(maxlen=2048),
            'next_states': deque(maxlen=2048),
            'dones': deque(maxlen=2048),
        }
        
        self.episode_rewards = []
        self.episode_length = 0
        
    def get_state_vector(self, player_coords, monsters, climbing_objects, is_climbing, 
                         need_hp, need_mp, current_actions, damage_count):
        """
        Constructs a continuous state vector from game observations.
        State space: [player_x, player_y, nearest_monster_dx, nearest_monster_dy, 
                      is_climbing, need_hp, need_mp, nearest_climb_dx, nearest_climb_dy,
                      damage_count_norm, hp_need_float, mp_need_float]
        """
        state = np.zeros(12, dtype=np.float32)
        
        if player_coords is None:
            return torch.tensor(state, dtype=torch.float32, device=self.device)
        
        player_x, player_y = player_coords
        state[0] = player_x / 800.0  # Normalize to screen width
        state[1] = player_y / 625.0  # Normalize to screen height
        
        # Nearest monster relative position
        if monsters:
            nearest_monster = min(monsters, key=lambda m: np.hypot(m[0] - player_x, m[1] - player_y))
            state[2] = (nearest_monster[0] - player_x) / 400.0  # Normalize
            state[3] = (nearest_monster[1] - player_y) / 312.5  # Normalize
        
        # Climbing state
        state[4] = float(is_climbing)
        state[5] = float(need_hp)
        state[6] = float(need_mp)
        
        # Nearest climbing object relative position
        if climbing_objects:
            nearest_climb = min(climbing_objects, key=lambda c: np.hypot(c[0] - player_x, c[1] - player_y))
            state[7] = (nearest_climb[0] - player_x) / 400.0
            state[8] = (nearest_climb[1] - player_y) / 312.5
        
        # Damage and resource indicators
        state[9] = min(damage_count / 10.0, 1.0)  # Normalize damage count
        state[10] = float(need_hp)
        state[11] = float(need_mp)
        
        return torch.tensor(state, dtype=torch.float32, device=self.device)
    
    def select_action(self, state_vector, training=True):
        """
        Selects an action based on the policy network.
        During training, samples from the distribution. During inference, uses greedy selection.
        """
        with torch.no_grad():
            state_batch = state_vector.unsqueeze(0)  # Add batch dimension
            action_logits, state_value = self.network(state_batch)
        
        # Convert logits to probabilities
        action_probs = torch.softmax(action_logits, dim=-1).squeeze(0)
        
        if training:
            # Sample action from the distribution
            action_idx = torch.multinomial(action_probs, 1).item()
        else:
            # Greedy action selection
            action_idx = torch.argmax(action_probs).item()
        
        return action_idx, action_probs, state_value.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the experience buffer."""
        self.memory['states'].append(state.cpu().numpy())
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['next_states'].append(next_state.cpu().numpy())
        self.memory['dones'].append(done)
    
    def compute_returns(self):
        """Computes discounted cumulative returns from rewards."""
        if not self.memory['rewards']:
            return None
        
        returns = []
        cumulative_return = 0
        
        # Iterate backwards through rewards
        for reward in reversed(list(self.memory['rewards'])):
            cumulative_return = reward + self.gamma * cumulative_return
            returns.insert(0, cumulative_return)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self):
        """
        Updates the network using the collected experiences.
        Uses Actor-Critic loss with policy gradient and value function approximation.
        """
        if len(self.memory['states']) == 0:
            return 0.0
        
        # Convert memory to tensors
        states = torch.tensor(np.array(list(self.memory['states'])), dtype=torch.float32, device=self.device)
        actions = torch.tensor(list(self.memory['actions']), dtype=torch.long, device=self.device)
        returns = self.compute_returns()
        
        if returns is None:
            return 0.0
        
        # Forward pass
        action_logits, state_values = self.network(states)
        state_values = state_values.squeeze(-1)
        
        # Compute advantage
        advantages = returns - state_values.detach()
        
        # Policy loss (Actor)
        action_log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(selected_log_probs * advantages).mean()
        
        # Value loss (Critic)
        value_loss = torch.nn.functional.smooth_l1_loss(state_values, returns)
        
        # Entropy bonus for exploration
        action_probs = torch.softmax(action_logits, dim=-1)
        entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Clear memory
        self.memory['states'].clear()
        self.memory['actions'].clear()
        self.memory['rewards'].clear()
        self.memory['next_states'].clear()
        self.memory['dones'].clear()
        
        return total_loss.item()
    
    def save_model(self, path):
        """Saves the model to disk."""
        torch.save(self.network.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Loads the model from disk."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


class RLEnvironment:
    """
    Wrapper for the game environment that computes rewards and manages state transitions.
    """
    def __init__(self, reward_config=None):
        self.last_damage_count = 0
        self.step_count = 0
        self.reward_config = reward_config or {
            "damage_reward": 10.0,
            "step_penalty": 0.01,
            "low_hp_penalty": 0.5,
            "low_mp_penalty": 0.25,
        }
        
    def compute_reward(self, damage_count, need_hp, need_mp):
        """
        Computes reward based on configured rewards/penalties.
        """
        reward = 0.0
        
        # Reward for damage dealt
        damage_dealt = max(0, damage_count - self.last_damage_count)
        reward += damage_dealt * self.reward_config["damage_reward"]
        self.last_damage_count = damage_count
        
        # Penalty per step (encourages quick completion)
        reward -= self.reward_config["step_penalty"]
        
        # Extra penalty for low health
        if need_hp:
            reward -= self.reward_config["low_hp_penalty"]
        if need_mp:
            reward -= self.reward_config["low_mp_penalty"]
        
        self.step_count += 1
        return reward
    
    def reset(self):
        """Resets the environment for a new episode."""
        self.last_damage_count = 0
        self.step_count = 0


def action_idx_to_key_sequence(action_idx):
    """
    Converts action index to the corresponding key sequence.
    Some actions require multiple keys (e.g., climbing requires 'alt' + direction).
    """
    action = IDX_TO_ACTION[action_idx]

    if action == 'idle':
        return []
    if isinstance(action, list):
        return action
    elif action in ['up', 'down', 'left', 'right']:
        return [action]
    elif action in ATTACK_KEYS:
        return [action]
    elif action in POTION_KEYS:
        return [action]
    elif action == 'alt':
        return ['alt']
    else:
        return [action]


def execute_action(action_idx, current_actions):
    """
    Executes the action chosen by the RL agent.
    Manages continuous key presses and single-press actions efficiently.
    """
    next_actions = action_idx_to_key_sequence(action_idx)
    next_actions_set = set(next_actions)
    current_actions_set = set(current_actions)
    
    # Release keys that are no longer needed
    keys_to_release = current_actions_set - next_actions_set
    for key in keys_to_release:
        if key != 'idle':
            pydirectinput.keyUp(key)
    
    # Handle potions (single press)
    if any(key in POTION_KEYS for key in next_actions):
        for key in next_actions:
            if key in POTION_KEYS:
                pydirectinput.press(key)
        return ['idle']

    # Handle composite actions with ordered key timing (e.g., down + alt)
    action_name = IDX_TO_ACTION[action_idx]
    if isinstance(action_name, list):
        first_key, second_key = action_name
        pydirectinput.press(first_key)
        time.sleep(COMPOSITE_DELAY_SECONDS)
        if second_key in MOVEMENT_KEYS:
            pydirectinput.keyDown(second_key)
        else:
            pydirectinput.press(second_key)
        return [first_key, second_key]
    
    # Handle continuous actions
    for key in next_actions:
        if key in ATTACK_KEYS or key == 'alt':
            pydirectinput.press(key)
        elif key not in current_actions_set and key != 'idle':
            pydirectinput.keyDown(key)
    
    return next_actions

def normalize_pressed_keys(pressed_keys):
    normalized = set()
    for key in pressed_keys:
        alias = KEY_ALIASES.get(str(key).lower(), str(key))
        normalized.add(alias)
    return normalized

def action_idx_from_pressed_keys(pressed_keys):
    normalized = normalize_pressed_keys(pressed_keys)

    # Check for direction+alt combos (direction first in tuple)
    if 'alt' in normalized and 'down' in normalized:
        action_key = ('down', 'alt')
    elif 'alt' in normalized and 'up' in normalized:
        action_key = ('up', 'alt')
    elif 'alt' in normalized and 'left' in normalized:
        action_key = ('left', 'alt')
    elif 'alt' in normalized and 'right' in normalized:
        action_key = ('right', 'alt')
    elif 'Home' in normalized:
        action_key = 'Home'
    elif 'End' in normalized:
        action_key = 'End'
    elif 'space' in normalized:
        action_key = 'space'
    elif 'left' in normalized:
        action_key = 'left'
    elif 'right' in normalized:
        action_key = 'right'
    elif 'up' in normalized:
        action_key = 'up'
    elif 'down' in normalized:
        action_key = 'down'
    elif 'alt' in normalized:
        action_key = 'alt'
    else:
        action_key = 'idle'
    ans =  ACTION_TO_IDX.get(action_key, ACTION_TO_IDX.get('idle'))
    return ans