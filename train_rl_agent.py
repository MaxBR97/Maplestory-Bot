import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import cv2
from maple_env import MapleStoryEnv

# --- HYPERPARAMETERS ---
BATCH_SIZE = 8
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 5000
TARGET_UPDATE = 1
MEMORY_SIZE = 50
LEARNING_RATE = 1e-3
NUM_EPISODES = 10000
LOOP_DELAY = 0.1  # Requested delay per loop iteration (seconds)
MAX_STEPS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NEURAL NETWORK ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Input dim is larger now due to multiple monsters
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

def train():
    env = MapleStoryEnv()
    
    n_actions = env.action_dim
    n_states = env.state_dim
    
    policy_net = DQN(n_states, n_actions).to(device)
    target_net = DQN(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)

    steps_done = 0

    print("AI Agent Active. Focus the MapleStory window NOW!")
    time.sleep(3)

    try:
        for episode in range(NUM_EPISODES):
            state = env.reset()
            total_reward = 0
            
            # Max steps per episode to prevent getting stuck
            for t in range(MAX_STEPS): 
                
                # 1. Visualization (The Panel)
                env.render()

                # 2. Select Action (Epsilon-Greedy)
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                np.exp(-1. * steps_done / EPS_DECAY)
                steps_done += 1
                
                if random.random() > eps_threshold:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        action = policy_net(state_t).max(1)[1].item()
                else:
                    action = random.randrange(n_actions)

                # 3. Step Environment (Includes delay inside step for action duration)
                next_state, reward, done, info = env.step(action)
                total_reward += reward

                # 4. Detailed Logging (As requested)
                # Format: Time | Char Pos | Monsters Count | Action Taken | Reward
                player_str = f"({info['player_pos'][0]}, {info['player_pos'][1]})" if info['player_pos'] else "Not Found"
                climb_str = "[C]" if info['is_climbing'] else ""
                hp_str = "LOW HP!" if info['low_hp'] else "HP OK"
                
                print(f"Time: {time.strftime('%H:%M:%S')} | "
                      f"Char: {player_str} {climb_str} | "
                      f"Monsters: {len(info['monsters'])} | "
                      f"Action: {info['action_name']} | "
                      f"Reward: {reward:.2f} | "
                      f"Status: {hp_str}")

                # 5. Store & Optimize
                memory.push(state, action, reward, next_state, done)
                state = next_state

                if len(memory) > BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                    
                    batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
                    batch_action = torch.LongTensor(batch_action).unsqueeze(1).to(device)
                    batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                    batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
                    batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

                    curr_q = policy_net(batch_state).gather(1, batch_action)
                    next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                    expected_q = batch_reward + (GAMMA * next_q * (1 - batch_done))

                    loss = nn.MSELoss()(curr_q, expected_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 6. Requested Loop Delay
                time.sleep(LOOP_DELAY)

                if done:
                    break
            
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            print(f"--- Episode {episode+1} Complete. Total Reward: {total_reward:.2f} ---")
            
            if episode % 50 == 0:
                torch.save(policy_net.state_dict(), f"maple_dqn_latest.pth")

    except KeyboardInterrupt:
        print("Stopping Training...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    train()