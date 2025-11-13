import gymnasium as gym
import ale_py
import numpy as np
import random
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import signal
import matplotlib.pyplot as plt
import sys

TOTAL_TRAIN_STEPS = 2_000_000
MAX_STEPS_PER_EPISODE = 2000
LEARNING_RATE = 0.0001
GAMMA = 0.999 
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000  
TARGET_UPDATE_FREQ_STEPS = 10000 
LEARN_EVERY_STEPS = 4       
BUFFER_WARMUP_SIZE = 50000  
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 500_000 
REWARD_PENALTY_ON_DEATH = -1001.0

WEIGHTS_FILE = "network_double_dqn.pth"

gym.register_envs(ale_py)
env = gym.make('ALE/Asterix-v5', obs_type="ram", render_mode='rgb_array')

state_size = env.observation_space.shape[0] 
action_size = env.action_space.n

print(f"Розмір стану (State Size): {state_size} (128 байт)")
print(f"Кількість дій (Action Size): {action_size}")


rewards = []

def show_plot(sig=None, frame=None):
    if not rewards:
        print("Немає даних для побудови графіка.")
        return
    
    plt.figure(figsize=(12, 6))
    
    if len(rewards) > 100:
        window_size = 100
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        x_axis = np.arange(len(moving_avg))
        plt.plot(x_axis, moving_avg, label=f'Ковзне середнє ({window_size} епізодів)')
    else:
        plt.plot(rewards, label='Середня винагорода')

    plt.xlabel("Епізод (x100)")
    plt.ylabel("Середня винагорода")
    plt.title("Прогрес навчання DQN")
    plt.legend()
    plt.grid(True)
    print("Показ графіка... (Закрийте вікно, щоб продовжити)")
    plt.show(block=True) 

signal.signal(signal.SIGINT, lambda: (show_plot(), sys.exit(0)))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = x.to(self.layer1.weight.device) / 255.0
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Agent:
    def __init__(self, state_size, action_size, learning_rate, gamma, batch_size, buffer_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)
        self.loss_fn = nn.SmoothL1Loss()
        
    def choose_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return env.action_space.sample()

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
        
        state_batch = torch.tensor(np.array(batch_state), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            policy_next_q_values = self.policy_net(next_state_batch)
            best_next_actions = policy_next_q_values.argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, best_next_actions)
            
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        loss = self.loss_fn(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def get_epsilon(step):
    fraction = min(1.0, step / EPSILON_DECAY_STEPS)
    return EPSILON_START - fraction * (EPSILON_START - EPSILON_END)


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(state_size, action_size, LEARNING_RATE, GAMMA, BATCH_SIZE, BUFFER_SIZE, device)

    if Path(WEIGHTS_FILE).exists():
        agent.policy_net.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.policy_net.eval()
        agent.target_net.eval()
        print(f"Завантажено навчені ваги з {WEIGHTS_FILE}")
    else: 
        total_rewards = []
        episode_count = 0 
        total_steps_counter = 0

        print("Початок тренування (Double DQN на Asterix RAM)...")
        print(f"Навчання почнеться після {BUFFER_WARMUP_SIZE} кроків.")
        print(f"Тренування триватиме {TOTAL_TRAIN_STEPS} кроків.")

        while total_steps_counter < TOTAL_TRAIN_STEPS:
            state, info = env.reset()
            episode_reward = 0
            episode_count += 1
            
            for step in range(MAX_STEPS_PER_EPISODE):
                total_steps_counter += 1 

                epsilon = get_epsilon(total_steps_counter)
                action = agent.choose_action(state, epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                modified_reward = reward 
                
                if terminated:
                    modified_reward = REWARD_PENALTY_ON_DEATH

                agent.buffer.push(state, action, modified_reward, next_state, done)

                state = next_state
                episode_reward += reward 

                if (total_steps_counter > BUFFER_WARMUP_SIZE) and \
                   (total_steps_counter % LEARN_EVERY_STEPS == 0):
                    agent.learn()
                    
                if (total_steps_counter > BUFFER_WARMUP_SIZE) and \
                   (total_steps_counter % TARGET_UPDATE_FREQ_STEPS == 0):
                   agent.update_target_net()
                   print(f"Крок {total_steps_counter}: Target-мережа оновлена.")

                if done or (total_steps_counter >= TOTAL_TRAIN_STEPS):
                    break

            total_rewards.append(episode_reward)
            
            if (episode_count + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-100:]) if total_rewards else 0.0
                rewards.append(avg_reward)
                print(f"Епізод: {episode_count + 1}, Кроків: {total_steps_counter}/{TOTAL_TRAIN_STEPS}, Сер. винагорода (100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

        print("Тренування завершено!")
        show_plot() 
        print(f"Збереження навченої моделі у {WEIGHTS_FILE}...")
        torch.save(agent.policy_net.state_dict(), WEIGHTS_FILE)

    print("\n--- Тестування навченого агента ---")
    env_test = gym.make('ALE/Asterix-v5', obs_type="ram", render_mode='human')
    num_test_episodes = 5

    for episode in range(num_test_episodes):
        state, info = env_test.reset()
        terminated = False
        truncated = False
        total_test_reward = 0
        print(f"\n*** Тестовий епізод {episode + 1} ***")

        while not (terminated or truncated):
            action = agent.choose_action(state, 0.0)
            new_state, reward, terminated, truncated, info = env_test.step(action)
            state = new_state
            total_test_reward += reward
            env_test.render()
            time.sleep(0.02)

        print(f"Загальна винагорода: {total_test_reward}")

    env.close()
    env_test.close()

if __name__ == "__main__":
    main()