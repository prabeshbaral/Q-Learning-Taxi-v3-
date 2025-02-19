import numpy as np
import gym
import random

env = gym.make('Taxi-v3')

alpha = 0.9  # learning rate
gamma = 0.95  # discount factor
epsilon = 1  # exploration rate
epsilon_decay = 0.9995
min_epsilon = 0.01

num_episodes = 1000  # number of episodes
max_steps = 100  # maximum steps per episode

# Get environment details
state_size = env.observation_space.n  # Number of states
action_size = env.action_space.n  # Number of actions

q_table = np.zeros((state_size, action_size))

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore action space
    else:
        return np.argmax(q_table[state, :])  # Exploit learned values

# Training loop
rewards = []
for episode in range(num_episodes):
    state, _ = env.reset()  # Reset the environment
    done = False
    total_reward = 0

    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)

        # Update Q-table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    epsilon = max(min_epsilon, epsilon_decay * epsilon)  # Decay epsilon
    rewards.append(total_reward)

# Testing
test_episodes = 5
env = gym.make('Taxi-v3', render_mode='human')

for episode in range(test_episodes):
    state, _ = env.reset()
    done = False
    print('Episode', episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)

        state = next_state

        if done or truncated:
            env.render()
            break

env.close()