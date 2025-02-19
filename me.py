import numpy as np
import gym
import random

env=gym.make('Taxi-v3')

alpha=0.9 #learning rate # How much new info overrides old info

gamma=0.95#long term reward are important(discount factor) # Future reward discount

epsilon=1  #1 for random and 0 for all from the q-table

epsilon_decay=0.9995
min_epsilon=0.01

num_episodes=1000 #how many time the agent plays the game,epoches

max_steps=100

# Get environment details
state_size = env.observation_space.n  # Number of states here it is 5*5 grid and 4 places of hotels and 4 place of client so state size is 400
action_size = env.action_space.n  # Number of actions ie 4 left right up down

q_table = np.zeros((state_size,action_size))


def choose_action(state):

    if random.uniform(0,1)<epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])  #gives the index that have maximum qvalue ie return best action
    
#a trainin loop
rewards=[]
for episode in range(len(num_episodes)):
    state,_=env.reset() #state will be input for our first decision

    done=False
    total_reward=0
    for step in range(max_steps):
        action=choose_action(state)

        #env.step(action) to do or perform the action
        next_state,reward,done,truncated,info=env.step(action)

        #updating q-table

        old_value= q_table[state, action]
        next_max=np.max(q_table[next_state , :])   #give for the next_state, the max q-value in the table

        q_table[state,action]=(1-alpha)*old_value +alpha*(reward+ gamma*next_max)         #(1-alpha) is inertia

        state=next_state
        total_reward+=reward

        if done or truncated:
            break
    epsilon=max(min_epsilon,epsilon_decay*epsilon)
    rewards.append(total_reward)



#testing
test_episodes=5

env=gym.make('Taxi-v3',render_mode='human')

for episode in range(test_episodes):
    state,_= env.reset()

    done=False
    print('Episode',episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state ,:])
        next_state,reward,done,truncated,info=env.step(action)

        state=next_state

        if done or truncated:
            env.render()
            break
env.close()
