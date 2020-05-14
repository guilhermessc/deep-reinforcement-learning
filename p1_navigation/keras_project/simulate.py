#! /bin/python3

import numpy as np
from agent import *
from mtools import *


# Create the environment
env, brain_name, action_size, state_size = create_env()

# Create the agent
agent = DQNAgent(action_size, state_size,
	alpha=1,
	gamma=0.99,
	eps=1,
	eps_decay=0.99995,
	eps_min=0.05,
	memory_size=10000,
	batch_size=64,
	train_time=4,
	exploration_boost=0.00, # give exploration boost to penalize spinning around
	reset_eps_every=300*700, # 300 iterations per episode * n episodes
	# DQN parameters
	hidden_layers=[1024, 256, 128, 64],
	activation=['relu', 'relu', 'relu', 'relu', 'linear'],
	lr=0.0005,
	tau=0.001)

# Simulate the agent
scores = simulate(env=env, brain_name=brain_name, agent=agent, n_episodes=4000, verbose=3, filename='mbot.bin')

# Save the agent
save_agent(agent, filename='agent.bin')

# Plot the results
plot_score(scores, filename='agent_score.png', title='Agent Score')

# Close the environment
env.close()
