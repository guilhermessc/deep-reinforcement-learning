#! /bin/python3

import numpy as np
from agent import *
from mtools import *


# Create the environment
env, brain_name, action_size, state_size = create_env()

# Create the agent
agent = DeepQAgent(action_size, state_size)

# Simulate the agent
scores = simulate(env=env, brain_name=brain_name, agent=agent, n_episodes=10000, verbose=1)

# Plot the results
plot_score(scores, filename='agent_score.png', title='Agent Score')

# Save the agent
save_agent(agent, filename='agent.bin')

# Close the environment
env.close()