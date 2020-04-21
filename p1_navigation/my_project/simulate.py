#! /bin/python3

from unityagents import UnityEnvironment
import numpy as np
from agent import *

env = UnityEnvironment(file_name="./Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
done = False
agent = RandomAgent(action_size, state_size)

while not done:
	
	action = agent.act(state)					   # select an action
	
	env_info = env.step(action)[brain_name]        # send the action to the environment
	next_state = env_info.vector_observations[0]   # get the next state
	reward = env_info.rewards[0]                   # get the reward
	done = env_info.local_done[0]                  # see if episode has finished
	
	agent.learn(state, next_state, action, reward, done) # train the agent (if trainable)

	score += reward                                # update the score
	state = next_state                             # roll over the state to next time step
    
print("Score: {}".format(score))

env.close()

