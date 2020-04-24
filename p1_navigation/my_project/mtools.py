import numpy as np

# TODO: Update requirements for deploy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from unityagents import UnityEnvironment

import matplotlib.pyplot as plt

import pickle

from copy import deepcopy



class ReplayBuffer:
	'''Fixed-size buffer to store experience tuples.'''

	def __init__(self, buffer_size, batch_size):
		self.memory = []
		self.buffer_size = buffer_size
		self.batch_size = batch_size


	def add(self, state, action, reward, next_state, done, error=1):
		'''Add a new experience to memory.'''

		# Append the experience
		experience = (state, action, reward, next_state, done, error)
		self.memory.append(experience)

		# Trim the memory with respect to the size
		if len(self.memory) > self.buffer_size:
			self.memory = self.memory[-self.buffer_size:]


	def sample(self, k=None):
		'''Randomly sample a batch of experiences from memory.'''
		if k is None:
			k=self.batch_size

		# check for empty buffer
		mlen = len(self.memory)
		if mlen == 0:
			return []

		# randomly sample the memory
		probas = np.array([float(experience[5]) for experience in self.memory])
		probas /= sum(probas)
		sample = np.random.choice(mlen, k, p=probas)
		# sample the experiences without the error weight
		samples = [self.memory[s][:-1] for s in sample]

		return samples


class QNN:

	def __init__(self, input_size, output_size, hidden_layers=[64, 64], activation=['relu', 'relu', 'linear'], lr=0.0005):

		# Create the neural network model
		model = Sequential()
		layers = np.concatenate([hidden_layers, [output_size]])

		# Create input layer
		model.add(Dense(hidden_layers[0], input_dim=input_size, activation=activation[0]))

		# Stack the layers after the first
		for layer, activ in zip(layers[1:], activation[1:]):
			model.add(Dropout(0.1)) # Add dropout in between layers
			model.add(Dense(layer, activation=activ))

		adam = Adam(lr=lr)
		model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy'])

		self.model = model

	def get(self, x):
		''' Return the Q-Value for each action from a given state x '''

		# Transform x to Keras input format
		if len(np.shape(x)) == 1:
			x = [x]
		x = np.array(x)

		y = self.model.predict(x)

		if len(y) == 1:
			return y[0] # return one item for a simple prediction
		return y # return a batch for batch prediction

	def update(self, x, loss):

		# Compute desired Y-update value
		y_pred = self.get(x)
		y_real = y_pred + np.array(loss)

		# Transform x to Keras input format
		if len(np.shape(x)) == 1:
			x = [x]
		x = np.array(x)

		self.model.train_on_batch(x, y_real)


class DQN:

	def __init__(self, input_size, output_size, hidden_layers=[64, 64], activation=['relu', 'relu', 'linear'], lr=0.0005, tau=0.25):

		# Create the neural network model
		model = Sequential()
		layers = np.concatenate([hidden_layers, [output_size]])

		# Create input layer
		model.add(Dense(hidden_layers[0], input_dim=input_size, activation=activation[0]))

		# Stack the layers after the first
		for layer, activ in zip(layers[1:], activation[1:]):
			model.add(Dense(layer, activation=activ))

		adam = Adam(lr=lr)
		model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy'])

		self.model = model
		self.model_minus = deepcopy(model)
		self.tau = tau

	def get(self, x):
		''' Return the Q-Value for each action from a given state x '''

		# Transform x to Keras input format
		if len(np.shape(x)) == 1:
			x = [x]
		x = np.array(x)

		y = self.model.predict(x)
		y_minus = self.model_minus.predict(x)

		if len(y) == 1:
			return (y[0], y_minus[0]) # return one item for a simple prediction
		return (y, y_minus) # return a batch for batch prediction

	def update(self, x, loss):

		# Compute desired Y-update value
		y_pred = self.get(x)[0]

		# "Real" value is based on the Q-fixed target so it doesn't change itself at every update
		# y_real is Q(S', argmax_a(Q(S',a,w)), w'): the decision taken by the main model evaluated by the old one
		y_real = y_pred + np.array(loss)

		# Transform x to Keras input format
		if len(np.shape(x)) == 1:
			x = [x]
		x = np.array(x)

		self.model.train_on_batch(x, y_real)
		self.interpolate_models()

	def interpolate_models(self):
		''' Interpolates the weights in both models for a soft update '''
		# Recover each model weights
		weight = np.array(self.model.get_weights())
		weight_minus = np.array(self.model_minus.get_weights())

		# Compute the interpolated weights by a rate of self.tau
		interpolated_weight = (1-self.tau)*weight_minus + self.tau*weight

		# Update model_minus
		self.model_minus.set_weights(interpolated_weight)


def create_env(verbose=0):
	env = UnityEnvironment(file_name="./Banana.x86_64")

	# get the default brain
	brain_name = env.brain_names[0]
	brain = env.brains[brain_name]

	# number of actions
	action_size = brain.vector_action_space_size

	# examine the state space
	env_info = env.reset(train_mode=True)[brain_name]
	state = env_info.vector_observations[0]
	state_size = len(state)

	if verbose > 0:
		print('Number of actions:', action_size)
		print('States look like:', state)
		print('States have length:', state_size)

	return (env, brain_name, action_size, state_size)



def simulate(env, brain_name, agent, n_episodes=10000, verbose=0, train_mode=True):

	scores = []
	for episode in range(n_episodes):

		env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
		state = env_info.vector_observations[0]            # get the current state
		done = False
		score = 0                                          # initialize the score
		while not done:
			
			action = agent.act(state)					   # select an action
			
			env_info = env.step(action)[brain_name]        # send the action to the environment
			next_state = env_info.vector_observations[0]   # get the next state
			reward = env_info.rewards[0]                   # get the reward
			done = env_info.local_done[0]                  # see if episode has finished
			
			agent.learn(state, next_state, action, reward, done) # train the agent (if trainable)

			score += reward                                # update the score
			state = next_state                             # roll over the state to next time step

		if verbose > 0:
			print("{}: \tScore: {}".format(episode, score))

		scores.append(score)

	return scores


def plot_score(score, filename=None, title='Agent Score'):

	# Compute the rolling average
	rolling_avg = np.zeros(len(score))
	for i in range(99, len(score)):
		rolling_avg[i] = np.mean(score[i-99:i+1])

	# Create a 16x9 figure
	fig, ax = plt.subplots(figsize=(16,9))

	# Plot both curves over the same axis
	ax.plot(range(len(score)), score, label='Score', color='blue', alpha=0.75)
	ax.plot(range(len(rolling_avg)), rolling_avg, label='Rolling average', color='red')


	ax.set_title(title)
	ax.legend()


	if filename is not None:
		fig.savefig(filename)
	else:
		fig.show()

	# return the maximum avg score over 100 consecutives episodes
	return max(rolling_avg)


def save_agent(agent, filename='agent.bin'):

	filehandler = open(filename, 'wb')
	pickle.dump(agent, filehandler)


def load_agent(filename='agent.bin'):

	filehandler = open(filename, 'rb')
	agent = pickle.load(filehandler)
	return agent
