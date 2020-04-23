import numpy as np

# TODO: Update requirements for deploy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from unityagents import UnityEnvironment



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
