import numpy as np

# TODO: Update requirements for deploy
from keras.models import Sequential
from keras.layers import Dense, Dropout


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
		probas = [experience[5] for experience in self.memory]
		probas/=sum(probas)
		sample = np.random.choice(mlen, k, p=probas)
		samples = self.memory[sample]

		states      = [s[0] for s in samples]
		actions     = [s[1] for s in samples]
		rewards     = [s[2] for s in samples]
		next_states = [s[3] for s in samples]
		dones       = [s[4] for s in samples]

		return (states, actions, rewards, next_states, dones)


class QNN:

	def __init__(self, input_size, output_size, hidden_layers=[15, 8], activation=['relu', 'relu', 'linear']):

		# Create the neural network model
		model = Sequential()
		layers = np.concatenate([hidden_layers, [output_size]])

		# Create input layer
		model.add(Dense(hidden_layers[0], input_dim=input_size, activation=activation[0]))

		# Stack the layers after the first
		for layer, activ in zip(layers[1:], activation[1:]):
			model.add(Dropout(0.1)) # Add dropout in between layers
			model.add(Dense(layer, activation=activ))

		model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['accuracy'])

		self.model = model

	def get(self, x):
		''' Return the Q-Value for each action from a given state x '''
		
		# Transform x to Keras input format
		if len(np.shape(x)) == 1:
			x = np.array([x])
	
		y = self.model.predict(x)

		if len(y) == 1:
			return y[0] # return one item for a simple prediction 
		return y # return a batch for batch prediction

	def update(self, x, loss):
		
		# Compute desired Y-update value
		y_pred = self.get(x)
		y_real += np.array(loss)

		# Train on the batch many times to have a significant impact on the gradient
		# TODO: refactor this
		for i in range(100):
			self.model.train_on_batch(x, y_real)

