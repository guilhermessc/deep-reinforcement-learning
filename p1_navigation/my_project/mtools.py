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


