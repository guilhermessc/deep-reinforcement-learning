import numpy as np

class RandomAgent:
	'''
	This agent acts randomly
	'''
	def __init__(self, nA, space_size):

		self.nA = nA
		self.space_size = space_size

	def act(self, state):
		self.previous_state = state
		return np.random.choice(self.nA)

	def learn(self, previous_state, new_state, action, reward, done):
		pass
