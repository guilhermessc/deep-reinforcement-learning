import numpy as np
from mtools import *

EPSILON = 0.000001
MAX_POLICY=True

class RandomAgent:
	'''
	This agent acts randomly
	'''
	def __init__(self, nA, space_size):

		self.nA = nA
		self.space_size = space_size

	def act(self, state):
		return np.random.choice(self.nA)

	def learn(self, previous_state, new_state, action, reward, done):
		pass



class DeepQAgent:
	'''
	This agent acts based on a Q-function computed by a Neural Network
	'''
	# TODO: Space pre-processing based on the limits of the observed space
	def __init__(self, nA, space_size, alpha=0.25, gamma=0.9, eps=1, eps_decay=0.9999, eps_min=0.05, memory_size=1000, batch_size=64, train_time=4, exploration_boost=0.001, reset_eps_every=100000):

		self.nA = nA
		self.space_size = space_size
		self.alpha = alpha
		self.gamma = gamma
		self.eps = max(eps, 1)                               # limit eps to >=1
		self.eps_decay = max(min(eps_decay, 1), 0)           # limit eps_decay to [0, 1]
		self.memory = ReplayBuffer(memory_size, batch_size)
		self.eps_min = eps_min
		self.counter = 0
		self.train_time = train_time
		self.Q = QNN(input_size=space_size, output_size=nA)
		self.exploration_boost = exploration_boost
		self.reset_eps_every = reset_eps_every

	def act(self, state):
		# update counter
		self.counter += 1
		if self.counter % self.reset_eps_every == 0:
			self.reset_eps(eps=0.5)

		# epsilon-greedy policy update
		self.eps*=self.eps_decay
		self.eps = max(self.eps, self.eps_min)

		# Exploration
		if np.random.uniform() < self.eps:
			return np.random.choice(self.nA)

		# Exploitation
		qs = self.Q.get(state) # get Q-values from the Q-function
		probas = self.policy(qs)
		return np.random.choice(self.nA, p=probas)


	# TODO: Try different policy distributions
	def policy(self, qs):

		policy = np.zeros(self.nA)
		policy[np.argmax(qs)] = 1
		return policy


	def learn(self, previous_state, new_state, action, reward, done):
		# Boost exploration
		if reward == 0:
			reward -= self.exploration_boost

		# Put experience into memory
		self.memory.add(previous_state, action, reward, new_state, done)

		# periodic train
		if self.counter % self.train_time == 0:
			sample = self.memory.sample()

			losses = []
			states_to_update = []
			for s in sample:
				state, action, reward, next_state, done = tuple(s)
				states_to_update.append(state)

				# Compute TD-error
				qsa = self.Q.get(state)[action]
				qsa_next = max(self.Q.get(next_state))
				# Do not compute reward specifically for a final state since
                                #    it could be interpreted as any other state
				if done and False:
					td_error = reward - qsa
				else:
					td_error = reward + self.gamma*qsa_next - qsa

				# Compute loss
				loss = np.zeros(self.nA)
				loss[action] = td_error
				losses.append(loss)

			# Update QNetwork
			self.Q.update(states_to_update, losses)

	def reset_eps(self, eps=1):
		self.eps = max(eps, 1)


class DQNAgent:
	'''
	This agent acts based on a Q-function computed by a Neural Network
	'''
	# TODO: Space pre-processing based on the limits of the observed space
	def __init__(self, nA, space_size, alpha=1, gamma=0.95, eps=1, eps_decay=0.995, eps_min=0.01, memory_size=2000, batch_size=128, train_time=4, exploration_boost=0.0, reset_eps_every=100000, hidden_layers=[6, 64, 64], activation=['relu', 'relu', 'relu', 'linear'], lr=0.001, tau=0.05):

		self.nA = nA
		self.space_size = space_size
		self.alpha = alpha
		self.gamma = gamma
		self.eps = max(eps, 1)                               # limit eps to >=1
		self.eps_decay = max(min(eps_decay, 1), 0)           # limit eps_decay to [0, 1]
		self.memory = ReplayBuffer(memory_size, batch_size)
		self.eps_min = eps_min
		self.counter = 0
		self.train_time = train_time
		self.Q = DQN(input_size=space_size, output_size=nA, hidden_layers=hidden_layers, activation=activation, lr=lr, tau=tau)
		self.exploration_boost = exploration_boost
		self.reset_eps_every = reset_eps_every

	def act(self, state):
		# update counter
		self.counter += 1
		if self.counter % self.reset_eps_every == 0:
			self.reset_eps()

		# epsilon-greedy policy update
		self.eps*=self.eps_decay
		self.eps = max(self.eps, self.eps_min)

		# Exploration
		if np.random.uniform() < self.eps:
			return np.random.choice(self.nA)

		# Exploitation
		# Act based on Q_minus
		qs = self.Q.get(state)[0] # get Q-values from the Q-function
		probas = self.policy(qs)
		try:
			return np.random.choice(self.nA, p=probas)
		except Exception as err:
			print(err)
			return np.random.choice(self.nA) # return a random action

	def policy(self, qs, get_max=MAX_POLICY):

		policy = np.zeros(self.nA)

		if get_max:
			policy[np.argmax(qs)]=1
			return policy

		shift = 0
		for i, qsa in enumerate(qs):
			# if positivelly weighted use the weight
			if qsa > 0:
				policy[i] = qsa
			# if weight is negative distribute it among the others
			else:
				shift += -qsa/(self.nA-1)
				policy[i] = qsa/(self.nA-1)

		# Shift up all negative values
		policy += shift

		# Normalize
		policy /= sum(policy)
		policy = np.around(policy, decimals=3)
		# sum* == 1
		policy /= sum(policy)

		return policy


	def td_error(self, state, action, reward, next_state, done):
		# Compute TD-error
		qsa = self.Q.get(state)[0][action]
		# Compute the best action by Q and evaluate by the fixed target Q'
		qsa_next = self.Q.get(next_state)[1][np.argmax(self.Q.get(next_state)[0])]
		if done and False: # Deactivate done because final state is not an objective
			td_error = reward - qsa
		else:
			td_error = reward + self.gamma*qsa_next - qsa

		return td_error


	def learn(self, previous_state, new_state, action, reward, done):
		# Boost exploration
		if reward == 0:
			reward -= self.exploration_boost

		# Put experience into memory
		td_error = self.td_error(previous_state, action, reward, new_state, done)
		self.memory.add(previous_state, action, reward, new_state, done, EPSILON + np.abs(td_error))
		ret = td_error

		# periodic train
		if self.counter % self.train_time == 0:
			sample = self.memory.sample()

			losses = []
			states_to_update = []
			for s in sample:
				state, action, reward, next_state, done = tuple(s)
				states_to_update.append(state)

				# Compute TD-error
				td_error = self.td_error(state, action, reward, next_state, done)

				# Compute loss
				loss = np.zeros(self.nA)
				loss[action] = td_error
				losses.append(loss)

			# Update QNetwork
			self.Q.update(states_to_update, losses)

		return ret

	def reset_eps(self, eps=1):
		self.eps = min(eps, 1)
