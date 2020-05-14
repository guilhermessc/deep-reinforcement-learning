BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 500        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 1e-8     # L2 weight decay

ACTOR_FC1_UNITS = 200
ACTOR_FC2_UNITS = 150
CRITIC_FC1_UNITS = 400
CRITIC_FC2_UNITS = 300