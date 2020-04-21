# Project Details

This project implements an agent based on a **`Deep Q-Learning algorithm`**. The agent's goal is to navigate in a box and collect yellow bananas while avoiding blue bananas.

The agent is rewarded with `+1` for each `yellow banana` collected and `-1` for each `blue banana`. 

The **observation space** has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. The **action space** has 4 discrete options:

- 0 : Move forward.
- 1 : Move backward.
- 2 : Turn left.
- 3 : Turn right.

The task is episodic and it is considered solved once the agent reaches an average score of `13+` over 100 consecutive episodes.


# Getting Started

1. To use the environment follow the instructions on the [Getting Started](https://github.com/guilhermessc/deep-reinforcement-learning/blob/master/p1_navigation/README.md "Parent directory README") section of the parent directory and copy the files here.

[//]: # (TODO: Improve this set up)
2. Rename the executable environment file as _Banana.x86_64_.

3. Install the dependencies on `deep-reinforcement-learning/python/` using the following command:

``bash
$ pip3 install .
``

