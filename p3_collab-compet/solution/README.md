# Project 3: Collaboration and Competition

## Project Details

For this project, I worked with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment, where two players (agents) control rackets to bounce a ball over a net in a 2 dimensional vertical space.

In this environment, if an agent hits the ball over the net, it is rewarded with +0.1.  If it lets a ball hit the ground or go out of bounds, the reward is -0.01.  Thus, the goal is to cooperate to keep the ball in play.

Each agent observes 8 variables corresponding to the position and velocity of the ball and racket from its own perspective.  Two continuous actions are available, corresponding to movement toward or away from the net, and jumping. 

This task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, the rewards that each agent received (without discounting) are added to get a score for each agent. This yields 2 scores.
- To yield a single **score** for episode, the maximum of the 2 scores is taken.

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) To train the agent on AWS (without [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.

2. Place the file in this directory and unzip (or decompress) the file.

## Instructions

Follow the instructions in `Tennis.ipynb`.

