# Project 2: Continuous Control

## Project details

In this project, the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment will be used.

This environment contains, an agent, a double-jointed arm that can move freely in a 3 dimensional space and must be trained to reach target locations. The agent receives a reward of +0.1 for each step it has its hands in the moving target.

The agent observes 33 continuous variables (position, rotation, velocity, and angular velocities of the arm) and performs an action. An action is the torque applicable to each joints and it is represented by a vector with four numbers. These numbers must belong to the interval between -1 and 1.

### Distributed Training

In order to decrease the training time, the environment contains 20 identical agents, each with its own copy of the environment. This allows the use of more complex algorithms, where the agent trains from several different perspectives at a time thus better exploring the environment and often performing better.

### Solving the Environment

To solve the environment, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, the rewards that each agent received (without discounting), are added up to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in this directory. Then, unzip (or decompress) the file. 

## Instructions

After downloading the environment, follow the instructions in `Continuous_Control.ipynb` to get started. 
 
