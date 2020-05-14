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

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Unzip (or decompress) the file, rename the executable as _Banana.x86_64_ and move all content to this directory.

3. Install the dependencies on `./python/` typing on a terminal from this directory the following command:

``
$ pip3 install ./python/
``

# Instructions
Follow the instructions on the `Navigation.ipynb` to run the code.

