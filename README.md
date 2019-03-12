
[image1]: ./images/trained_agents.gif "trained agents" 

# Continuous Control

Project 2 in Udacity deep reinforcement learning nanodegree.

In this project we controlled a double-jointed arm to maintain its position at the moving target location.

![alt text][image1]

# Overview

In [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of an agent is to maintain its position at the target location for as many time steps as possible.  

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

In this project we solved a multi-agent version of Reacher environment. This version has 20 identical agents and arms, each with its own copy of the environment. 

To solve the environment, an algorithm must get an average score of +30  over 100 consecutive episodes, and over all agents.  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an average score for each episode (where the average is over all 20 agents).  
- The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


[Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment is provided by 
[Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). 

# Setup 

1. Follow steps 1, 3 and 4 in the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required for this project.

2. Clone this repository.

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)



4. Place the file in the root folder of this repository, and unzip (or decompress) the file. 

# Usage   
Follow the instructions in `Continuous-Integration.ipynb` to train the agent.

# Files

Continuous-Integration.ipynb - notebook that trains an agent and runs it through the environment  
project_report.md - technical details of training environment and process  
checkpoint.pth - weights of the actor neural network of DDPG algorithm  
code - folder with Python files that implement DDPG algorithm  
images - folder with auxiliary images

