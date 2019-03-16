[image1]: ./images/graph.jpg "convergence graph" 
[image2]: ./images/trained_agents.gif "trained agent" 

# Project Report

In this project we controlled 20 double-jointed arms to maintain their positions at the moving target locations.

The 20 agents were trained using DDPG algorithm. In the next two sections we describe this environment as well as the training process. 

## Reacher Environment

In [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of an agent is to maintain its position at the target location for as many time steps as possible.  

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

In this project we solved a multi-agent version of Reacher environment. This version has 20 identical agents and arms, each with its own copy of the environment. 

To solve the environment, an algorithm must get an average score of +30  over 100 consecutive episodes, and over all agents.  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an average score for each episode (where the average is over all 20 agents).  
- The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


[Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment is provided by 
[Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). 

## Training 20 Agents using multi-agent DDPG Algorithm

We trained 20 agents using a multi-agent version of [Deep Deterministic Policy Gradient (DDPG) algorithm](https://arxiv.org/abs/1509.02971). DDPG algorithm is an extension of [DQN algorithm](https://deepmind.com/research/dqn/) to continuous actions. In the next 4 paragraphs we describe various components of DDPG and then we will present its complete pseudocode.

DDPG algorithm uses local and target actor functions a and a'. These functions are implemented as feedforward neural networks and have identical architecture. The networks take as input a state vector S and output a vector of continuous actions A. The dimensionality of A is the number of continuous actions performed by a single agent at every time point. We denote by A=a(S,w<sub>a</sub>) the vector of continuous actions at state S, computed by local actor neural network with weights w<sub>a</sub>. Similarly, A'=a'(S,w'<sub>a</sub>) is the vector of continuous actions at state S, computed by target actor neural network with weights w'<sub>a</sub>. 

In addition to actor functions, DDPG uses local and target critic functions q and q'. Similarly to actor functions, critic functions are implemented as feedforward neural networks. Local and target critic networks have the same architecture, which is different from the architecture of actor networks. The critic networks take as input a state vector S and a vector of continuous actions A. The output of critic network is the value of an action A taken at state S. We denote by q(S,A,w<sub>c</sub>) the value of the vector of continuous actions A at state S, computed by local critic neural network with weights w<sub>c</sub>. Similarly, q'(S,A,w'<sub>c</sub>) is the value of continuous actions A at state S, computed by target critic neural network with weights w'<sub>c</sub>.

We denote by S<sup>i</sup> the state of the i-th agent, by A<sup>i</sup> the action vector of the i-th agent and by R<sup>i</sup> the reward received by the i-th agent. Finally S'<sup>i</sup> is the next state of the i-th agent and done<sup>i</sup> is an indicator if S'<sup>i</sup> is a final state.  

DDPG uses a replay buffer that has a finite capacity and works in FIFO way. The replay buffer stores recent experiences of all agents and is a source of training data for actor and critic networks.

An policy p(a(S,w<sub>a</sub>),&epsilon;,N) chooses an action a(S,w<sub>a</sub>) + &epsilon; &middot; N, where N is [Ornstein-Uhlenbeck noise process](https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process) and &epsilon;>0 is a scaling factor. This scaling factor controls the amount of exploration done by the policy. We decrease the scaling factor after each episode, thus allowing more exploration in the initial episodes and less exploration in the final ones.

Out implementation of multi-agent DDPG algorithm is summarized below:

Initialize replay buffer with capacity BUFFER_SIZE  
Initialize local actor function a with weights w<sub>a</sub>  
Initialize target actor function a' with weights w'<sub>a</sub> = w<sub>a</sub>  
Initialize local critic function q with weights w<sub>c</sub>  
Initialize target critic function q' with weights w'<sub>c</sub> = w<sub>c</sub>   
Initialize number of steps by 0  
Set &epsilon;=NOISE_START   

While environment is not solved:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Get from the environment initial state S<sup>i</sup>, i=1,...,20 of every agent      
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; While all of done<sup>i</sup>, i=1,...,20, are false:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For agent i=1,...,20:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Choose action A<sup>i</sup> from state S<sup>i</sup> using policy p(a(S<sup>i</sup>,w<sub>a</sub>),&epsilon;,N)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Take action A<sup>i</sup>, observe reward R<sup>i</sup>, next state S'<sup>i</sup> and the indicator done<sup>i</sup>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Store experience tuple (S<sup>i</sup>,A<sup>i</sup>,R<sup>i</sup>,S'<sup>i</sup>,done<sup>i</sup>) in the replay buffer    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;S<sup>i</sup> = S'<sup>i</sup>    

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Increase the number of steps by 1   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If 
the number of observed experiences is a multiple of UPDATE_EVERY:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For i = 1,...,N_UPDATES:   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample a batch of K experiences (S<sub>i</sub>,A<sub>i</sub>,R<sub>i</sub>,S'<sub>i</sub>,done<sub>i</sub>), i=1 ... K, from the replay buffer    

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Get next-state actions from actor target network   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A'<sub>i</sub>=a'(S'<sub>i</sub>,w'<sub>a</sub>), i=1,...,K     

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Get next-state action values from critic target network  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q'<sub>i</sub>=q'(S'<sub>i</sub>,A'<sub>i</sub>,w'<sub>c</sub>), i=1,...,K   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
// Compute state-action value targets for current states (S_i)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q<sub>i</sub>=r<sub>i</sub>+&gamma;&middot;Q'<sub>i</sub>&middot;(1-done<sub>i</sub>), i=1,...,K   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
// Update local critic network   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &Delta;w<sub>c</sub>=&sum;<sub>i=1 ... K</sub> (Q<sub>i</sub>-q(S<sub>i</sub>,w,A<sub>i</sub>))&middot;&nabla;q(S<sub>i</sub>,A<sub>i</sub>,w<sub>c</sub>)    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If   ||&Delta;w<sub>c</sub>||<sub>2</sub>>1 then rescale &Delta;w<sub>c</sub> to have ||&Delta;w<sub>c</sub>||<sub>2</sub>=1   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; w<sub>c</sub> = w<sub>c</sub> + &alpha;<sub>c</sub>&middot;&Delta;w<sub>c</sub>   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // Update local actor network   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Let q'(S<sub>i</sub>,a(S<sub>i</sub>,w<sub>a</sub>),w<sub>c</sub>) be a derivative of q(S<sub>i</sub>,a(S<sub>i</sub>,w<sub>a</sub>),w<sub>c</sub>) with respect to a(S<sub>i</sub>,w<sub>a</sub>)    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &Delta;w<sub>a</sub>=&sum;<sub>i=1 ... K</sub> -q(S<sub>i</sub>,a(S<sub>i</sub>,w<sub>a</sub>),w<sub>c</sub>)&middot;&nabla;a(S<sub>i</sub>,w<sub>a</sub>)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; w<sub>a</sub> = w<sub>a</sub> + &alpha;<sub>a</sub>&middot;&Delta;w<sub>a</sub>   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // Update target networks  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;w'<sub>a</sub>=&tau;&middot;w<sub>a</sub>+(1-&tau;)&middot;w'<sub>a</sub>   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;w'<sub>c</sub>=&tau;&middot;w<sub>c</sub>+(1-&tau;)&middot;w'<sub>c</sub>  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &epsilon; = max(NOISE_END,
NOISE_DECAY&middot;&epsilon;)  



The following table lists the architecture of local and target actor networks.

| Name | Type | Input | Output | Activation function | 
|:-:|:-----:|:-----:|:------:|:-------------------:|  
|Input Layer| Fully Connected  | 33    | 400    | Batch Normalization + ReLU                |
|Hidden Layer| Fully Connected   | 400    | 300     | Batch Normalization + ReLU                |
|Output Layer| Fully Connected   | 300    | 4      | Batch Normalization + tanh             |

The following table lists the architecture of local and target critic networks.

| Name | Type | Input | Output | Activation function | 
|:-:|:-----:|:-----:|:------:|:-------------------:|  
|Input Layer| Fully Connected  | 33    | 400    | Batch Normalization + ReLU                |
|Hidden Layer| Fully Connected   | 404    | 300     | Batch Normalization + ReLU                |
|Output Layer| Fully Connected   | 300    | 1      | Linear        |

We tuned the values of hyperparameters to solve the environment. The next table summarizes the our final values of hyperparameters:

| Hyperparameter | Description | Value |
|:--------------:|:-----------:|:-----:|
| BUFFER_SIZE | Size of replay buffer | 1000000 |
| UPDATE_EVERY | Number of steps between every round of updates | 20 |
| N_UPDATES | Number of batches in a single round of updates | 10 |
| K | Batch size | 256 |
| &gamma; | Discount factor | 0.99 |
| &alpha;<sub>a</sub> | Learning rate of local actor network | 0.001 |
| &alpha;<sub>c</sub> | Learning rate of local critic network | 0.001 | 
| WEIGHT_DECAY | L2 regularization coeffcient  | 0 | 
| &tau; | Weight of local network when updating target network| 0.001 |
| NOISE_START | Initial value of noise scaling factor | 1 |
| NOISE_END | Minimal value of noise scaling factor | 0.01 |
| NOISE_DECAY | Decay rate of noise scaling factor | 0.995 |

## Software packages

We used PyTorch to train neural network and  the API of Unity Machine Learning Agents Toolkit to interact with Reacher environment. 

## Results

DDPG ealgorithm solves environment within 47 episodes. The following graph shows the total reward as a function of the number of episode.  

![alt text][image1]

And here is the video of the trained agents following the target locations:

![alt text][image2]

## Ideas for Future Work

While we were able to solve the environment and reach an average reward of 30, there are some agents that still have mediocre performance. For example the central agent in the first row. It would be interesting to try to reach a worst-case reward of 30 by tuning hyperparameters. 

Another direction for future work is to try other algorithms that generate continuous actions (e.g. [TRPO](https://arxiv.org/pdf/1502.05477.pdf), [PPO](https://arxiv.org/pdf/1707.06347.pdf) and [D4PG](https://openreview.net/forum?id=SyZipzbCb)) and see if they can solve Reacher environment substantially faster than DDPG.

