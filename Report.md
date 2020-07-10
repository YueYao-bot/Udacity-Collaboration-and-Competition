## Learning Algorithm
The learning algorithm here used is MADDPG with 2 DDPG agents for each player with the help of "Experience Replay" and "Fixed Targets". The DDPG for two players has same structure. Each DDPG agent has a local actor and a target actor, both are 3-layers fully connected networks. And it also has a local critic and a target critic, which are 3-layers fully connected networks.

Both target actor and target critic will be updated in *n* timesteps, where *n* is also a hyperparameter.

Hyperparameters are:
- learning_rate: 0.0005
- gamma: 0.99
- tau: 0.0005
- Epsilon: 1.0
- Target network will be updated in every 2 timesteps

## Plot of Rewards
<img src="https://github.com/YueYao-bot/Udacity-Collaboration-and-Competition/blob/master/Scores_over_Episodes.png"/>

Problem solved at 926 Episodes.

## Test Result
<img src="https://github.com/YueYao-bot/Udacity-Collaboration-and-Competition/blob/master/test_result.gif"/>

## Ideas for Future Work
- Use some other complex networks instead of fully connected network with few layers.
- Try to use shared weights for the agents in MADDPG.