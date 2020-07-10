# Project3: Tennis


## Project Details
<img src="https://github.com/YueYao-bot/Udacity-Collaboration-and-Competition/blob/master/test_result.gif"/>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

Tested with python 3.6.0 with
- pytorch 1.4.0
- unityagents 0.4.0

## Getting Started
The required environment *Tennis.app* is uploaded in this project. You can simply start the project without any additional work.



## Instructions
*Tennis.ipynb* is the entry of the project and has two modes. One for training and one for testing. Be sure that you have MADDPG networks "checkpoint.pth". Otherwise you need train the MADDPG-Agent firstly. 

You also use *Tennis.py* if you do not want to use jupyter notebook. It has same code as in *Tennis.ipynb*.

*checkpoint_actor_player1.pth* and *checkpoint_critic_player1.pth* save the weights of a trained model for player1 and player2 also has its own weights saved. you can load it and start the test mode!

There is a second version of weights, which can achieve avg score of 1.0 during training.