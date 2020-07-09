## Learning Algorithm
The learning algorithm here used is Deep Q Network with the help of "Experience Replay" and "Fixed Q Targets". The Q net is a simple 3-layers fully connected network.

Hyperparameters are:
- learning_rate: 0.0005
- gamma: 0.99
- tau: 0.001
- Epsilon: 1.0
- Epsilon_Decay: 0.99
- Epsilon_Min: 0.01
- Target network updates after 4 times updates of local network.

## Plot of Rewards
<img src="https://github.com/YueYao-bot/Udacity-Navigation-Project/blob/master/Scores_over_Episodes.png"/>

Problem solved at 361 Episodes.

## Ideas for Future Work
- Use some other complex networks like CNN instead of fully connected network
- Instead use the states from unity enviroment, we can try to train the agent directly based on the video / images.