from DDPG_Agent import DDPG_Agent
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, state_size, action_size, seed):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPG_Agent(state_size, action_size, seed),
                             DDPG_Agent(state_size, action_size, seed)]

        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor_net_local for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.actor_net_target for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs) for ddpg_agent, obs in
                          zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def step(self, states, actions, rewards, next_states, dones):
        for i in range(2):
            self.maddpg_agent[i].step(states[i,:], actions[i], rewards[i], next_states[i,:], dones[i])

    def save(self,version = "v1"):
        for i in range(len(self.maddpg_agent)):
            torch.save(self.maddpg_agent[i].actor_net_local.state_dict(), './weights/checkpoint_actor_player'+ str(i+1)+'_' + version + '.pth')
            torch.save(self.maddpg_agent[i].critic_net_local.state_dict(), './weights/checkpoint_critic_player' + str(i+1) + '_'+version+ '.pth')

    def load(self, version = "v2"):
        self.maddpg_agent[0].actor_net_local.load_state_dict(torch.load('./weights/checkpoint_actor_player1_'+version+'.pth'))
        self.maddpg_agent[0].critic_net_local.load_state_dict(torch.load('./weights/checkpoint_critic_player1_'+version+'.pth'))
        self.maddpg_agent[1].actor_net_local.load_state_dict(torch.load('./weights/checkpoint_actor_player2_'+version+'.pth'))
        self.maddpg_agent[1].critic_net_local.load_state_dict(torch.load('./weights/checkpoint_critic_player2_'+version+'.pth'))