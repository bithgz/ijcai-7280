import numpy as np
from agent.agent import Agents
from RmCooCEnv import RmCooCEnv
from RmComCEnv import RmComCEnv

class RolloutWorker:

    def __init__(self, port, args, no_graphics=True, time_scale=100):
        self.args = args
        self.port = port
        self.env = RmComCEnv(port=self.port, no_graphics=no_graphics, time_scale = time_scale)
        self.env_test = RmCooCEnv(port=self.port + 1, no_graphics=no_graphics, time_scale = time_scale)
        

        self.is_cuda = False
        self.agents = Agents(self.args, self.is_cuda)

        self.n_actions = args.n_actions
        self.n_agents_per_party = args.n_agents_per_party
        self.obs_shape = args.obs_shape
        self.args = args

        print('Init RolloutWorker')





    def generate_episode(self, model):

        self.agents.policy.load_para(model)

        o,  u,  r, terminate = [], [], [], []
       
        d = False
        obs = self.env.reset(None)
        
        while not d:
            
            action = self.agents.get_action(np.array(obs))

            obs2, reward, d, _ = self.env.step(action)
          
           
            o.append(obs)
            u.append(np.reshape(action, [2 * self.args.n_agents_per_party, self.args.n_actions]))
            r.append([reward[0]])
            terminate.append([d])

            obs = obs2


        episode = None
       
        o.append(obs)
        o_next  = o[1:]
        o  = o[:-1]


        episode = dict( o=o.copy(),
                        u=u.copy(),
                        r=r.copy(),
                        o_next=o_next.copy(),
                        terminated=terminate.copy()
                    )

        for key in episode.keys():
            episode[key] = np.squeeze(np.array([episode[key]]), 0)
                 
        return episode

        




    def generate_episode_test(self, model):

        self.agents.policy.load_para(model)

        d = False
        obs = self.env_test.reset(None)
        

        while not d:
            

            actions  = self.agents.get_action(np.array(obs))

            obs2, reward, d, _ = self.env_test.step(actions)
          
            obs = obs2

        win_tag = 1 if reward[0]>13 else 0 
            
        return win_tag

