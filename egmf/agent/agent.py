class Agents:
    def __init__(self, args, is_cuda):

        from policy.fme_c import FME_C
        self.policy = FME_C(args, is_cuda)
        self.args = args

    
    def get_action(self, obs):
        return self.policy.get_action(obs)
    

    def train_critic_and_actor(self, batch, train_step, target_update): 
        self.policy.train_critic_and_actor(batch, train_step, target_update)