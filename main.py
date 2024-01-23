from runner import Runner

from common.arguments import get_common_args, get_mixer_args

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

if __name__ == '__main__':
                                      
    args = get_common_args()       

    args = get_mixer_args(args)   

    env_info = {"n_actions": 3, "n_agents_per_party": 2, "state_shape": 21, "obs_shape": 21, "episode_limit": 50}
    args.n_actions = env_info["n_actions"]
    args.n_agents_per_party = env_info["n_agents_per_party"] 
    args.obs_shape = env_info["obs_shape"]
    args.act_limit = 1.0

    os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES

    runner = Runner(args)   
    
    runner.run()

