import argparse

"""
Here are the param for the training

"""


def get_common_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--env_num', type=int, default=16, help='number of the workers')

    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')

    # parser.add_argument('--n_steps', type=int, default=4e6, help='total time steps')
    parser.add_argument('--n_eps', type=int, default=5000, help='total eps')
    
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--alpha', type=float, default=0.05, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')

    parser.add_argument('--evaluate_cycle', type=int, default=50, help='how often to evaluate the model, every 100 ep')
    parser.add_argument('--evaluate_epoch', type=int, default=6, help='number of the epoch to evaluate the agent, test 100 ep')


    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--log_dir', type=str, default='./log', help='log directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')

    
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--test', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='1', help='log directory of the policy')

    args = parser.parse_args()
    return args




# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 128

    args.lr = 5e-4

    # experience replay
    args.batch_size = 2000
    args.buffer_size = int(4e6)

    # how often to save the model
    args.save_cycle = 50

    # how often to update the target_net
    args.target_update_cycle = 10

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args




