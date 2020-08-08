import tensorflow as tf
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
import argparse
from onail.ONAIL import ONAIL
from argparse import Namespace
from configs.offline_il_configs import OFFLINE_IL_CONFIGS
from sys import argv
import numpy as np
from common.utlis import boolean_string
import random
from gym.envs.registration import register
register(id='AntNoEarlyTerm-v3', entry_point='gym.envs.mujoco.ant_v3:AntEnv',
         kwargs={'terminate_when_unhealthy': False}, max_episode_steps=1000)
register(id='HopperNoEarlyTerm-v3', entry_point='gym.envs.mujoco.hopper_v3:HopperEnv',
         kwargs={'terminate_when_unhealthy': False}, max_episode_steps=1000)
register(id='WalkerNoEarlyTerm-v3', entry_point='gym.envs.mujoco.walker2d_v3:Walker2dEnv',
         kwargs={'terminate_when_unhealthy': False}, max_episode_steps=1000)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mle_params', action="append",
                        help="path to npz file storing initial policy params. Parameter can be repeated when learning"
                             "in multiple envinronments.", type=str)
    parser.add_argument('--path', help="path for logging", type=str)
    parser.add_argument('--seed', help="random seed", type=int, default=0)
    parser.add_argument('--max_iter', help="number of learning iterations", type=int, default=int(1e7))
    parser.add_argument('--bc_only', help='dont do any learning after behavioral cloning',
                        type=boolean_string, default=False)
    parser.add_argument('--config_name',
                        help='If not none, unspecified arguments will use the specified preset. The name should correspond'
                             ' to a key in configs/rl_configs.py or (if --il_algo is set) configs/offline_il_configs.py',
                        type=str, default="HALFCHEETAH_ONAIL")
    parser.add_argument('--env_name', action='append', type=str,
                        help='name of the environment. This parameter can be repeated to learn in several environments '
                             'with a shared reward function. The parameter --il_demo_path must be provided as many times'
                             'as --env-name to provide demonstrations for each environment')
    parser.add_argument('--policy_layers', help='list of widths for the policy network', type=int, nargs='+',
                        required=False)
    parser.add_argument('--policy_squash_actions', help='if true the actions are squashed using tanh',
                        type=boolean_string, default=True)
    parser.add_argument('--policy_state_dependent_std', help='should the standard deviation of the Gaussian depend '
                                                              'on the state?', type=boolean_string, default=True)

    parser.add_argument('--il_all_states_are_init_states', help="treat every expert state as an initial state",
                        type=boolean_string, default=True)
    parser.add_argument('--il_vali_steps', help='number of samples used to compute vali/test loss',
                        type=int, default=0)
    parser.add_argument('--il_initialize_bc', help='initilize policy by maximizing likelihood of demonstrations',
                        type=boolean_string, default=True)
    parser.add_argument('--il_bc_vali_split', help='ratio for validation data for behavioral cloning',
                        type=float, default=0.1)
    parser.add_argument('--il_bc_iter', help='maximum number of iterations for behavioral cloning',
                        type=int, default=int(1e7))
    parser.add_argument('--il_bc_learning_rate', help='learning rate for behavioral cloning',
                        type=float, default=1e-4)
    parser.add_argument('--il_bc_policy_reg_coef', help='policy regularization for behavioral cloning',
                        type=float, default=1e-5)



    parser.add_argument('--il_kl_factor', help='weight of the KL wenn optimizing policy w.r.t. Q function',
                        type=float, default=0.0)
    parser.add_argument('--il_gamma', help='discount factor', type=float, default=0.99)
    parser.add_argument('--il_batch_size', help='batch_size for expert and replay samples', type=int, default=256)
    parser.add_argument('--Q_layers', help='list of widths for the nu network', type=int, nargs='+', required=False)
    parser.add_argument('--Q_l2_reg', help='l2 regularization for nu network', type=float, default=0.)
    parser.add_argument('--il_Q_learning_rate', help='learning rate for nu network', type=float, default=1e-3)
    parser.add_argument('--il_nu_gradient_penalty', help='gradient penalty for nu network', type=float, default=10.)
    parser.add_argument('--il_policy_learning_rate', help='learning rate for policy network', type=float, default=1e-5)
    parser.add_argument('--il_max_demos', help='maximum number of expert trajectories to use', type=int, default=int(1e8))
    parser.add_argument('--il_demo_path', action='append',
                        help='path to npz storing expert trajs. This parameter can be repeated when learning in several '
                             'environments. The parameter --il_demo_path must be provided as many times'
                             'as --env-name to provide demonstrations for each environment',
                        type=str)
    parser.add_argument('--il_ent_coef', help='coefficient for policy entropy regularization', type=float, default=0.)
    parser.add_argument('--il_standardize_states', help='should we standardize the states?', type=boolean_string, default=True)
    parser.add_argument('--il_num_nu_steps',
                        help='number of critic gradient steps before updating policy', type=int, default=1)
    parser.add_argument('--il_num_policy_steps',
                        help='number of policy gradient steps before updating critic', type=int, default=1)

    parser.add_argument('--il_use_donsker_varadhan_for_pi_update',
                        help="If False, the policy is updated to maximize the Q-Function",
                        type=boolean_string, default=False)
    parser.add_argument('--il_use_fgan_loss', help="Use F-GAN loss instead of Donsker-Varadhan",
                        type=boolean_string, default=False)
    parser.add_argument('--il_use_lower_bound_reward_for_nu',
                        help="Learn the Q-Function for the lower bound reward (not for the density ratio)",
                        type=boolean_string, default=False)
    return parser


def overwrite_dict_with_kwargs(prefix, current_dict, args, ignore_keys=[""]):
    '''
        extracts all commandline parameters that start with a given prefix (e.g. "Q_", "il_", etc.)
        and stores the corresponding values in the given dictionary under the key that is obtained when removing
        the prefix. Hence args = {"trpo_entcoeff" : 1e-2} would add (or overwrite) {"entcoeff" : 1e-2} to current_dict
    '''
    arg_dict = args.__dict__
    keys = list(arg_dict.keys())
    params = [key.startswith(prefix) for key in keys]
    param_keys = np.array(keys)[np.where(params)]
    for param_key in param_keys:
        if param_key not in ignore_keys:
            current_dict[param_key[len(prefix):]] = arg_dict[param_key]

def build_kwargs_from_command_line_args(args):
    offpolicy_il_kwargs = dict()
    policy_kwargs = dict()
    Q_kwargs = dict()
    overwrite_dict_with_kwargs("il_", offpolicy_il_kwargs, args)
    overwrite_dict_with_kwargs("policy_", policy_kwargs, args)
    overwrite_dict_with_kwargs("Q_", Q_kwargs, args)
    return offpolicy_il_kwargs, policy_kwargs, Q_kwargs,

def get_final_experiment_config():
    def_args = argparser().parse_args()
    if def_args.config_name is not None:
        conf_from_file = OFFLINE_IL_CONFIGS[def_args.config_name]
        final_conf = def_args.__dict__
        for key in conf_from_file.keys():
            # We only want to override those argparse parameters that have not been explicitly specified (defaults)
            # Explicitly specified command line parameters have higher priority than those from the config file
            if "--" + key not in argv:
                final_conf[key] = conf_from_file[key]
        return Namespace(**final_conf)
    else:
        return def_args

def run_experiment(args):
    start_time = datetime.now().strftime('%Y-%m-%d%_H:%M:%S.%f')
    if args.path is None:
        path = os.path.join("/tmp", start_time, "")
    else:
        path = os.path.join(args.path, start_time, "")

    offpolicy_il_kwargs, policy_kwargs, Q_kwargs,= build_kwargs_from_command_line_args(args)
    set_seed(args.seed)

    this_demo_path = offpolicy_il_kwargs['demo_path'][0]
    this_env = gym.make(args.env_name[0])
    this_eval_env = gym.make(args.env_name[0])
    this_path = os.path.join(path, "_"+args.env_name[0], "")
    this_offpolicy_il_kwargs = dict(**offpolicy_il_kwargs)
    this_offpolicy_il_kwargs.update(
            dict(name="il_{}".format(0), policy_kwargs=policy_kwargs, Q_kwargs=Q_kwargs,
                 log_path=this_path, demo_path=this_demo_path, env=this_env, eval_env=this_eval_env))

    il_algorithm = ONAIL(**this_offpolicy_il_kwargs)

    for iters in range(args.max_iter):
      il_algorithm.learn_one_step()

if __name__ == "__main__":
    args = get_final_experiment_config()
    run_experiment(args)