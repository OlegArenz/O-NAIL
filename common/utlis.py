import numpy as np
import tensorflow as tf

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def roll_out_policy(env, policy, max_episodes=int(1e8), max_samps=np.inf, reset_env=False, render=False, deterministic=False):
    all_trajs = []
    all_rews = []
    if max_episodes==int(1e8) and np.isinf(max_samps):
        print("you should either specify max_episodes or max_samps!")
        return all_trajs, all_rews
    num_steps = 0
    for i in range(max_episodes):
        this_rews = []
        if reset_env:
            obs = env.reset()
        else:
            obs = env.env._get_obs()
        this_obs = []
        this_actions = []
        while True:
            action = np.squeeze(policy.step(np.atleast_2d(obs), deterministic=deterministic)[0])
            action = np.clip(action, -1, 1)
            this_obs.append(obs)
            this_actions.append(action)
            obs, rewards, dones, info = env.step(action)
            this_rews.append(rewards)
            if render:
                env.render()
            if dones:
                env.reset()
                all_trajs.append(np.hstack((this_obs, this_actions)))
                all_rews.append(this_rews)
                break
            num_steps += 1
            if num_steps > max_samps:
                all_trajs.append(np.hstack((this_obs, this_actions)))
                all_rews.append(this_rews)
                return all_trajs, all_rews
    return all_trajs, all_rews

def process_trajectories(trajectories, num_states, return_as_tensors=False):
    states = []
    actions = []
    next_states = []
    next_actions = []
    init_states = []
    dones = []
    for traj in trajectories:
        states.append(traj[:-1, :num_states])
        actions.append(traj[:-1, num_states:])
        next_states.append(traj[1:, :num_states])
        next_actions.append(traj[1:, num_states:])
        init_states.append(traj[0:1, :num_states])
        this_dones = np.zeros(len(traj), dtype=np.bool)
        this_dones[-1] = True
        dones.append(this_dones)
    if return_as_tensors:
        return tf.convert_to_tensor(np.concatenate(states), tf.float32), \
               tf.convert_to_tensor(np.concatenate(actions), tf.float32), \
               tf.convert_to_tensor(np.concatenate(next_states), tf.float32), \
               tf.convert_to_tensor(np.concatenate(next_actions), tf.float32), \
               tf.convert_to_tensor(np.concatenate(init_states), tf.float32), \
               tf.convert_to_tensor(np.concatenate(dones), tf.bool)
    else:
        return np.float32(np.concatenate(states)), np.float32(np.concatenate(actions)), \
               np.float32(np.concatenate(next_states)),np.float32(np.concatenate(next_actions)),\
               np.float32(np.concatenate(init_states)), np.concatenate(dones)

def process_demos(expert_data, max_demos, standardize, convert_to_tensor):
    if "state_trajs" in expert_data.keys() and "action_trajs" in expert_data.keys():
        state_trajs_in = expert_data["state_trajs"]
        action_trajs_in = expert_data["action_trajs"]
    elif "state_action_traj_stoch" in expert_data.keys() and "state_shape" in expert_data.keys():
        num_states = expert_data["state_shape"][0]
        sa_traj = expert_data["state_action_traj_stoch"]
        state_trajs_in = np.array([traj[:, :num_states] for traj in sa_traj if len(traj)==1000])
        action_trajs_in = np.array([traj[:, num_states:] for traj in sa_traj if len(traj)==1000])
    else:
        raise Exception("Unsupported demo format!")

    # Randomly sample trajectories
    num_trajs_in = state_trajs_in.shape[0]
    chosen_trajectories = np.random.permutation(np.arange(num_trajs_in))[:max_demos]
    state_trajs = np.float32(state_trajs_in[chosen_trajectories])
    action_trajs = np.float32(action_trajs_in[chosen_trajectories])

    num_trajs = state_trajs.shape[0]
    horizon = state_trajs.shape[1]
    num_states = state_trajs.shape[2]
    num_actions = action_trajs.shape[2]

    if standardize:
        state_mean = np.mean(state_trajs.reshape((-1, num_states)), axis=0)
        state_std = np.std(state_trajs.reshape((-1, num_states)), axis=0)
        state_std[np.where(state_std==0)] += 1e-5
        state_trajs -= state_mean.reshape((1,1,-1))
        state_trajs /= state_std.reshape((1,1,-1))

    steps = np.tile(np.arange(horizon)[np.newaxis], (num_trajs, 1))[:,:,np.newaxis]
    traj_ids = np.tile(np.arange(num_trajs)[:,np.newaxis], (1, horizon))[:,:,np.newaxis]

    expert_state_mat = state_trajs[:, :horizon - 1]
    expert_action_mat = action_trajs[:, :horizon - 1]
    expert_states = expert_state_mat.reshape((-1,num_states))
    expert_actions = expert_action_mat.reshape((-1,num_actions))
    expert_next_states = state_trajs[:, 1:horizon].reshape((-1,num_states))
    expert_traj_ids = traj_ids[:, :horizon - 1].reshape((-1))
    expert_steps = steps[:, :horizon - 1].reshape((-1,1))
    expert_step_ids = np.arange(len(expert_steps))

    if convert_to_tensor:
        expert_states = tf.convert_to_tensor(expert_states)
        expert_actions = tf.convert_to_tensor(expert_actions)
        expert_next_states = tf.convert_to_tensor(expert_next_states)
        expert_traj_ids = tf.convert_to_tensor(expert_traj_ids)
        expert_steps = tf.convert_to_tensor(expert_steps)
        expert_step_ids = tf.convert_to_tensor(expert_step_ids)

    if standardize:
        return expert_state_mat, expert_action_mat, expert_states, expert_actions, expert_next_states, expert_traj_ids,\
               expert_steps, expert_step_ids, state_mean, state_std
    else:
        return expert_state_mat, expert_action_mat,expert_states, expert_actions, expert_next_states, expert_traj_ids, \
               expert_steps, expert_step_ids,
