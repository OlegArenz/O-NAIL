import tensorflow as tf
from gym.wrappers import TransformObservation
from tensorflow.data import Dataset
from models.ValueFunction import ValueFunction
from models.Policy import Policy
from common.utlis import roll_out_policy, process_demos
import numpy as np
import os
EPS = 1e-20
MAX_INPUT_TO_EXP = 20.0

class ONAIL:
    def __init__(self, name, env, eval_env, demo_path, policy_kwargs, Q_learning_rate, policy_learning_rate,
                 Q_kwargs, gamma, standardize_states, log_path, batch_size, nu_gradient_penalty,
                  ent_coef, num_nu_steps, num_policy_steps, kl_factor, use_fgan_loss,
                 all_states_are_init_states,
                 use_donsker_varadhan_for_pi_update,
                 use_lower_bound_reward_for_nu, initialize_bc, vali_steps, bc_vali_split, bc_iter, bc_learning_rate,
                 bc_policy_reg_coef, max_demos):
        self.policy_reg_coef = bc_policy_reg_coef
        self.bc_vali_split = bc_vali_split
        self.bc_iter = bc_iter
        self.bc_policy_lr = bc_learning_rate
        self.all_rews_det = []
        self.all_rews_stoch = []
        self.approximate_logpi_with_zeta = False
        self.use_donsker_varadhan_for_pi_update = use_donsker_varadhan_for_pi_update
        self.summary_writer = tf.summary.create_file_writer(log_path)
        self.summary_writer.set_as_default()
        self.log_path = log_path
        self.name = name
        self.gamma = gamma
        self.env = env
        self.eval_env = eval_env
        self.ent_coef = ent_coef
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.kl_factor = kl_factor
        self.Q_lr = Q_learning_rate
        self.policy_lr = policy_learning_rate
        expert_data = np.load(demo_path, allow_pickle=True)
        self.standardize_states = standardize_states
        self.batch_size = batch_size
        self.gradient_penalty_Q = nu_gradient_penalty
        if self.standardize_states:
            if "states" in expert_data.keys():
                self.expert_states, self.expert_actions, self.expert_next_states = expert_data["states"], expert_data["actions"], expert_data["next_states"]
                chosen_states = np.random.permutation(np.arange(len(self.expert_states)))[:max_demos]
                self.expert_states = self.expert_states[chosen_states]
                self.expert_actions = self.expert_actions[chosen_states]
                self.expert_actions = np.clip(self.expert_actions,-1+1e-6,1-1e-6)
                self.expert_next_states = self.expert_next_states[chosen_states]
                self.expert_step_ids = tf.ones(len(self.expert_states))
                self.expert_traj_ids = tf.range(len(self.expert_states))
                self.expert_steps = tf.ones((len(self.expert_states), 1))
                self.state_mean = np.mean(np.concatenate([self.expert_states, self.expert_next_states]), 0)
                self.state_std = np.std(np.concatenate([self.expert_states, self.expert_next_states]), 0)
                self.state_std[np.where(self.state_std == 0)] = 1.
                self.expert_states -= self.state_mean
                self.expert_states /= self.state_std
                self.expert_next_states -= self.state_mean
                self.expert_next_states /= self.state_std
                print("Warning. Using old demo format, steps and traj-ids not available")
            else:
                _, _, self.expert_states, self.expert_actions, \
                self.expert_next_states, self.expert_traj_ids, self.expert_steps, self.expert_step_ids, \
                self.state_mean, self.state_std = \
                    process_demos(expert_data, max_demos, True, True)
                self.expert_actions = np.clip(self.expert_actions,-1+1e-6,1-1e-6)
            self.env = TransformObservation(self.env, lambda obs: (obs - self.state_mean) / self.state_std)
            self.eval_env = TransformObservation(self.eval_env, lambda obs: (obs - self.state_mean) / self.state_std)
            os.makedirs(os.path.join(self.log_path, "checkpoints"), exist_ok=True)
            np.savez(os.path.join(self.log_path, "checkpoints", "state_stats.npz"), state_mean=self.state_mean,
                     state_std=self.state_std)
        else:
            _, _, self.expert_states, self.expert_actions, \
            self.expert_next_states, self.expert_traj_ids, self.expert_steps, self.expert_step_ids = \
                process_demos(expert_data['state_trajs'], expert_data['action_trajs'], max_demos, False, True)

        self.expert_weights = (1. - self.gamma) * self.gamma ** tf.cast(self.expert_steps, tf.float32)

        self.vali_steps = vali_steps
        if self.vali_steps > 0:
            train_indices, vali_indices = tf.split(tf.random.shuffle(tf.range(len(self.expert_states))),
                                                   [len(self.expert_states) - self.vali_steps, self.vali_steps])
            self.expert_states_vali = tf.gather(self.expert_states, vali_indices)
            self.expert_actions_vali = tf.gather(self.expert_actions, vali_indices)
            self.expert_next_states_vali = tf.gather(self.expert_next_states, vali_indices)
            self.expert_weights_vali = tf.gather(self.expert_weights, vali_indices)
            self.expert_steps_vali = tf.gather(self.expert_steps, vali_indices)
            self.expert_step_ids_vali = tf.gather(self.expert_step_ids, vali_indices)

            self.expert_states = tf.gather(self.expert_states, train_indices)
            self.expert_actions = tf.gather(self.expert_actions, train_indices)
            self.expert_next_states = tf.gather(self.expert_next_states, train_indices)
            self.expert_weights = tf.gather(self.expert_weights, train_indices)
            self.expert_steps = tf.gather(self.expert_steps, train_indices)
            self.expert_step_ids = tf.gather(self.expert_step_ids, train_indices)

        self.expert_demos = Dataset.from_tensor_slices((self.expert_states, self.expert_actions, self.expert_next_states,
                                                        self.expert_weights, self.expert_step_ids))
        self.expert_batch = iter(self.expert_demos\
            .shuffle(buffer_size=len(self.expert_states), reshuffle_each_iteration=True)\
            .repeat()\
            .batch(self.batch_size, drop_remainder=False))

        self.total_steps = 0
        self.num_nu_steps = num_nu_steps
        self.num_policy_steps = num_policy_steps

        self.use_lower_bound_reward_for_phi = use_lower_bound_reward_for_nu
        self.use_donsker_varadhan_loss = not use_fgan_loss

        self.rb_states = tf.zeros((0, self.num_states), dtype=tf.float32)
        self.rb_actions = tf.zeros((0, self.num_actions), dtype=tf.float32)
        self.rb_next_states = tf.zeros((0, self.num_states), dtype=tf.float32)
        self.rb_init_states = tf.zeros((0, self.num_states), dtype=tf.float32)

        self.learned_expert_step_rewards = tf.Variable(tf.zeros((len(self.expert_states), 1)), dtype=tf.float32)

        self.all_states_are_init_states = all_states_are_init_states
        initial_states = np.empty((1000, self.num_states))
        for i in range(1000):
            initial_states[i] = self.env.reset()
        self.tf_initial_states = Dataset.from_tensor_slices((np.float32(initial_states)))
        self.initial_states_batch = iter(self.tf_initial_states\
            .shuffle(buffer_size=10000, reshuffle_each_iteration=True)\
            .repeat()\
            .batch(self.batch_size, drop_remainder=False))

        self._build_models(Q_kwargs, policy_kwargs)
        self._build_optimizers()
        self._build_tensorboard_metrics()

        if initialize_bc:
            self.initialize_behavioral_cloning(self.bc_vali_split, self.bc_iter, self.policy_reg_coef)
        self.reference_policy.update_from(self.policy)

    def _build_models(self, q_kwargs, policy_kwargs):
        self.Q = ValueFunction(input_dim=self.num_states+self.num_actions, **q_kwargs)
        self.policy = Policy(num_states=self.num_states, num_actions=self.num_actions, **policy_kwargs) ## This one gets optimized and regularly overwrites reference_policy
        self.reference_policy = Policy(num_states=self.num_states, num_actions=self.num_actions,
                                       **policy_kwargs) ## This one is used for computing the density ratio / lower bound reward

    def _build_optimizers(self):
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.policy_lr)
        self.policy_optimizer_bc = tf.keras.optimizers.Adam(learning_rate=self.bc_policy_lr)
        self.policy_trainables = self.policy.trainable_variables
        self.reference_policy_trainables = self.reference_policy.trainable_variables
        self.Q_optimizer = tf.keras.optimizers.Adam(learning_rate=self.Q_lr)
        self.Q_trainables = self.Q.trainable_variables

    def _build_tensorboard_metrics(self):
        if self.vali_steps > 0:
            self.tb_vali_loss =  tf.keras.metrics.Mean('vali_loss', dtype=tf.float32)
        self.tb_nu_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.tb_entropy = tf.keras.metrics.Mean('entropy', dtype=tf.float32)
        self.tb_policy_loss = tf.keras.metrics.Mean('policy_loss', dtype=tf.float32)

    def update_reference(self):
        if not self.use_donsker_varadhan_for_pi_update:
            self.reference_policy.update_from(self.policy)
        else:
            self.policy.update_from(self.reference_policy)

    @tf.function
    def behavioral_cloning_loss(self, states, actions, reg_coeff):
        return -tf.reduce_mean(self.policy.log_density(states, actions)) \
               + reg_coeff * self.policy.get_regularization_loss(states)

    def initialize_behavioral_cloning(self, vali_split=0.1, max_iter=int(1e7), reg_coeff=1e-4):
        print("Initializing with Behavioral Cloning:")
        num_vali = tf.cast(tf.floor(vali_split * len(self.expert_states)), tf.int32)
        num_train = len(self.expert_states) - num_vali
        train_indices, vali_indices = tf.split(tf.random.shuffle(tf.range(len(self.expert_states))), [num_train, num_vali], axis=0)
        train_set = Dataset.from_tensor_slices((tf.gather(self.expert_states, train_indices),
                                                     tf.gather(self.expert_actions, train_indices)))
        train_batch = iter(train_set.shuffle(buffer_size=len(self.expert_states), reshuffle_each_iteration=True)\
            .repeat().batch(self.batch_size, drop_remainder=False))

        vali_states = tf.gather(self.expert_states, vali_indices)
        vali_actions = tf.gather(self.expert_actions, vali_indices)

        iter_since_last_best = 0
        best_vali_loss = np.inf
        best_params = self.policy.trainable_variables
        for i in tf.range(max_iter):
            states, actions = train_batch.get_next()
            with tf.GradientTape() as tape:
                tape.watch(self.policy_trainables)
                train_loss = self.behavioral_cloning_loss(states, actions, reg_coeff)
                gradients = tape.gradient(train_loss, self.policy_trainables)
            self.policy_optimizer_bc.apply_gradients(zip(gradients, self.policy_trainables))
            if i % 1000 == 0:
                if len(vali_states) > 0:
                    vali_loss = self.behavioral_cloning_loss(vali_states, vali_actions, reg_coeff=0)
                    if vali_loss < best_vali_loss:
                        iter_since_last_best = 0
                        best_vali_loss = vali_loss
                        best_params = self.policy_trainables
                    else:
                        iter_since_last_best += 1
                        if iter_since_last_best > 10:
                            break
                    print("train_loss: {} vali_loss: {} best: {}".format(train_loss, vali_loss, best_vali_loss))
                else:
                    print("train_loss: {}".format(train_loss))

        for params, best in zip(self.policy_trainables, best_params):
            params.assign(best)


    @tf.function
    def phi(self, states, actions, next_states, next_actions, ref_pi_logpds, ref_pi_logpds_next):
        phi = self.Q.eval(states, actions) \
              - self.gamma * self.Q.eval(next_states, next_actions)
        if self.use_lower_bound_reward_for_phi:
            phi += ref_pi_logpds - self.gamma * ref_pi_logpds_next
        return phi

    @tf.function
    def loss(self, init_states, expert_states, expert_actions, expert_next_states, step_weights, expert_step_ids):
        policy_samples_on_expert_nextStates, policy_log_densities_on_own_actions = self.reference_policy(expert_next_states)
        ref_pi_log_densities_on_expert_state_actions = self.reference_policy.log_density(expert_states, expert_actions)
        policy_samples_on_init_states = self.reference_policy.sample(init_states)
        linear_loss = - tf.reduce_mean((1. - self.gamma) * self.Q.eval(init_states, policy_samples_on_init_states))

        # The non-linear loss has the form sum(weights * exp(input_to_exp)), with an additional log in front when using
        # Donsker-Varadhan loss.
        phi = self.phi(expert_states, expert_actions, expert_next_states, policy_samples_on_expert_nextStates,
                       ref_pi_log_densities_on_expert_state_actions, policy_log_densities_on_own_actions)

        input_to_exp = phi + tf.math.log(step_weights)
        nonlinear_loss = tf.reduce_logsumexp(input_to_exp)
        sign = 1.

        if not self.use_donsker_varadhan_loss:
            if nonlinear_loss > MAX_INPUT_TO_EXP:
                # The value of the loss will be completely wrong, but its gradient should be okay
                nonlinear_loss = tf.stop_gradient(sign * tf.exp(MAX_INPUT_TO_EXP)) * nonlinear_loss
            else:
                nonlinear_loss = sign * tf.exp(nonlinear_loss)

        loss = linear_loss + nonlinear_loss
        return loss

    @tf.function
    def policy_update_step(self, states):
        with tf.GradientTape() as tape:
            tape.watch(self.policy_trainables)
            actions, unsquashed = self.policy.sample(states, return_unsquashed=True)
            logpds = self.policy.log_density(states, unsquashed, actions_are_unsquashed=True)
            #If the policy does not do action squashing we can use the analytic value: entropy = self.policy.entropy
            entropy = -tf.reduce_mean(logpds)
            entropy_loss = -self.ent_coef * entropy
            if self.kl_factor > 0:
                old_logpds = self.reference_policy.log_density(states, unsquashed, actions_are_unsquashed=True)
                kl_loss = self.kl_factor * tf.reduce_mean(logpds - old_logpds)
            else:
                kl_loss = tf.constant(0.)
            policy_loss = tf.reduce_mean(self.Q.eval(states, actions)) + entropy_loss + kl_loss
            gradients = tape.gradient(policy_loss, self.policy_trainables)
            self.policy_optimizer.apply_gradients(zip(gradients, self.policy_trainables))
        self.tb_entropy(entropy)
        self.tb_policy_loss(policy_loss)
        return policy_loss

    @tf.function
    def policy_update_step_dv(self, init_states, expert_states, expert_actions, expert_next_states, expert_weights,
                              step_ids):
        with tf.GradientTape() as tape:
            tape.watch(self.reference_policy_trainables)
            policy_loss = -self.loss(init_states=init_states, expert_states=expert_states,
                               expert_actions=expert_actions, expert_next_states=expert_next_states,
                               step_weights=expert_weights, expert_step_ids=step_ids)
            gradients = tape.gradient(policy_loss, self.reference_policy_trainables)
            self.policy_optimizer.apply_gradients(zip(gradients, self.reference_policy_trainables))
        self.tb_policy_loss(policy_loss)
        return policy_loss

    @tf.function
    def Q_update_step(self, init_states, expert_states, expert_actions, expert_next_states, expert_weights, step_ids):
        with tf.GradientTape() as tape:
            tape.watch(self.Q_trainables)
            with tf.GradientTape() as tape2:
                tape2.watch([expert_states, expert_actions])
                dre_nu = self.Q.eval(expert_states, expert_actions) #self.density_ratio_nu(expert_states, expert_actions, expert_next_states)
                gradient_wrt_input = tape2.gradient(dre_nu, [expert_states, expert_actions])
                gradient_penalty = self.gradient_penalty_Q * tf.reduce_mean(
                    tf.square(tf.norm(tf.concat(gradient_wrt_input, 1), axis=1) - 1.))
            Q_loss = self.loss(init_states=init_states, expert_states=expert_states,
                               expert_actions=expert_actions, expert_next_states=expert_next_states,
                               step_weights=expert_weights, expert_step_ids=step_ids)
            total_loss = Q_loss + gradient_penalty + tf.add_n(self.Q.losses)
            gradients = tape.gradient(total_loss, self.Q_trainables)
            self.Q_optimizer.apply_gradients(zip(gradients, self.Q_trainables))
        self.tb_nu_loss(Q_loss)
        return Q_loss

    def learn_one_step(self):
        if self.vali_steps > 0 and self.total_steps % 1000 == 0:
            self.tb_vali_loss(self.loss(self.expert_states_vali, self.expert_states_vali,
                                        self.expert_actions_vali, self.expert_next_states_vali,
                                        self.expert_weights_vali, self.expert_step_ids_vali))

        steps_this_outer_loop = self.total_steps % (self.num_nu_steps + self.num_policy_steps)
        if steps_this_outer_loop == 0:
            self.update_reference()

        expert_states, expert_actions, expert_next_states, expert_weights, step_ids = self.expert_batch.get_next()
        if self.all_states_are_init_states:
            init_states = expert_states
            expert_weights = tf.ones_like(expert_weights)
            expert_weights = expert_weights / tf.reduce_sum(expert_weights)
        else:
            init_states = self.initial_states_batch.get_next()

        if steps_this_outer_loop < self.num_nu_steps:
            # Update Q-Function
            self.Q_update_step(init_states, expert_states, expert_actions, expert_next_states, expert_weights,
                                   step_ids)
        else:
            # Update Policy
            if self.use_donsker_varadhan_for_pi_update:
                self.policy_update_step_dv(init_states, expert_states, expert_actions, expert_next_states, expert_weights,
                                   step_ids)
            else:
                self.policy_update_step(tf.concat(expert_states, axis=0))

        if self.total_steps % 10000 == 0:
            self._evaluate_and_log()
            self._write_tb_summaries()
            self.reference_policy.save(os.path.join(self.log_path, "checkpoints", "", "steps_" + str(self.total_steps)))
        self.total_steps += 1

    def _write_tb_summaries(self):
        with self.summary_writer.as_default():
            tf.summary.scalar(self.name+'/nu_loss', self.tb_nu_loss.result(), step=self.total_steps)
            tf.summary.scalar(self.name+'/entropy', self.tb_entropy.result(), step=self.total_steps)
            tf.summary.scalar(self.name+'/policy_loss', self.tb_policy_loss.result(), step=self.total_steps)
            if self.vali_steps > 0:
                tf.summary.scalar(self.name + '/vali_loss', self.tb_vali_loss.result(), step=self.total_steps)
                self.tb_vali_loss.reset_states()
            self.tb_nu_loss.reset_states()
            self.tb_entropy.reset_states()
            self.tb_policy_loss.reset_states()

    @tf.function
    def density_ratio_nu(self, states, actions, next_states, lower_bound_ratio=False):
        next_actions, logp_own = self.reference_policy(next_states)
        phi = self.Q.eval(states, actions) \
              - self.gamma * self.Q.eval(next_states, next_actions)

        if self.use_lower_bound_reward_for_phi:
            if lower_bound_ratio:
                phi += -self.gamma * logp_own
            else:
                logp_expert = self.reference_policy.log_density(tf.convert_to_tensor(states),
                                                                tf.convert_to_tensor(actions))
                phi += logp_expert - self.gamma * logp_own
        return phi

    def _evaluate_and_log(self):
        _, rews_det = roll_out_policy(env=self.eval_env, policy=self.policy, max_episodes=5, reset_env=True,
                                      render=False, deterministic=True)
        _, rews_stoch = roll_out_policy(env=self.eval_env, policy=self.policy, max_episodes=5, reset_env=True,
                                        render=False, deterministic=False)
        mean_return_det = np.mean([np.sum(rew_traj) for rew_traj in rews_det])
        mean_return_stoch = np.mean([np.sum(rew_traj) for rew_traj in rews_stoch])
        self.all_rews_det += [rews_det]
        self.all_rews_stoch += [rews_stoch]
        np.savez(os.path.join(self.log_path, "rews.npz"), rews_det=self.all_rews_det, rews_stoch=self.all_rews_stoch)
        tf.summary.scalar(self.name+'/mean_ret_det', data=mean_return_det, step=self.total_steps)
        tf.summary.scalar(self.name+'/mean_ret_stoch', data=mean_return_stoch, step=self.total_steps)
        print("Steps: {} \t mean return: {:.4f}".format(self.total_steps, mean_return_stoch))