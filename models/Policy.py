import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from math import pi

class Policy(Model):
  def __init__(self, num_states, num_actions, layers, squash_actions, state_dependent_std, log_std_min=-5,
               log_std_max=0, activation=tf.nn.leaky_relu):
    super(Policy, self).__init__()
    self.num_actions = num_actions
    self.num_states = num_states
    self.squash_actions = squash_actions
    if self.squash_actions:
      self.bijector = tfp.bijectors.Tanh()
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    self.mean_model = Sequential()
    self.mean_model.add(Dense(units=layers[0], input_dim=num_states, activation=activation))
    for units in layers[1:]:
      self.mean_model.add(Dense(units=units, activation=activation))
    self.mean_model.add(Dense(units=num_actions, activation=None))
    self.state_dependent_std = state_dependent_std
    if self.state_dependent_std:
      self.log_std_model = Sequential()
      self.log_std_model.add(Dense(units=layers[0], input_dim=num_states, activation=activation))
      for units in layers[1:]:
        self.log_std_model.add(Dense(units=units, activation=activation))
      self.log_std_model.add(Dense(units=num_actions, activation=None))
    else:
      self.log_std_unclipped = tf.Variable(initial_value=tf.zeros(self.num_actions), trainable=True)
    self.call(tf.zeros((1, self.num_states)))

  @tf.function
  def get_log_stds_unclipped(self, states):
    if self.state_dependent_std:
      return self.log_std_model(states)
    else:
      return self.log_std_unclipped

  @tf.function
  def log_std(self, states):
    log_stds_unclipped = self.get_log_stds_unclipped(states)
    return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (tf.tanh(log_stds_unclipped) + 1)

  @tf.function
  def std(self, states):
    return tf.exp(self.log_std(states))

  @tf.function
  def get_regularization_loss(self, states):
    return tf.reduce_sum(tf.square(self.mean_model(states))) + tf.reduce_sum(tf.square(self.log_std(states)))


  def call(self, inputs, **kwargs):
    actions = self.sample(inputs)
    likelihoods = self.log_density(inputs, actions)
    return actions, likelihoods

  @tf.function
  def step_tf(self, inputs, deterministic=False):
    if deterministic:
      if self.squash_actions:
        return self.bijector.forward(self.mean_model(inputs))
      else:
        return self.mean_model(inputs)
    else:
      return self.sample(inputs)

  def step(self, inputs, deterministic=False):
    return self.step_tf(tf.convert_to_tensor(inputs), deterministic).numpy()

  def gaussian_likelihood(self, mu, actions, log_stds):
    return tf.reduce_sum(-0.5 * tf.square((actions - mu) / tf.exp(log_stds)) - log_stds - 0.5 * tf.math.log(2 * pi),
                         axis=1)

  @tf.function
  def log_density(self, states, actions, actions_are_unsquashed=False):
    log_stds = self.log_std(tf.convert_to_tensor(states))
    actions = tf.convert_to_tensor(actions)
    mu = self.mean_model(states)
    if self.squash_actions:
      if actions_are_unsquashed:
        unsquashed_actions = actions
        log_det_jac = -tf.reduce_sum(self.bijector.forward_log_det_jacobian(actions, 0),1)
      else:
        unsquashed_actions = self.bijector.inverse(actions)
        log_det_jac = tf.reduce_sum(self.bijector.inverse_log_det_jacobian(actions, 0),1)
      likelihoods = self.gaussian_likelihood(mu, unsquashed_actions, log_stds) + log_det_jac
    else:
      likelihoods = self.gaussian_likelihood(mu, actions, log_stds)
    return tf.expand_dims(likelihoods, -1)

  def sample(self, inputs, num_actions=1, return_unsquashed=False):
    mu = self.mean_model(inputs)
    std = self.std(inputs)
    if num_actions > 1:
      mu = tf.tile(tf.expand_dims(mu,axis=1), [1,num_actions, 1])
      std = tf.tile(tf.reshape(self.std,(1,1,self.num_actions)), [1,num_actions,1])
      unsquashed_actions = mu + tf.random.normal(tf.shape(mu)) * std
    else:
      unsquashed_actions = mu + tf.random.normal(tf.shape(mu)) * std
    if self.squash_actions:
      actions = self.bijector.forward(unsquashed_actions)
    else:
      actions = unsquashed_actions
    if return_unsquashed:
      return actions, unsquashed_actions
    else:
      return actions

  def update_from(self, other_policy):
    for old, new in zip(self.trainable_variables, other_policy.trainable_variables):
      old.assign(new)