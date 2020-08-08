import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import Model


class ValueFunction(Model):
  '''
  This class is used both, for the state-value-function V(s) and for the state-action-value-function Q(s,a)
  '''
  def __init__(self, layers, input_dim, activation=tf.nn.leaky_relu, l2_reg=0.):
    super(ValueFunction, self).__init__()
    self.value_model = Sequential()
    self.value_model.add(Dense(input_dim=input_dim, units=layers[0], activation=activation,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                               kernel_initializer=tf.initializers.glorot_normal()))
    for units in layers[1:]:
      self.value_model.add(Dense(units=units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                               kernel_initializer=tf.initializers.glorot_normal()))
    self.value_model.add(Dense(units=1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                               kernel_initializer=tf.initializers.glorot_normal()))
    self.predict(tf.zeros((1, input_dim)))
    self.initial_weights = self.value_model.get_weights()

  def call(self, inputs, **kwargs):
    return self.value_model(inputs)

  def reset_weights(self):
    self.value_model.set_weights(self.initial_weights)

  @tf.function
  def eval(self, states, actions):
    """ This one only makes sense for Q(s,a) """
    return self.call(tf.concat((states, actions), axis=1))

  def update_from(self, other):
    for old, new in zip(self.trainable_variables, other.trainable_variables):
      old.assign(new)