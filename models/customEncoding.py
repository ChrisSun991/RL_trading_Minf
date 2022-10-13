from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
# import numpy as np
import tf_agents.utils.eager_utils
import tflearn
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils
from tf_agents.utils import common as common_utils

tf.config.run_functions_eagerly(True)
# from tf_agents.agents.ddpg.critic_network import CriticNetwork

# from tf_agents.agents.td3 import ActorNetwork
# tf_agents.utils.eager_utils.future_in_eager_mode()

class ActorNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 s_dim,
                 # state_spec,
                 a_dim,
                 action_spec,
                 predictor_type,
                 use_batch_norm,
                 name='ActorNetwork'):

        # Actions will not be flatten not as example due
        # to the fact we are performing multi-asset trading.
        self._action_spec = action_spec
        self.a_dim = a_dim
        self.inputs = tflearn.input_data(shape=[None] + s_dim + [1], name="input")
        self.use_batch_norm = use_batch_norm
        # self.input_tensor_spec = input_tensor_spec
        windows_length = self.inputs.get_shape()[2]
        num_stocks = self.inputs.get_shape()[1]
        hidden_dim = 32

        assert predictor_type in ['cnn', 'lstm'], 'type must be either cnn or lstm'
        self.predictor_type = predictor_type
        # self._flat_preprocessing_layers = None

        super(ActorNetwork, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        layers = []
        if predictor_type == 'cnn':
            layers.append(tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 3)))
            # filters = 32, filtersize = 1,3 kernel size (5,5)
            if use_batch_norm:
                layers.append(tf.keras.layers.BatchNormalization)
            layers.append(tf.keras.layers.Activation(tf.nn.relu))
            layers.append(tf.keras.layers.Conv2D(filters=32, kernel_size=(1, windows_length - 2)))
            if use_batch_norm:
                layers.append(tf.keras.layers.BatchNormalization)
            layers.append(tf.keras.layers.Flatten)
        elif predictor_type == 'lstm':
            layers.append(tf.keras.layers.reshape((-1, windows_length, 1)))
            layers.append(tf.keras.layers.LSTM(hidden_dim))
            layers.append(tf.keras.layers.reshape((-1, num_stocks, hidden_dim)))
            layers.append(tf.keras.layers.Flatten)
        else:
            raise NotImplementedError

        layers.append(tf.keras.layers.Dense(64))
        if self.use_batch_norm:
            layers.append(tf.keras.layers.BatchNormalization)
        layers.append(tf.keras.layers.Activation(tf.nn.relu))
        layers.append(tf.keras.layers.Dense(64))
        if self.use_batch_norm:
            layers.append(tf.keras.layers.BatchNormalization)
        layers.append(tf.keras.layers.Activation(tf.nn.relu))
        layers.append(tf.keras.layers.Dense(self.a_dim[0], activation="softmax",
                                            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003,
                                                                                                   maxval=0.003)))
        self._postprocessing_layers = layers

    def call(self,
             observation,
             step_type=None,
             network_state=(),
             training=False):
        del step_type
        print("tf.executing_eagerly() =", tf.executing_eagerly())
        outer_rank = nest_utils.get_outer_rank(
            observation, self.input_tensor_spec
        )
        # We use batch_squash here in case the observations have a time sequence
        # compoment.
        # if self.predictor_type == 'cnn':
        batch_squash = utils.BatchSquash(outer_rank)
        observation = tf.nest.map_structure(batch_squash.flatten, observation)

        states = observation
        for layer in self._postprocessing_layers:
            states = layer(states, training=training)
        # if self.predictor_type == 'cnn':

        # states = tf.nest.map_structure(batch_squash.unflatten, states)
        actions = common_utils.scale_to_spec(states, self._action_spec)
        actions = batch_squash.unflatten(actions)

        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state


class CriticNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 # s_dim,
                 # state_spec,
                 a_dim,
                 # action_spec,
                 predictor_type,
                 use_batch_norm,
                 name='CriticNetwork'):
        observation_spec, action_spec = input_tensor_spec
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.a_dim = a_dim
        windows_length = self.inputs.get_shape()[2]
        num_stocks = self.inputs.get_shape()[1]
        hidden_dim = 32
        stock_net = []
        if predictor_type == 'cnn':
            stock_net.append(tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 3)))
            # filters = 32, filtersize = 1,3 kernel size (5,5)
            if use_batch_norm:
                stock_net.append(tf.keras.layers.BatchNormalization)
            stock_net.append(tf.keras.layers.Activation(tf.nn.relu))
            stock_net.append(tf.keras.layers.Conv2D(filters=32, kernel_size=(1, windows_length - 2)))
            if use_batch_norm:
                stock_net.append(tf.keras.layers.BatchNormalization)
            stock_net.append(tf.keras.layers.Flatten)
        elif predictor_type == 'lstm':
            stock_net.append(tf.keras.layers.reshape((-1, windows_length, 1)))
            stock_net.append(tf.keras.layers.LSTM(hidden_dim))
            stock_net.append(tf.keras.layers.reshape((-1, num_stocks, hidden_dim)))
            stock_net.append(tf.keras.layers.Flatten)
        else:
            raise NotImplementedError

        processing_net = []
        if use_batch_norm:
            processing_net.append(tf.keras.layers.BatchNormalization)
        processing_net.append(tf.keras.layers.Activation(tf.nn.relu))
        processing_net.append(tf.keras.layers.Dense(self.a_dim[0], activation="softmax",
                                                    kernel_initializer=tf.keras.initializers.RandomUniform(
                                                        minval=-0.003,
                                                        maxval=0.003)))

        super(CriticNetwork, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
        self._trading_net = stock_net
        self._processing_net = processing_net

    def call(self, inputs, step_type=(), network_state=(), training=False):
        observations, actions = inputs
        del step_type
        observations = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
        actions = tf.cast(tf.nest.flatten(actions)[0], tf.float32)

        for layers in self._trading_net:
            actions = layers(actions, training=training)
        for layers in self._trading_net:
            observations = layers(observations, training=training)

        join_networks = tf.concat([observations, actions], 1)
        for layer in self._processing_net:
            join_networks = layer(join_networks, training=training)

        return tf.reshape(join_networks, [-1]), network_state
