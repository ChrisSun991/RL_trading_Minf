import keras
import tf_agents.agents.ddpg.actor_network as Actor
import tensorflow as tf
import tflearn
import tf_agents.networks
from models.customEncoding import *
from tf_agents.networks import encoding_network
from tf_agents.networks import lstm_encoding_network

from models.StockPredictor import stock_predictor


## DDPG and TD3 have the same format in Actor
class StockActor(keras.Model):  # /Actor):
    def __init__(self, a_dim, a_bound, predictor_type,  # output_tensor_spec
                 use_batch_norm):
        # self.learning_rate = learning_rate
        # self.tau = tau
        # self.batch_size = batch_size
        super(StockActor, self).__init__()
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.bn = tflearn.normalization.batch_normalization
        self.relu_act = tflearn.activations.relu
        # self.s_dim = s_dim
        # super(Actor, self).__init__(input_tensor_spec, output_tensor_spec)

        # self.fully_connected_net = tflearn.fully_connected(n_units=64)
        # self.batch_norm_lay = tflearn.normalization.batch_normalization()

    def call(self, s_dim):
        """
        DDPG_stock actor:
        	input = tflearn.input_data(shape=[None] + self.s_dim + [1], name='input’)
			x= stock_predictor(input) - LSTM/ CNN
			x = fully_conneted(x,64)
			“bn”
			x = relu(x)
			x = fully_conneted(x,64)
			“bn”
			x = relu(x)
			Weights = tf.init.uniform(minval=-0.003, maxval=0.003)
			x = fully_connected(x, a_dim[0], activation = softmax, weights_init = weights)
			scaled_out = tf.multi(out, a_bound)
        """
        input = tflearn.input_data(shape=[None] + s_dim + [1], name="input")  ##
        x = stock_predictor(input, self.predictor_type, self.use_batch_norm)
        x = tflearn.fully_connected(x, 64)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.relu_act(x)
        x = tflearn.fully_connected(x, 64)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.relu_act(x)
        w_initializations = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        output = tflearn.fully_connected(x, self.a_dim[0], activation='softmax', weights_init=w_initializations)
        scaled_outputs = tf.multiply(output, self.a_bound)
        return input, output, scaled_outputs


# ## TF_agents requires tf_agents.network.Network kind
# class StockActorV2(tf_agents.networks.Network):
#     def __init__(self,
#                  input_tensor_spec,
#                  # observation_specation_spec,
#                  action_spec,
#                  state_spec,
#                  s_dim,
#                  a_dim,
#                  predictor_type,
#                  use_batch_norm,
#                  name="ActorNetwork"):
#         super(StockActorV2, self).__init__(
#             input_tensor_spec=observation_spec, state_spec=(), name=name
#         )
#
#         self._action_spec = action_spec
#         flat_action_spec = tf.nest.flatten(action_spec)
#         window_length = input.get_shape()[2]
#         self._encoding_network = ActorForward(
#             input_tensor_spec,
#             s_dim,
#             state_spec,
#             a_dim,
#             predictor_type,
#             use_batch_norm,
#             name
#         )
#         self._cnn_encoder = encoding_network.EncodingNetwork()
#         #     input_tensor_spec=observation_spec,
#         #     preprocessing_layers=None,
#         #     preprocessing_combiner=None,
#         #     conv_layer_params=((32, (5, 5), (1, 3)), (32, (5, 5), (1, window_length - 2))),
#         #     ## filters, kernel_size, stride
#         #     # filters = 32, filtersize = 1,3 kernel size (5,5)
#         #     # strides = 1
#         #     fc_layer_params=(64, 64),
#         #     dropout_layer_params=None,
#         #     activation_fn=tflearn.activations.relu,
#         #     weight_decay_params=None,
#         #     kernel_initializer=None,
#         #     batch_squash=False,
#         #     Name="CNNActor",
#         #     conv_type='2d'
#         # )
#         # self._lstm_encoder = lstm_encoding_network(
#         # #         input_tensor_spec, preprocessing_layers=None, preprocessing_combiner=None,
#         #     #     conv_layer_params=None, input_fc_layer_params=(75, 40), lstm_size=None,
#         #     #     output_fc_layer_params=(75, 40), activation_fn=tf.keras.activations.relu,
#         #     #     rnn_construction_fn=None, rnn_construction_kwargs=None, dtype=tf.float32,
#         #     #     name='LSTMEncodingNetwork'
#         #     input_tensor_spec=observation_spec,
#         #     preprocessing_layers=None,
#         #     preprocessing_combiner=None,
#         #     conv_layer_params= None,
#         #
#         # )
#
