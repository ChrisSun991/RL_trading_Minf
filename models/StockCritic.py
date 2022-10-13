import keras
import tensorflow as tf
import tflearn

from models.StockPredictor import stock_predictor


# DDPG

class DDPGCritic(keras.Model):  # /Actor):
    def __init__(self, a_dim, predictor_type, use_batch_norm):
        # self.learning_rate = learning_rate
        # self.tau = tau
        # self.batch_size = batch_size
        super(DDPGCritic, self).__init__()
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.a_dim = a_dim
        self.bn = tflearn.normalization.batch_normalization
        self.relu_act = tflearn.activations.relu
        # self.s_dim = s_dim
        # super(Actor, self).__init__(input_tensor_spec, output_tensor_spec)

        # self.fully_connected_net = tflearn.fully_connected(n_units=64)
        # self.batch_norm_lay = tflearn.normalization.batch_normalization()

    def call(self, s_dim):
        """
        DDPG_stock actor:
			inputs = tf.input_data (shape=[None] + self.s_dim +[1])
			a = tf.input_data(shape=[None] + self.a_dim)
			x = stock_predictor(x)
			t1 = fully_connected(x, 64)
			t2 = fully_connected(a,64)
			x = tf.add(t1,t2)
			“bn”
			relu(x)
			Weights = tf.init.uniform(minval=-0.003, maxval=0.003)
			out = fully_connected(x,1,weights)
        """
        input = tflearn.input_data(shape=[None] + s_dim + [1])
        actions = tflearn.input_data(shape=[None] + s_dim)

        x = stock_predictor(input, self.predictor_type, self.use_batch_norm)

        t1 = tflearn.fully_connected(x, 64)
        t2 = tflearn.fully_connected(actions, 64)

        x = tf.add(t1, t2)

        if self.use_batch_norm:
            x = self.bn(x)

        x = self.relu_act(x)

        w_initializations = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        output = tflearn.fully_connected(x, 1, weights_init=w_initializations)
        return input, actions, output


# TD3
class TD3Critic(keras.Model):  # /Actor):
    def __init__(self, a_dim, predictor_type, use_batch_norm):
        # self.learning_rate = learning_rate
        # self.tau = tau
        # self.batch_size = batch_size
        super(TD3Critic, self).__init__()
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.a_dim = a_dim
        self.bn = tflearn.normalization.batch_normalization
        self.relu_act = tflearn.activations.relu
        # self.s_dim = s_dim
        # super(Actor, self).__init__(input_tensor_spec, output_tensor_spec)

        # self.fully_connected_net = tflearn.fully_connected(n_units=64)
        # self.batch_norm_lay = tflearn.normalization.batch_normalization()

    def call(self, s_dim):
        """
				net1 = stock_predictor(x)
				t1 = fully_connected(net1,64)
				t2 = fully_connected(a, 64)
				net1 = tf.add(t1,t2)
				“bn”
				relu(net1)
				Weights = tf.init.uniform(minval=-0.003, maxval=0.003)
				out1 = fully_connected(net1,1,weights)

				net2 = tf.add(t1,t2)
				“bn”
				relu(net2)
				Weights = tf.init.uniform(minval=-0.003, maxval=0.003)
				out2 = fully_connected(net2,1,weights)

				return out1,out2
        """

        input = tflearn.input_data(shape=[None] + s_dim + [1])
        actions = tflearn.input_data(shape=[None] + s_dim)

        x = stock_predictor(input, self.predictor_type, self.use_batch_norm)

        t1 = tflearn.fully_connected(x, 64)
        t2 = tflearn.fully_connected(actions, 64)

        # TD3 first net
        network1 = tf.add(t1, t2)

        if self.use_batch_norm:
            network1 = self.bn(network1)

        network1 = self.relu_act(network1)

        w_initializations = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        output1 = tflearn.fully_connected(network1, 1, weights_init=w_initializations)

        # TD3 second net
        network2 = tf.add(t1, t2)
        if self.use_batch_norm:
            network2 = self.bn(network2)

        network2 = self.relu_act(network2)
        output2 = tflearn.fully_connected(network2, 1, weights_init=w_initializations)

        return output1, output2
