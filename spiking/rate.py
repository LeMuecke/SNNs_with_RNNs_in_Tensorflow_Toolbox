import tensorflow as tf
import numpy as np
from typing import Tuple, Callable

from spiking.helpers import spike_function

# based on D. Auge, J. Hille, R. Dietrich, and E. Mueller. “Tutorial: Spiking Network Simulation inTensorFlow”. unpublished. 2020.

class IntegratorNeuronCell(tf.keras.layers.Layer):
    """
    A simple spiking neuron layer that integrates (sums up) the outputs of the previous layer.
    """

    def __init__(self, n_in, n_neurons, **kwargs):
        """
        Initialization function of the IntegratorNeuronCell.

        @param n_in: Number of inputs, i.e. outputs of previous layer.
        @param n_neurons: Number of neurons, i.e. outputs of this layer.
        @param kwargs: Additional parameters, forwarded to standard Layer init function of tf.
        """
        super(IntegratorNeuronCell, self).__init__(**kwargs)
        self.n_in = n_in
        self.n_neurons = n_neurons

        self.w_in = None
        self.b_in = None

    def build(self, input_shape):
        """
        Creates the variables of this layer, i.e. creates and initializes the weights
        for all neurons within this layer.

        @param input_shape: Not needed for this layer.
        @type input_shape:
        """
        del input_shape  # Unused

        w_in = tf.random.normal((self.n_in, self.n_neurons), dtype=self.dtype)
        self.w_in = tf.Variable(initial_value=w_in / np.sqrt(self.n_in), trainable=True)

        b_in = tf.random.normal((self.n_neurons,), dtype=self.dtype)
        # TODO why "/ np.sqrt"? Should not matter as we don't train
        self.b_in = tf.Variable(initial_value=b_in / np.sqrt(self.n_in), trainable=True)

    @property
    def state_size(self) -> Tuple[int, int]:
        """
        Returns the state size depicted of cell and hidden state  as a tuple of number of neurons, number of neurons.
        @return:
        """
        return self.n_neurons, self.n_neurons

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """

        @param inputs:
        @param batch_size:
        @param dtype:
        @return:
        """
        del inputs  # Unused

        zeros = tf.zeros((batch_size, self.n_neurons), dtype=dtype)
        return zeros, zeros

    def call(self, input_at_t, states_at_t):
        """

        @param input_at_t:
        @param states_at_t:
        @return:
        """
        old_v, old_z = states_at_t

        i_t = tf.add(tf.matmul(input_at_t, self.w_in), self.b_in)

        new_v = old_v + i_t
        new_z = tf.nn.softmax(new_v)

        return new_z, (new_v, new_z)


class LifNeuronCell(IntegratorNeuronCell):
    """
    A more advanced spiking tf layer building upon the IntegratorNeuronCell,
    but augmenting it with a leaky and fire functionality.
    """

    def __init__(self, n_in: int, n_neurons: int, tau: float = 20., threshold: float = 0.1,
                 activation_function: Callable[[tf.Tensor], tuple] = spike_function, **kwargs):
        """
        Initializes a (Recurrent)LifNeuronCell.
        @param n_in: Number of inputs, i.e. outputs of previous layer.
        @param n_neurons: Number of neurons, i.e. outputs of this layer.
        @param tau: The time constant tau.
        @param threshold: The threshold for the neurons in this layer.
        @param activation_function: The activation function for the LIF-Neuron, defaults to a simple spike-function.
        @param kwargs: Additional parameters, forwarded to standard Layer init function of tf.
        """
        super(LifNeuronCell, self).__init__(n_in, n_neurons, **kwargs)
        self.tau = tau
        self.decay = tf.exp(-1 / tau)
        self.threshold = threshold

        self.activation_function = activation_function

    def call(self, input_at_t, states_at_t):
        old_v, old_z = states_at_t

        i_t = tf.add(tf.matmul(input_at_t, self.w_in), self.b_in)
        i_reset = old_z * self.threshold

        new_v = self.decay * old_v + (1.0 - self.decay) * i_t - i_reset
        new_z = self.activation_function(new_v / self.threshold)

        return new_z, (new_v, new_z)


# TODO LifNeuronConv2DCell broken
class LifNeuronCellConv2D(IntegratorNeuronCell):
    """
    A more advanced spiking tf layer building upon the IntegratorNeuronCell,
    but augmenting it with a leaky and fire functionality.
    """

    def __init__(self, input_shape: Tuple, output_shape: Tuple, tau: float = 20., threshold: float = 0.1,
                 kernel_size: (int, int) = (3, 3), filters: int = 3, strides: (int, int) = (1, 1),
                 padding: str = "VALID", data_format: str = "NHWC", dilations: int or list = 1,
                 activation_function: Callable[[tf.Tensor], tuple] = spike_function, **kwargs):
        """
        Initializes a (Recurrent)LifNeuronCell.
        @param n_in: Number of inputs, i.e. outputs of previous layer.
        @param n_neurons: Number of neurons, i.e. outputs of this layer.
        @param tau: The time constant tau.
        @param threshold: The threshold for the neurons in this layer.
        @param activation_function: The activation function for the LIF-Neuron, defaults to a simple spike-function.
        @param kwargs: Additional parameters, forwarded to standard Layer init function of tf.
        """
        input_shape_np = np.array(input_shape)
        output_shape_np = np.array(output_shape)
        n_in = np.prod(input_shape_np[input_shape_np != np.array(None)])
        n_neurons = np.prod(output_shape_np[output_shape_np != np.array(None)])
        super(LifNeuronCellConv2D, self).__init__(n_in, n_neurons, **kwargs)
        self.conv_input_shape = input_shape_np
        self.conv_output_shape = output_shape_np
        self.tau = tau
        self.decay = tf.exp(-1 / tau)
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.data_format = "NHWC" if data_format == "channels_last" else "NCHW"
        self.dilations = dilations

        self.activation_function = activation_function

    def build(self, input_shape):
        """
        Creates the variables of this layer, i.e. creates and initializes the weights
        for all neurons within this layer.

        @param input_shape: Not needed for this layer.
        @type input_shape:
        """
        del input_shape  # Unused

        # TODO 1 in shape has to be replace!
        w_in = tf.random.normal((self.kernel_size[0], self.kernel_size[1], self.conv_input_shape[3], self.filters), dtype=self.dtype)
        self.w_in = tf.Variable(initial_value=w_in / np.sqrt(self.n_in), trainable=True)

        b_in = tf.random.normal((self.filters,), dtype=self.dtype)
        self.b_in = tf.Variable(initial_value=b_in / np.sqrt(self.n_in), trainable=True)

    @property
    def state_size(self) -> Tuple[np.array, np.array]:
        """
        Returns the state size depicted of cell and hidden state  as a tuple of number of neurons, number of neurons.
        @return:
        """
        return self.conv_output_shape, self.conv_output_shape

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """

        @param inputs:
        @param batch_size:
        @param dtype:
        @return:
        """
        del inputs  # Unused
        conv_output_shape_without_none = self.conv_output_shape[self.conv_output_shape != np.array(None)]
        zeros = tf.zeros(tuple(np.append(np.array((batch_size,)), conv_output_shape_without_none)), dtype=dtype)
        return zeros, zeros

    def call(self, input_at_t, states_at_t):
        old_v, old_z = states_at_t

        i_t = tf.nn.conv2d(input_at_t, filters=self.w_in, strides=self.strides, padding=self.padding, data_format=self.data_format, dilations=self.dilations)
        i_t = tf.nn.bias_add(i_t, self.b_in)

        # i_t = tf.add(tf.matmul(input_at_t, self.w_in), self.b_in)
        i_reset = old_z * self.threshold

        new_v = self.decay * old_v + (1.0 - self.decay) * i_t - i_reset
        new_z = self.activation_function(new_v / self.threshold)

        return new_z, (new_v, new_z)


class RecurrentLifNeuronCell(LifNeuronCell):
    """
    A recurrent spiking layer implementing a recurrent layer of LIF-Neurons.
    Each neuron has a connection to the previous/next layer as well recurrent
    connection to itself.
    """

    def build(self, input_shape):
        del input_shape  # Unused

        w_in = tf.random.normal((self.n_in, self.n_neurons), dtype=self.dtype)
        self.w_in = tf.Variable(initial_value=w_in / np.sqrt(self.n_in), trainable=True)

        w_rec = tf.random.normal((self.n_neurons, self.n_neurons), dtype=self.dtype)
        w_rec = tf.linalg.set_diag(w_rec, np.zeros(self.n_neurons))
        self.w_rec = tf.Variable(initial_value=w_rec / np.sqrt(self.n_neurons), trainable=True)

    def call(self, input_at_t, states_at_t):
        old_v, old_z = states_at_t

        i_t = tf.matmul(input_at_t, self.w_in) + tf.matmul(old_z, self.w_rec)
        i_reset = old_z * self.threshold

        new_v = self.decay * old_v + (1.0 - self.decay) * i_t - i_reset
        new_z = self.activation_function(new_v / self.threshold)

        return new_z, (new_v, new_z)
