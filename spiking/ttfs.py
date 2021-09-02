import tensorflow as tf
import numpy as np
from typing import Tuple, Callable

from spiking.helpers import spike_function


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
    def state_size(self) -> Tuple[int, int, int]:
        """
        Returns the state size depicted of cell and hidden state  as a tuple of number of neurons, number of neurons.
        @return:
        """
        return self.n_neurons, self.n_neurons, 1

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """

        @param inputs:
        @param batch_size:
        @param dtype:
        @return:
        """
        del inputs  # Unused

        zeros = tf.zeros((batch_size, self.n_neurons), dtype=dtype)
        return zeros, zeros, 0.

    def call(self, input_at_t, states_at_t):
        """

        @param input_at_t:
        @param states_at_t:
        @return:
        """
        old_v, old_z, t = states_at_t

        # TODO this is never used, everything but dynthresh uses rate integrator, has to be changed for clarity!
        u_t = tf.add(tf.matmul(tf.subtract(t, input_at_t), self.w_in), tf.multiply(self.b_in, t))

        new_v = old_v + u_t
        new_z = tf.nn.softmax(new_v)

        return new_z, (new_v, new_z, t + 1)


class LifNeuronCell(IntegratorNeuronCell):
    """
    A LifNeuron that uses time-to-first-spike (ttfs) encoding.
    Be aware that this implementation does not resemble the way ttfs normally works.
    Normally, ttfs would output one spike only, in this implementation, ttfs outputs spikes at every timestep after
    it has spiked for the first time.
    The reason for that is the way the neuron potential for ttfs in the Rueackauer 2018 paper is defined. They sum over
    all timesteps and multiply the weight with the time since the first spike.
    Here, for every timestep, the weight is added to the potential of the previous timestep, hence the continuous
    spiking after the first spike.
    """

    def __init__(self, n_in: int, n_neurons: int, tau: float = 999999., threshold: float = 0.1,
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
        self.threshold = threshold
        self.alpha = tf.exp(-1 / tau)

        self.activation_function = activation_function

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
        old_v, old_z = states_at_t

        # membrane potential that will be added this timestep
        # input * weights + bias
        i_t = tf.add(tf.matmul(input_at_t, self.w_in), self.b_in)

        # add new potential of this timestep to the old potential
        new_v = self.alpha * old_v + i_t
        # new_z -> output_at_t, tf.maximum makes output stay at 1 when the first spike was emitted
        # this is due to the way the Rueckauer TTFS math is translated to TensorFlow (TODO more to come)
        new_z = tf.maximum(self.activation_function(new_v / self.threshold), old_z)

        return new_z, (new_v, new_z)


class LifNeuronCellConv2D(IntegratorNeuronCell):
    """
    A LifNeuron that uses time-to-first-spike (ttfs) encoding.
    Be aware that this implementation does not resemble the way ttfs normally works.
    Normally, ttfs would output one spike only, in this implementation, ttfs outputs spikes at every timestep after
    it has spiked for the first time.
    The reason for that is the way the neuron potential for ttfs in the Rueackauer 2018 paper is defined. They sum over
    all timesteps and multiply the weight with the time since the first spike.
    Here, for every timestep, the weight is added to the potential of the previous timestep, hence the continuous
    spiking after the first spike.
    This class implements ttfs with convolutions.
    """

    def __init__(self, input_shape: int, output_shape: int, tau: float = 999999.,  threshold: float = 0.1,
                 activation_function: Callable[[tf.Tensor], tuple] = spike_function,
                 kernel_size: (int, int) = (3, 3), filters: int = 3, strides: (int, int) = (1, 1),
                 padding: str = "VALID", data_format: str = "NHWC", dilations: int or list = 1, **kwargs):
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
        self.threshold = threshold
        self.alpha = tf.exp(-1 / tau)

        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = padding.upper()
        self.activation_function = activation_function
        self.data_format = "NHWC" if data_format == "channels_last" else "NCHW"
        self.dilations = dilations

    def build(self, input_shape):
        """
        Creates the variables of this layer, i.e. creates and initializes the weights
        for all neurons within this layer.

        @param input_shape: Not needed for this layer.
        @type input_shape:
        """
        del input_shape  # Unused

        # kernel, kernel, in_depth, out_depth
        w_in = tf.random.normal((self.kernel_size[0],
                                 self.kernel_size[1],
                                 self.conv_input_shape[3],
                                 self.filters), dtype=self.dtype)
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

        # add new potential of this timestep to the old potential
        new_v = self.alpha * old_v + i_t
        # new_z -> output_at_t, tf.maximum makes output stay at 1 when the first spike was emitted
        # this is due to the way the Rueckauer TTFS math is translated to TensorFlow (TODO more to come)
        new_z = tf.maximum(self.activation_function(new_v / self.threshold), old_z)

        return new_z, (new_v, new_z)


class LifNeuronCellMaxPool2D(IntegratorNeuronCell):
    # TODO this can probably also be achieved with a regular maxpool2d layer
    """
    A LifNeuron that uses time-to-first-spike (ttfs) encoding.
    Be aware that this implementation does not resemble the way ttfs normally works.
    Normally, ttfs would output one spike only, in this implementation, ttfs outputs spikes at every timestep after
    it has spiked for the first time.
    The reason for that is the way the neuron potential for ttfs in the Rueackauer 2018 paper is defined. They sum over
    all timesteps and multiply the weight with the time since the first spike.
    Here, for every timestep, the weight is added to the potential of the previous timestep, hence the continuous
    spiking after the first spike.
    This implements ttfs with max pooling.
    """

    def __init__(self, n_in: int, n_neurons: int, threshold: float = 0.1,
                 activation_function: Callable[[tf.Tensor], tuple] = spike_function,
                 pool_size: Tuple = (2, 2), strides: int = None, padding: str = "valid",
                 data_format: str = "channels_last", **kwargs):
        """
        Initializes a (Recurrent)LifNeuronCell.
        @param n_in: Number of inputs, i.e. outputs of previous layer.
        @param n_neurons: Number of neurons, i.e. outputs of this layer.
        @param tau: The time constant tau.
        @param threshold: The threshold for the neurons in this layer.
        @param activation_function: The activation function for the LIF-Neuron, defaults to a simple spike-function.
        @param kwargs: Additional parameters, forwarded to standard Layer init function of tf.
        """
        super(LifNeuronCellMaxPool2D, self).__init__(n_in, n_neurons, **kwargs)
        self.threshold = threshold
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.activation_function = activation_function
        if data_format not in ["channels_last", "channels_first"]:
            print("WARNING: data_format has to be channels_last for NHWC or channels_first for NCHW. "
                  + str(data_format) + " is not known, so the default channels_last/NHWC will be used.")
        self.data_format = "NHWC" if data_format == "channels_last" else "NCHW"

    @property
    def state_size(self) -> int:
        """
        Returns the state size depicted of cell and hidden state  as a tuple of number of neurons, number of neurons.
        @return:
        """
        return 1

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """

        @param inputs:
        @param batch_size:
        @param dtype:
        @return:
        """
        del inputs  # Unused

        return 0

    def call(self, input_at_t, states_at_t):
        empty = states_at_t

        pooled_input = tf.nn.max_pool2d(input_at_t,
                                        ksize=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding.upper(),
                                        data_format=self.data_format)

        return pooled_input, (empty,)
