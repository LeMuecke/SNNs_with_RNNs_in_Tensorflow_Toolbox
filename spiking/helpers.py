import tensorflow as tf


@tf.custom_gradient
def spike_function(v_to_threshold: tf.Tensor) -> tuple:
    """
    A custom gradient for networks of spiking neurons.

    @param v_to_threshold: The difference between current and threshold voltage of the neuron.
    @type v_to_threshold: tf.float32
    @return: Activation z and gradient grad.
    @rtype: tuple
    """
    z = tf.cast(tf.greater(v_to_threshold, 1.), dtype=tf.float32)

    def grad(dy: tf.Tensor) -> tf.Tensor:
        """
        The gradient function for calculating the derivative of the spike-function.

        The return value is determined as follows:

        # @negative: v_to_threshold < 0 -> dy*0
        # @rest: v_to_threshold = 0 -> dy*0+
        # @thresh: v_to_threshold = 1 -> dy*1
        # @+thresh: v_to_threshold > 1 -> dy*1-
        # @2thresh: v_to_threshold > 2 -> dy*0
        #
        #         /\
        #        /  \
        # ______/    \______
        # -1   0   1  2   3  v_to_threshold

        @param dy: The previous upstream gradient.
        @return: The calculated gradient of this stage of the network
        """
        return [dy * tf.maximum(1 - tf.abs(v_to_threshold - 1), 0)]

    return z, grad