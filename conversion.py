import itertools
import concurrent.futures

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend

from spiking import helpers, rate, ttfs, ttfs_dynthresh
import time


class PercentileEstimator:
    # TODO docs
    def __init__(self, percentile: float, max_buffer_size: int, dirty_ram_mode=False):
        self.percentile = percentile
        self.max_buffer_size = max_buffer_size
        self.buffer = np.empty(0, dtype=np.float32)
        self.n_lower_than_thresh = 0
        self.issued_buffer_size_warning = False
        self.dirty_ram_mode = dirty_ram_mode
        if self.dirty_ram_mode:
            pass
        else:
            pass

    # when a batch was calculated, the activations will be given to the class via this method
    def add_batch(self, batch: np.array):
        if self.dirty_ram_mode:
            self.buffer = np.append(self.buffer, batch)
        else:
            # TODO check if batch is flat
            if len(self.buffer) < self.n_lower_than_thresh * (1 - self.percentile / 100) and not self.issued_buffer_size_warning:
                print("WARNING: Buffer is too small to correctly calculate the percentile!")
                self.issued_buffer_size_warning = True

            self.buffer = np.append(self.buffer, batch)
            self.buffer.sort(kind="stable")
            self.n_lower_than_thresh += max(0, len(self.buffer) - self.max_buffer_size)
            self.buffer = self.buffer[-self.max_buffer_size:]
            #print_spent_time(timestep, "sorting stuff.")

    def get_percentile_value(self):
        if self.dirty_ram_mode:
            #self.buffer.sort(kind="stable")
            timestep = int(time.time())
            perc = np.percentile(self.buffer, self.percentile)
            print_spent_time(timestep, "percentiles.")
            return perc
        else:
            theoretical_array_size = self.n_lower_than_thresh + len(self.buffer)
            percentile_element_index = round(theoretical_array_size * (self.percentile / 100)) - self.n_lower_than_thresh
            if percentile_element_index < 0:
                print("ERROR: The buffer used for calculating percentiles has become too small to hold the necessary "
                      "portion of the batches. Consider making the buffer bigger, otherwise normalization will not be"
                      "as expected. The buffer needs to be at least " + str(abs(percentile_element_index)) + " bigger.")
                percentile_element_index = 0
            return self.buffer[percentile_element_index]

    def reset(self):
        self.buffer = np.empty(0)
        self.n_lower_than_thresh = 0


class NormalizationSNN(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs


def print_spent_time(time_breakpoint, part_name):
    time_diff = time.time() - time_breakpoint
    print("%%% Time spent with " + part_name + ": " + str(time_diff) + "s")
    return time.time()


def spikalize_img(experiment, image, label):
    """
    Transform image to spikes. Spike with poisson distributed rate proportional to pixel brightness.
    :param experiment:
    :param image:
    :param label:
    :return:
    """

    image_shape = np.append(np.array(experiment.timesteps), np.array(image.shape))
    rand = tf.random.uniform(shape=image_shape)
    spiked_img = tf.cast(image / 255 * experiment.max_rate > rand, tf.float32)

    return spiked_img, label


def timealize_img(experiment, image, label):
    """
    Transforms an image to an image with timesteps to be fed into a SNN.
    Copies the image timesteps times and stacks them together.
    Equivalent of feeding the SNN a raw input current, not feeding it spikes.
    :param experiment:
    :param image:
    :param label:
    :return:
    """
    img_with_time = tf.stack([image] * experiment.timesteps)

    return img_with_time, label


def get_previous_layer_regarding_normalization(model, layer):
    """
    Helper method, returns the previous layer but jumps over BatchNormalization layers, as they should not be
    used to calculate weight normalization
    :return:
    """

    layer_index = model.layers.index(layer)
    for i in range(layer_index - 1, -1, -1):
        if isinstance(model.layers[i], tf.keras.layers.BatchNormalization):
            # if layer is batch norm do nothing and look for the layer before the current one
            pass
        else:
            # if none of the exemptions above are true, return the current element
            return model.layers[i]
    return None


def convert_ann_to_snn(experiment, model, ds_normalize):
    """
    Conversion of ANN to SNN.
    Creates, based on model (ANN), the corresponding SNN in the first part.
    In the second part, the weights are copied from the ANN model to the SNN model.
    With ds_normalize the activations are normalized and translated to SNN weights.
    :param experiment:
    :param model:
    :param ds_normalize:
    :return:
    """
    layer_mapping = dict()
    layer_list = []

    # calculate input shape
    input_shape = np.array(model.input.shape)
    input_shape = input_shape[input_shape != np.array(None)]
    input_shape = np.append(np.array(experiment.timesteps), input_shape)

    # first layer(s)
    if experiment.encoding == "ttfs_dyn_thresh":
        input_layer = (tf.keras.layers.Input(shape=input_shape), tf.keras.layers.Input(shape=input_shape))
    else:
        input_layer = tf.keras.layers.Input(shape=input_shape)

    for i, elem in enumerate(model.layers):
        if isinstance(elem, tf.keras.layers.Flatten):
            target_shape = (experiment.timesteps,) + tuple([x for x in elem.output_shape if x is not None])
            new_snn_layer = tf.keras.layers.Reshape(target_shape=target_shape)
            layer_list.append(new_snn_layer)
            layer_mapping[elem] = new_snn_layer
        elif isinstance(elem, tf.keras.layers.InputLayer):
            # input is added anyways, nothing to do
            pass
        elif isinstance(elem, tf.keras.layers.Dropout):
            pass
        elif isinstance(elem, tf.keras.layers.Dense):
            input_size = elem.input_shape[1]
            output_size = elem.output_shape[1]
            if i + 1 == len(model.layers):
                # last layer is different
                if experiment.encoding == "ttfs_dyn_thresh":
                    new_snn_layer = tf.keras.layers.RNN(ttfs_dynthresh.IntegratorNeuronCell(n_in=input_size,
                                                                                            n_neurons=output_size),
                                                        return_sequences=False)
                else:
                    new_snn_layer = tf.keras.layers.RNN(rate.IntegratorNeuronCell(n_in=input_size,
                                                                                  n_neurons=output_size),
                                                        return_sequences=False)
                layer_list.append(new_snn_layer)
                layer_mapping[elem] = new_snn_layer
            else:
                if experiment.encoding == "rate":
                    new_snn_layer = tf.keras.layers.RNN(rate.LifNeuronCell(n_in=input_size,
                                                                           n_neurons=output_size,
                                                                           tau=experiment.tau,
                                                                           threshold=experiment.threshold),
                                                        return_sequences=True)
                    layer_list.append(new_snn_layer)
                    layer_mapping[elem] = new_snn_layer
                elif experiment.encoding == "ttfs" or experiment.encoding == "ttfs_clamped":
                    new_snn_layer = tf.keras.layers.RNN(ttfs.LifNeuronCell(n_in=input_size,
                                                                           n_neurons=output_size,
                                                                           tau=experiment.tau,
                                                                           threshold=experiment.threshold),
                                                        return_sequences=True)
                    layer_list.append(new_snn_layer)
                    layer_mapping[elem] = new_snn_layer
                elif experiment.encoding == "ttfs_dyn_thresh":
                    new_snn_layer = tf.keras.layers.RNN(ttfs_dynthresh.LifNeuronCell(n_in=input_size,
                                                                                     n_neurons=output_size,
                                                                                     tau=experiment.tau,
                                                                                     threshold=experiment.threshold),
                                                        return_sequences=True)
                    layer_list.append(new_snn_layer)
                    layer_mapping[elem] = new_snn_layer

                else:
                    raise NotImplementedError("Encoding " + str(experiment.encoding) + " not implemented.")
        elif isinstance(elem, tf.keras.layers.BatchNormalization):
            pass
            # do nothing, batch normalization will be incorporated in the weight conversion and does not
            # need a dedicated layer
        elif isinstance(elem, tf.keras.layers.MaxPool2D):
            if experiment.encoding == "rate":
                raise NotImplementedError("Encoding " + str(experiment.encoding) + " of layer MaxPool2D not implemented.")
            elif experiment.encoding == "ttfs" or experiment.encoding == "ttfs_clamped":
                new_snn_layer = tf.keras.layers.RNN(ttfs.LifNeuronCellMaxPool2D(n_in=1,
                                                                                n_neurons=1,
                                                                                threshold=experiment.threshold,
                                                                                pool_size=elem.pool_size,
                                                                                strides=elem.strides,
                                                                                padding=elem.padding,
                                                                                data_format=elem.data_format),
                                                    return_sequences=True)
                layer_list.append(new_snn_layer)
                layer_mapping[elem] = new_snn_layer
            elif experiment.encoding == "ttfs_dyn_thresh":
                new_snn_layer = tf.keras.layers.RNN(ttfs_dynthresh.LifNeuronCellMaxPool2D(n_in=1,
                                                                                n_neurons=1,
                                                                                threshold=experiment.threshold,
                                                                                pool_size=elem.pool_size,
                                                                                strides=elem.strides,
                                                                                padding=elem.padding,
                                                                                data_format=elem.data_format),
                                                    return_sequences=True)
                layer_list.append(new_snn_layer)
                layer_mapping[elem] = new_snn_layer
            else:
                raise NotImplementedError("Encoding " + str(experiment.encoding) + " of layer MaxPool2D not implemented.")
            # do nothing, batch normalization will be incorporated in the weight conversion and does not
            # need a dedicated layer
        elif isinstance(elem, tf.keras.layers.Conv2D):
            if experiment.encoding == "rate":
                new_snn_layer = tf.keras.layers.RNN(rate.LifNeuronCellConv2D(input_shape=elem.input_shape,
                                                                             output_shape=elem.output_shape,
                                                                             tau=experiment.tau,
                                                                             threshold=experiment.threshold,
                                                                             kernel_size=elem.kernel_size,
                                                                             filters=elem.filters,
                                                                             strides=elem.strides,
                                                                             padding=elem.padding.upper(),
                                                                             data_format=elem.data_format,
                                                                             dilations=elem.dilation_rate),
                                                    return_sequences=True)
                layer_list.append(new_snn_layer)
                layer_mapping[elem] = new_snn_layer
            elif experiment.encoding == "ttfs" or experiment.encoding == "ttfs_clamped":
                new_snn_layer = tf.keras.layers.RNN(ttfs.LifNeuronCellConv2D(input_shape=elem.input_shape,
                                                                             output_shape=elem.output_shape,
                                                                             tau=experiment.tau,
                                                                             threshold=experiment.threshold,
                                                                             kernel_size=elem.kernel_size,
                                                                             filters=elem.filters,
                                                                             strides=elem.strides,
                                                                             padding=elem.padding.upper(),
                                                                             data_format=elem.data_format,
                                                                             dilations=elem.dilation_rate),
                                                    return_sequences=True)
                layer_list.append(new_snn_layer)
                layer_mapping[elem] = new_snn_layer
            elif experiment.encoding == "ttfs_dyn_thresh":
                new_snn_layer = tf.keras.layers.RNN(ttfs_dynthresh.LifNeuronCellConv2D(input_shape=elem.input_shape,
                                                                                       output_shape=elem.output_shape,
                                                                                       tau=experiment.tau,
                                                                                       threshold=experiment.threshold,
                                                                                       kernel_size=elem.kernel_size,
                                                                                       filters=elem.filters,
                                                                                       strides=elem.strides,
                                                                                       padding=elem.padding.upper(),
                                                                                       data_format=elem.data_format,
                                                                                       dilations=elem.dilation_rate),
                                                    return_sequences=True)
                layer_list.append(new_snn_layer)
                layer_mapping[elem] = new_snn_layer
            else:
                raise NotImplementedError("Encoding " + str(experiment.encoding) + " of layer Conv2D not implemented.")
        else:
            raise NotImplementedError("Layer of type " + str(type(elem)) + " not implemented yet.")

    # connect layers
    x = input_layer
    for elem in layer_list:
        if isinstance(elem, tf.keras.layers.Reshape) and experiment.encoding == "ttfs_dyn_thresh":
            x1, x2 = x
            elem2 = tf.keras.layers.Reshape(target_shape=elem.target_shape)
            x = elem(x1), elem2(x2)
        else:
            x = elem(x)
    output_layer = x

    model_snn = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model_feature_extractor = tf.keras.Model(inputs=model.inputs,
                                             outputs=[layer.output for layer in model.layers])

    # ############ NORMALIZE WEIGHTS FOR SNN #############

    def loop_over_dataset(func):
        print("Starting Thread!")
        i, func = func
        percentile_estimator = PercentileEstimator(experiment.norm_percentile, int(1e8), dirty_ram_mode=True)
        for elem in itertools.islice(ds_normalize.as_numpy_iterator(), 800):
            percentile_estimator.add_batch(func(elem[0])[0].flatten())
        print("Finished iteration!")
        return i, np.array(percentile_estimator.get_percentile_value())

    #percentile_estimator = PercentileEstimator(experiment.norm_percentile, int(2e8))

    #with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #    out = executor.map(loop_over_dataset, enumerate(functors))

    activations_complete_list = list()
    # full dataset at range(15) and .take(64) for mnist
    for i in range(30):
        print(f"i{i}")
        activations_complete_list.append([elem.flatten() for elem in model_feature_extractor.predict(ds_normalize.take(int(64 / 16)))])

    all_activations = list()
    for i in range(len(activations_complete_list[0])):
        all_activations.append(np.concatenate([elem[i] for elem in activations_complete_list]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        max_by_layer = executor.map(lambda x: np.percentile(x, experiment.norm_percentile), all_activations)

    max_by_layer = np.array(list(max_by_layer))

    del all_activations
    del activations_complete_list

    # assign layers to their max calculated before
    layer_max_mapping = dict()
    for layer, max_val in zip(model.layers, max_by_layer):
        layer_max_mapping[layer] = max_val

    # convert weights and biases and set them on the SNN
    for i, elem in enumerate(model.layers):
        if isinstance(elem, tf.keras.layers.Flatten):
            pass
        elif isinstance(elem, tf.keras.layers.InputLayer):
            pass
        elif isinstance(elem, tf.keras.layers.Dropout):
            pass
        elif isinstance(elem, tf.keras.layers.BatchNormalization):
            pass
            # do nothing, batch norm was already incorporated in the previous layer
        elif isinstance(elem, tf.keras.layers.Dense) or isinstance(elem, tf.keras.layers.Conv2D):
            weights, biases = elem.get_weights()
            # according to Rueckauer 2016

            max_of_this_layer = layer_max_mapping[elem]
            prev_layer = get_previous_layer_regarding_normalization(model, elem)
            if prev_layer is not None:
                max_of_layer_minus_one = layer_max_mapping[get_previous_layer_regarding_normalization(model, elem)]
            else:
                # this happens if there is no explicit input or flatten layer
                max_of_layer_minus_one = 0
                for ds_elem in ds_normalize.as_numpy_iterator():
                    max_of_layer_minus_one = np.maximum(max_of_layer_minus_one, np.max(ds_elem[0]))
            weights = weights * (max_of_layer_minus_one / max_of_this_layer)
            biases = biases / max_of_this_layer

            # if the next layer is a batch norm layer, the current layer will be transformed with the batch norm weights
            try:
                elem_plus_one = model.layers[i + 1]
            except IndexError:
                # elem is the last layer, so there is no next layer
                elem_plus_one = None
            if isinstance(elem_plus_one, tf.keras.layers.BatchNormalization):
                gamma, beta, moving_mean, moving_variance = model.layers[i + 1].get_weights()
                diag_matrix_factor = np.diag(gamma / moving_variance)
                weights = np.matmul(weights, diag_matrix_factor)
                biases = (gamma / moving_variance) * (biases - moving_mean) + beta

            snn_layer_index = model_snn.layers.index(layer_mapping[elem])
            model_snn.layers[snn_layer_index].set_weights([weights, biases])
        elif isinstance(elem, tf.keras.layers.MaxPool2D):
            # nothing to be converted
            pass
        else:
            raise NotImplementedError("ANN -> SNN conversion of " + str(type(elem)) + " is not implemented.")

    return model_snn


def ann_accuracy(model, ds_train, ds_test, experiment=None, max_images=None, disable_training_accuracy=False):
    """
    Test the accuracy of an ANN network with train and dev.
    If you supply a experiment, the values will be logged to the experiment
    :param model:
    :param ds_train:
    :param ds_test:
    :param experiment:
    :param max_images:
    :return:
    """
    print("####### ANN ########")
    if max_images is not None:
        ds_train = ds_train.take(max_images)
        ds_test = ds_test.take(max_images)

    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}', histogram_freq=1)
    if not disable_training_accuracy:
        eval_train = model.evaluate(ds_train)
    eval_test = model.evaluate(ds_test)

    if experiment is not None:
        try:
            if not disable_training_accuracy:
                experiment.ann_accuracy_train = dict((zip(model.metrics_names, eval_train)))["sparse_categorical_accuracy"]
            else:
                experiment.ann_accuracy_train = -1
            experiment.ann_accuracy_test = dict((zip(model.metrics_names, eval_test)))["sparse_categorical_accuracy"]
        except ValueError:
            print("Error: The accuracy testing function looks for spare_categorical_accuracy, if another measurement "
                  "is used, change the key of the dict in ann_accuracy().")
            raise

    if not disable_training_accuracy:
        print("Train: " + str(dict(zip(model.metrics_names, eval_train))))
    print("Test: " + str(dict(zip(model.metrics_names, eval_test))))


def snn_accuracy(model_snn, ds_train_spike, ds_test_spike, experiment, max_images=None, disable_training_accuracy=False):
    """
    Test the accuracy of an SNN network with train and dev.
    If you supply a experiment, the values will be logged to the experiment
    :param model_snn:
    :param ds_train_spike:
    :param ds_test_spike:
    :param experiment:
    :param max_images
    :return:
    """
    # optimizer and loss makes no sense here, just metrics are interesting
    model_snn.compile(optimizer="adam", loss=None, metrics=['sparse_categorical_accuracy'])
    print("####### SNN ########")

    if max_images is not None:
        ds_train_spike = ds_train_spike.take(max_images)
        ds_test_spike = ds_test_spike.take(max_images)

    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}', histogram_freq=1)
    if not disable_training_accuracy:
        eval_train = model_snn.evaluate(ds_train_spike)
    eval_test = model_snn.evaluate(ds_test_spike)

    if experiment is not None:
        try:
            if not disable_training_accuracy:
                experiment.snn_accuracy_train = dict((zip(model_snn.metrics_names, eval_train)))["sparse_categorical_accuracy"]
            else:
                experiment.snn_accuracy_train = -1
            experiment.snn_accuracy_test = dict((zip(model_snn.metrics_names, eval_test)))["sparse_categorical_accuracy"]
        except ValueError:
            print("Error: The accuracy testing function looks for spare_categorical_accuracy, if another measurement "
                  "is used, change the key of the dict in snn_accuracy().")
            raise

    if not disable_training_accuracy:
        print("Train: " + str(dict(zip(model_snn.metrics_names, eval_train))))
    print("Test: " + str(dict(zip(model_snn.metrics_names, eval_test))))


def snn_accuracy_per_timestep(model_snn, ds_train_spike, ds_test_spike, experiment, max_images=None, disable_training_accuracy=False):
    """
    Test the accuracy of an SNN network with train and dev.
    If you supply a experiment, the values will be logged to the experiment
    :param model_snn:
    :param ds_train_spike:
    :param ds_test_spike:
    :param experiment:
    :param max_images
    :return:
    """
    # optimizer and loss makes no sense here, just metrics are interesting
    model_snn.compile(optimizer="adam", loss=None, metrics=['sparse_categorical_accuracy'])

    print("####### SNN ########")

    if max_images is not None:
        ds_train_spike = ds_train_spike.take(max_images)
        ds_test_spike = ds_test_spike.take(max_images)

    if not disable_training_accuracy:
        raise NotImplementedError("SNN train accuracy per timestep is not implemented.")

    # fixed_dataset = list(ds_test_spike.as_numpy_iterator())

    #eval_results = model_snn.predict(np.concatenate([x for x, y in fixed_dataset], axis=0))
    start_t = int(time.time())
    eval_results = model_snn.predict(ds_test_spike)
    print(f"Prediction took {int(time.time()) - start_t}s")

    accuracy_by_timestep = dict()
    data_test_results = np.concatenate([y for x, y in ds_test_spike], axis=0)
    for i in range(experiment.timesteps):
        print(f"SNN Accuracy for timestep {i} starting.")
        sca = tf.keras.metrics.SparseCategoricalAccuracy()
        sca.update_state(data_test_results, eval_results[:, i, :])
        accuracy_by_timestep[i] = sca.result().numpy()

    if experiment is not None:
        try:
            if not disable_training_accuracy:
                pass
            else:
                experiment.snn_accuracy_train = -1
            experiment.snn_accuracy_test = accuracy_by_timestep[experiment.timesteps - 1]
        except ValueError:
            print("Error: The accuracy testing function looks for spare_categorical_accuracy, if another measurement "
                  "is used, change the key of the dict in snn_accuracy().")
            raise

    if not disable_training_accuracy:
        pass
    print("Test: " + str(accuracy_by_timestep[experiment.timesteps - 1]))
    return accuracy_by_timestep


def ann_and_snn_activations_per_layer(model, model_snn, ds_test, ds_test_spike):
    # calculate activation outputs for SNN for debug purposes

    model_feature_extractor = tf.keras.Model(inputs=model.inputs,
                                             outputs=[layer.output for layer in model.layers])
    model_snn_feature_extractor = tf.keras.Model(inputs=model_snn.inputs,
                                                 outputs=[layer.output for layer in model_snn.layers])

    model_features = model_feature_extractor.predict(ds_test.take(5))
    model_snn_features = model_snn_feature_extractor.predict(ds_test_spike.take(5))

    return model_features, model_snn_features


def snn_activation_analysis(model, model_snn, ds_analysis, experiment, disable_training_accuracy=False):

    def analyze_layer_output(layer_output):
        """
        Only usable for ttfs and variants.
        Reshapes the output of a layer accordingly and sums over the timesteps.
        With the way ttfs is implemented, this will results in a number which resemble the inverted timestep of the
        first (and only) spike. If a neuron spikes at t=10 with 120 timesteps, the value here will be 120-10 = 110.
        :param layer_output:
        :return:
        """

        if isinstance(layer_output, np.ndarray):
            if len(layer_output.shape) >= 3:
                # first two dimensions are batch and timestep
                batch_size = layer_output.shape[0]
                timesteps = layer_output.shape[1]
                rest_size = int(np.prod(layer_output.shape[2:]))
                layer_output = layer_output.reshape((batch_size, timesteps, rest_size))
            else:
                # edge case for last layer which has no timesteps, just batch
                batch_size = layer_output.shape[0]
                rest_size = layer_output.shape[1]
        elif isinstance(layer_output, tuple):
            # ttfs_dyn_thresh outputs spikes and neuron potential as tuple
            batch_size = layer_output[0].shape[0]
            timesteps = layer_output[0].shape[1]
            rest_size = int(np.prod(layer_output[0].shape[2:]))
            layer_output = layer_output[0].reshape((batch_size, timesteps, rest_size))
        else:
            raise NotImplementedError("Unkown layer output. Only numpy arrays or tuples implemented.")
        return np.sum(layer_output, axis=1)

    model_snn_feature_extractor = tf.keras.Model(inputs=model_snn.inputs,
                                                 outputs=[layer.output for layer in model_snn.layers])

    activations_complete_list = list()
    activation_samples = 10
    for i in range(activation_samples):
        activations_complete_list.append(
            [analyze_layer_output(elem) for elem in model_snn_feature_extractor.predict(ds_analysis.take(1))]
        )

    # if the layer is a layer to communicate values for dyn_thresh, ignore it in the tensorboard results
    # as they are not interpretable the same way as spikes are and only make the graphs less readble by
    # providing no insights into the network
    usable_for_tensorboard = list()
    for i, elem in enumerate(activations_complete_list[0]):
        # WARNING: This setting also removes spiking layers that never spike, therefore only enabled for dyn_thresh
        if elem.max() == 0 and experiment.encoding == "ttfs_dyn_thresh":
            usable_for_tensorboard.append(False)
        elif elem.max() != int(elem.max()) and elem.max() != 0:
            # you might want to exclude layers that are not consisting of integers from tensorboard
            # as those layers could be dyn thresh layers with neuron potential values and no spikes
            usable_for_tensorboard.append(True)
        else:
            usable_for_tensorboard.append(True)
    assert len(usable_for_tensorboard) == len(activations_complete_list[0])

    def calc_first_spike(x):
        try:
            return experiment.timesteps - x[x > 0].max()
        except ValueError:
            return 0

    tbfile = tf.summary.create_file_writer(f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
    scalar_summary = dict()

    with tbfile.as_default():
        # plotting accuracy and weight distributions
        if not disable_training_accuracy:
            tf.summary.scalar("snn/eval/train", experiment.snn_accuracy_train, 0)
        scalar_summary["snn/eval/train"] = experiment.snn_accuracy_train

        tf.summary.scalar("snn/eval/test", experiment.snn_accuracy_test, 0)
        scalar_summary["snn/eval/test"] = experiment.snn_accuracy_test

        if not disable_training_accuracy:
            tf.summary.scalar("ann/eval/train", experiment.ann_accuracy_train, 0)
        scalar_summary["ann/eval/train"] = experiment.ann_accuracy_train

        tf.summary.scalar("ann/eval/test", experiment.ann_accuracy_test, 0)
        scalar_summary["ann/eval/test"] = experiment.ann_accuracy_test

        if not disable_training_accuracy:
            tf.summary.scalar("snn/diff/train", experiment.ann_accuracy_train - experiment.snn_accuracy_train, 0)
        scalar_summary["snn/diff/train"] = experiment.ann_accuracy_train - experiment.snn_accuracy_train

        tf.summary.scalar("snn/diff/test", experiment.ann_accuracy_test - experiment.snn_accuracy_test, 0)
        scalar_summary["snn/diff/test"] = experiment.ann_accuracy_test - experiment.snn_accuracy_test

        tf.summary.scalar("experiment/timesteps", experiment.timesteps, 0)
        scalar_summary["experiment/timesteps"] = experiment.timesteps

        tf.summary.scalar("experiment/epochs", experiment.epochs, 0)
        scalar_summary["experiment/epochs"] = experiment.epochs

        tf.summary.scalar("experiment/norm_percentile", experiment.norm_percentile, 0)
        scalar_summary["experiment/norm_percentile"] = experiment.norm_percentile

        tf.summary.scalar("experiment/threshold", experiment.threshold, 0)
        scalar_summary["experiment/threshold"] = experiment.threshold

        tf.summary.scalar("experiment/beta", experiment.beta, 0)
        scalar_summary["experiment/beta"] = experiment.beta

        tf.summary.scalar("experiment/tau", experiment.tau, 0)
        scalar_summary["experiment/tau"] = experiment.tau

        tf.summary.scalar("experiment/img_size", experiment.image_size_rescale[0], 0)
        scalar_summary["experiment/img_size"] = experiment.image_size_rescale[0]



        # TODO also save to disk for graphing

        for i, elem in enumerate(model_snn.layers):
            if len(elem.get_weights()) > 0:
                tf.summary.histogram(f"snn/weights/{elem.name}", elem.get_weights()[0].flatten(), 0)
        for i, elem in enumerate(model.layers):
            if len(elem.get_weights()) > 0:
                tf.summary.histogram(f"ann/weights/{elem.name}", elem.get_weights()[0].flatten(), 0)

        # TODO the whole generating samples and then uniting them process is more complicated than it has to be
        activations_per_layer_list = list()
        for i in range(len(activations_complete_list[0])):
            if i+1 == len(activations_complete_list[0]):
                # last layer
                activations_per_layer_list.append(
                    np.concatenate([elem[i] for elem in activations_complete_list], axis=0)
                )
            else:
                activations_per_layer_list.append(
                    np.concatenate([elem[i] for elem in activations_complete_list], axis=0)
                )

        # plotting activations
        step_correction = 0
        for i, activations in enumerate(activations_per_layer_list):
            # elem is the activation of one certain image
            if usable_for_tensorboard[i]:
                n_used_neurons_list = list(map(lambda x: len(x[x > 0]), activations))
                first_spike_per_neuron_list = list(map(calc_first_spike, activations))
                print(f"#### ACTIVATIONS LAYER {i} | {model_snn.layers[i].name} ####")

                print("Avg. # of spikes: " + str(np.average(n_used_neurons_list)))
                tf.summary.scalar(f"snn/no_of_spikes/avg",
                                     np.average(n_used_neurons_list), i - step_correction)

                print("Var. # of spikes: " + str(np.var(n_used_neurons_list)))
                tf.summary.scalar(f"snn/no_of_spikes/var",
                                     np.var(n_used_neurons_list), i - step_correction)

                print("Avg. timestep spike: " + str(np.average(first_spike_per_neuron_list)))
                tf.summary.scalar(f"snn/timestep_spikes/avg",
                                     np.average(first_spike_per_neuron_list), i - step_correction)

                print("Var. timestep spike: " + str(np.var(first_spike_per_neuron_list)))
                tf.summary.scalar(f"snn/timestep_spikes/var",
                                     np.var(first_spike_per_neuron_list), i - step_correction)

                # plot to tensorboard
                tf.summary.histogram(f"snn/no_of_spikes/{model_snn.layers[i].name}",
                                     np.array(n_used_neurons_list), 0)
                tf.summary.histogram(f"snn/spike_timesteps/{model_snn.layers[i].name}",
                                     np.array(first_spike_per_neuron_list), 0)
                tf.summary.histogram(f"snn/activations/{model_snn.layers[i].name}",
                                     activations.flatten()[activations.flatten() != 0], 0)
            else:
                step_correction += 1

    return scalar_summary
