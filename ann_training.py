import copy

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from os import path

from conversion import spikalize_img, timealize_img


def prepare_dataset_ann(experiment, dataset="mnist", split=None):
    if split is None:
        split = ["train", "test"]
        # split = ["train", "validation"]
    (ds_train, ds_test) = tfds.load(
        dataset,
        split=split,
        shuffle_files=True,
        as_supervised=True
    )

    def resize_masked(image, label):
        return tf.image.resize(image, experiment.image_size_rescale), label

    def depth_increase_masked(image, label):
        return tf.image.grayscale_to_rgb(image), label

    # prepare ds_train and ds_test for ANN training
    ds_train = ds_train.map(lambda x, y: (x / 255, y),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if experiment.image_size_rescale is not None:
        ds_train = ds_train.map(
            resize_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if experiment.image_depth_increase:
        ds_train = ds_train.map(
            depth_increase_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(experiment.batch_size_ann_training)
    ds_train = ds_train.batch(experiment.batch_size_ann_training)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(lambda x, y: (x / 255, y),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if experiment.image_size_rescale is not None:
        ds_test = ds_test.map(
            resize_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if experiment.image_depth_increase:
        ds_test = ds_test.map(
            depth_increase_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(experiment.batch_size_ann_training)
    #ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test


def prepare_dataset_spike(experiment, dataset="mnist", split=None, spiking=False):
    if split is None:
        split = ["train", "test"]
        # split = ["train", "validation"]
    (ds_train_spike, ds_test_spike) = tfds.load(
        dataset,
        split=split,
        shuffle_files=False,
        as_supervised=True
    )
    print("ATTENTION DATASET SHUFFLE IS OFF!")
    # spikalize had no x / 255
    def spikalize_img_masked(image, label):
        return spikalize_img(experiment, image, label)

    def timealize_img_masked(image, label):
        return timealize_img(experiment, image, label)

    def resize_masked(image, label):
        return tf.image.resize(image, experiment.image_size_rescale), label

    def depth_increase_masked(image, label):
        return tf.image.grayscale_to_rgb(image), label

    def add_dynthresh_dim_masked(image, label):
        return (image, tf.zeros((experiment.timesteps, image.shape[1], image.shape[2], image.shape[3]))), label
        #return (image, tf.zeros((experiment.timesteps, 28, 28, 1))), label

    # prepare spiking variants for SNN testing
    ds_train_spike = ds_train_spike.map(lambda x, y: (x / 255, y),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if experiment.image_size_rescale is not None:
        ds_train_spike = ds_train_spike.map(
            resize_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if experiment.image_depth_increase:
        ds_train_spike = ds_train_spike.map(
            depth_increase_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train_spike = ds_train_spike.map(
        timealize_img_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if experiment.encoding == "ttfs_dyn_thresh":
        ds_train_spike = ds_train_spike.map(
            add_dynthresh_dim_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train_spike = ds_train_spike.shuffle(experiment.batch_size_snn_eval)
    ds_train_spike = ds_train_spike.batch(experiment.batch_size_snn_eval)
    ds_train_spike = ds_train_spike.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test_spike = ds_test_spike.map(lambda x, y: (x / 255, y),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if experiment.image_size_rescale is not None:
        ds_test_spike = ds_test_spike.map(
            resize_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if experiment.image_depth_increase:
        ds_test_spike = ds_test_spike.map(
            depth_increase_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test_spike = ds_test_spike.map(
        timealize_img_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if experiment.encoding == "ttfs_dyn_thresh":
        ds_test_spike = ds_test_spike.map(
            add_dynthresh_dim_masked, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test_spike = ds_test_spike.batch(experiment.batch_size_snn_eval)
    #ds_test_spike = ds_test_spike.cache()
    ds_test_spike = ds_test_spike.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train_spike, ds_test_spike


def train_network_base(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modelbase_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(784, activation=activation_function),
            tf.keras.layers.Dense(600, activation=activation_function),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model


def train_network_base_scale1(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modelbasescale1_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(784, activation=activation_function),
            tf.keras.layers.Dense(500, activation=activation_function),
            tf.keras.layers.Dense(200, activation=activation_function),
            tf.keras.layers.Dense(84, activation=activation_function),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model


def train_network_base_scale2(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modelbasescale2_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(784, activation=activation_function),
            tf.keras.layers.Dense(784, activation=activation_function),
            tf.keras.layers.Dense(784, activation=activation_function),
            tf.keras.layers.Dense(784, activation=activation_function),
            tf.keras.layers.Dense(500, activation=activation_function),
            tf.keras.layers.Dense(200, activation=activation_function),
            tf.keras.layers.Dense(84, activation=activation_function),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model


def train_network_base_conv(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modelbaseconv_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Conv2D(3, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Conv2D(3, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(484, activation=activation_function),
            tf.keras.layers.Dense(200, activation=activation_function),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model


def train_network_fully_conv(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modelfullyconv_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Conv2D(3, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Conv2D(3, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Conv2D(6, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Conv2D(6, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Conv2D(6, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model


def train_network_fully_conv_small(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modelfullyconvsmall_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Conv2D(3, 3, activation=activation_function, padding="valid"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model

def train_network_base_maxpool(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modelbasemaxpool_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, 3, activation=activation_function, padding="same"),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(3, 3, activation=activation_function, padding="same"),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(192, activation=activation_function),
            tf.keras.layers.Dense(100, activation=activation_function),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model


def train_network_lenet5(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modellenet5_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, 5, activation=activation_function, padding="valid"),
            tf.keras.layers.Conv2D(6, 2, activation=activation_function, padding="valid", strides=2),
            tf.keras.layers.Conv2D(16, 5, activation=activation_function, padding="valid"),
            tf.keras.layers.Conv2D(16, 2, activation=activation_function, padding="valid", strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation=activation_function),
            tf.keras.layers.Dense(84, activation=activation_function),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model


def train_network_lenet5_maxpool(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modellenet5_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, 5, activation=activation_function, padding="valid"),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(16, 5, activation=activation_function, padding="valid"),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation=activation_function),
            tf.keras.layers.Dense(84, activation=activation_function),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model


def train_network_lenet5_maxpool_extensions(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modellenet5ext_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([                                                           # 64x64x3
            tf.keras.layers.Conv2D(32, 5, activation=activation_function, padding="valid"),     # 60x60x32
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),                                  # 30x30x32
            tf.keras.layers.BatchNormalization(),                                               # 30x30x32
            tf.keras.layers.Conv2D(64, 5, activation=activation_function, padding="valid"),     # 26x26x64
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),                                  # 13x13x64
            tf.keras.layers.Conv2D(128, 3, activation=activation_function, padding="valid"),    # 24x24x128
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),                                  # 12x12x128
            tf.keras.layers.BatchNormalization(),                                               # 12x12x128
            tf.keras.layers.Conv2D(256, 3, activation=activation_function, padding="valid"),    # 10x10x256
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),                                  # 5x5x256
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation=activation_function),
            tf.keras.layers.Dense(2048, activation=activation_function),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model


def train_network_denseonly1(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/model_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(784, activation='relu'),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test)
        model.save(model_savefile)

    return model


def train_network_batchnorm_etal(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/model_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        if experiment.encoding == "ttfs_clamped":
            def activation_function(x): return tf.keras.activations.relu(x, threshold=experiment.beta)
        else:
            activation_function = "relu"
        model = tf.keras.Sequential([
            #tf.keras.layers.Flatten(),
            tf.keras.layers.InputLayer(input_shape=(32,32,3)),
            #tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same'),
            #tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same'),
            #tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(8, 3, activation=activation_function, padding='same'),
            tf.keras.layers.Conv2D(8, 3, activation=activation_function, padding='same'),
            tf.keras.layers.Conv2D(8, 3, activation=activation_function, padding='same'),
            tf.keras.layers.Conv2D(8, 3, activation=activation_function, padding='same'),
            #tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(16, 3, activation=activation_function, padding='same'),
            tf.keras.layers.Conv2D(16, 3, activation=activation_function, padding='same'),
            tf.keras.layers.Conv2D(16, 3, activation=activation_function, padding='same'),
            #tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            #tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            #tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            #tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            #tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            #tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            #tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            #tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            #tf.keras.layers.Dense(5000, activation='relu'),
            #tf.keras.layers.Dense(5000, activation='relu'),
            #tf.keras.layers.Dense(5000, activation='relu'),
            #tf.keras.layers.Dense(1000, activation='relu'),
            #tf.keras.layers.Dense(500, activation='relu'),
            #tf.keras.layers.Dense(784, activation='relu'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Dense(500, activation='relu'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Dense(300, activation='relu'),
            #tf.keras.layers.Dense(150, activation='relu'),
            #tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        #model = tf.keras.applications.VGG16(include_top=True, weights=None, classes=10)
        #model = tf.keras.applications.EfficientNetB0(include_top=True, input_shape=[224, 224, 3], weights=None, classes=100)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f'tblogs/run{experiment.start_time}{experiment.tensorboard_title}')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test, callbacks=[tensorboard_callback])
        model.save(model_savefile)

    return model


def train_network_vgg16(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modelvgg_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        model = tf.keras.Sequential([
            #tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same'),
            #tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same'),
            #tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dense(1000, activation='softmax')
        ])
        model = tf.keras.applications.VGG16(include_top=True, weights=None, classes=100)
        model = tf.keras.applications.EfficientNetB0(include_top=True, weights=None, classes=100)
        model = tf.keras.applications.ResNet50(include_top=True, weights=None, classes=100)
        model.compile(optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test)
        model.save(model_savefile)

    return model


def train_network_resnet_ish(experiment, ds_train, ds_test, force_training=False):
    model_savefile = "./experiments/modelresnetish_" + experiment.name
    if path.exists(model_savefile) and not force_training:
        model = keras.models.load_model(model_savefile)
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 7, activation='relu', padding='same', strides=2, kernel_regularizer=keras.regularizers.l2(0.001)),    # 112, 112, 64
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),                                    # 56, 56, 64

            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),  # 56, 56, 64
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),  # 56, 56, 64
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),                        # 28, 28, 64
            #tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),  # 28, 28, 128
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),  # 28, 28, 128
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),                        # 14, 14, 128

            tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),  # 14, 14, 256
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),  # 14, 14, 256
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),                        # 7, 7, 256
            #tf.keras.layers.Dropout(0.5),

            #tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),  # 7, 7, 512
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),  # 7, 7, 512
            #tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1000, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(100, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(ds_train, epochs=experiment.epochs, validation_data=ds_test)
        model.save(model_savefile)

    return model