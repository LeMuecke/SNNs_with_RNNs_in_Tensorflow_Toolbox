import pickle
import sys

import numpy as np
import pandas as pd

from ann_training import train_network_denseonly1, train_network_batchnorm_etal, train_network_vgg16, \
    prepare_dataset_spike, prepare_dataset_ann, train_network_base, train_network_lenet5, train_network_lenet5_maxpool, \
    train_network_base_scale1, train_network_base_conv, train_network_base_scale2, train_network_base_maxpool, \
    train_network_fully_conv, train_network_fully_conv_small, train_network_lenet5_maxpool_extensions, \
    train_network_resnet_ish
from conversion import convert_ann_to_snn, snn_accuracy, ann_accuracy, ann_and_snn_activations_per_layer, \
    snn_activation_analysis, snn_accuracy_per_timestep
import time
import tensorflow as tf
from optimization import OptimizableValue, DiscreteValues, RealInterval
from plotting import plot_spikes_as_eventplot, who_spiked_when, spiking_intensity_per_timestep, \
    repeated_experiment_plots, repeated_experiment_timestep_plot


class Experiment:
    def __init__(self, name, batch_size_ann_training, batch_size_snn_eval, timesteps, max_rate, epochs,
                 norm_percentile=100.0, tau=10.0, threshold=0.1, beta=0.0, encoding="rate", image_size_rescale=None,
                 image_depth_increase=None, tensorboard_title="", dataset="mnist", training_function=None,
                 force_training=True, max_images=None, disable_training_accuracy=False):

        self.allowed_encodings = ["rate", "ttfs", "ttfs_dyn_thresh", "ttfs_clamped"]

        if encoding not in self.allowed_encodings:
            raise ValueError("Encoding has to be one of the following: " + str(self.allowed_encodings))

        self.name = name
        self.batch_size_ann_training = batch_size_ann_training
        self.batch_size_snn_eval = batch_size_snn_eval
        self.timesteps = timesteps
        self.max_rate = max_rate
        self.epochs = epochs
        self.norm_percentile = norm_percentile
        self.tau = tau
        self.threshold = threshold
        self.beta = beta
        self.encoding = encoding
        self.dataset = dataset
        self.training_function = training_function
        self.force_training = force_training
        self.max_images = max_images
        self.disable_training_accuracy = disable_training_accuracy
        self.ann_accuracy_train = None
        self.ann_accuracy_test = None
        self.snn_accuracy_train = None
        self.snn_accuracy_test = None
        self.image_size_rescale = image_size_rescale
        self.image_depth_increase = image_depth_increase
        self.tensorboard_title = tensorboard_title
        self.start_time = str(int(time.time()))

    def run(self, timestep_analysis=False):
        time_breakpoint = int(time.time())

        ds_train, ds_test = prepare_dataset_ann(self, dataset=self.dataset)
        ds_normalize_train, ds_normalize_test = prepare_dataset_ann(self, dataset=self.dataset)
        ds_normalize = ds_normalize_train.concatenate(ds_normalize_test)
        ds_train_spike, ds_test_spike = prepare_dataset_spike(self, dataset=self.dataset)

        time_breakpoint = print_spent_time(time_breakpoint, "loading data")

        model = self.training_function(self, ds_train, ds_test, force_training=self.force_training)
        time_breakpoint = print_spent_time(time_breakpoint, "training model")

        model_snn = convert_ann_to_snn(self, model, ds_normalize)
        time_breakpoint = print_spent_time(time_breakpoint, "converting ann -> snn")

        ann_accuracy(model, ds_train, ds_test, experiment=self, max_images=self.max_images,
                     disable_training_accuracy=self.disable_training_accuracy)
        time_breakpoint = print_spent_time(time_breakpoint, "calculating accuracy ann")
        if timestep_analysis:
            accuracy_per_timestep = snn_accuracy_per_timestep(model_snn, ds_train_spike, ds_test_spike, self,
                                                              max_images=self.max_images,
                                                              disable_training_accuracy=self.disable_training_accuracy)
        else:
            snn_accuracy(model_snn, ds_train_spike, ds_test_spike, self, max_images=self.max_images,
                         disable_training_accuracy=self.disable_training_accuracy)
        time_breakpoint = print_spent_time(time_breakpoint, "calculating accuracy snn")

        #layer_outs_snn = ann_and_snn_activations_per_layer(model, model_snn, ds_test, ds_test_spike)
        time_breakpoint = print_spent_time(time_breakpoint, "activations debugging")

        metrics = snn_activation_analysis(model, model_snn, ds_test_spike, self,
                                          disable_training_accuracy=self.disable_training_accuracy)

        # plot_spikes_as_eventplot(layer_outs_snn, 0, 0)
        # print(who_spiked_when(layer_outs_snn, 4, 0))
        # spiking_intensity_per_timestep(layer_outs_snn, 0)
        if timestep_analysis:
            return metrics, accuracy_per_timestep
        return metrics


class RepeatedExperiment(Experiment):
    """
    A repeated experiment is, as the name tells, an experimen that gets repeated repetitions times.
    If one variable is set to an array built like [start, stop] instead of an integer, the class will not repeat
    the same experiment for repetitions times, but do (stop - start) runs, while setting the respective variable
    to any integer number between start and stop.
    """
    def __init__(self, name, batch_size_ann_training, batch_size_snn_eval, timesteps, max_rate, epochs,
                 norm_percentile=100.0, tau=10.0, threshold=0.1, beta=0.0, encoding="rate", image_size_rescale=None,
                 image_depth_increase=None, tensorboard_title="", dataset="mnist", training_function=None,
                 force_training=True, max_images=None, disable_training_accuracy=False, repetitions=1):

        if isinstance(timesteps, list):
            self.interval = timesteps
            timesteps = self.interval[0]
        else:
            self.interval = None

        super().__init__(name=name, batch_size_ann_training=batch_size_ann_training,
                         batch_size_snn_eval=batch_size_snn_eval, timesteps=timesteps, max_rate=max_rate, epochs=epochs,
                         norm_percentile=norm_percentile, tau=tau, threshold=threshold, beta=beta, encoding=encoding,
                         image_size_rescale=image_size_rescale, image_depth_increase=image_depth_increase,
                         tensorboard_title=tensorboard_title, dataset=dataset, training_function=training_function,
                         force_training=force_training, disable_training_accuracy=disable_training_accuracy,
                         max_images=max_images)
        self.repetitions = repetitions
        self.accuracy_by_timestep_list = list()
        self.scalar_summary = {
            "snn/eval/train": list(),
            "snn/eval/test": list(),
            "ann/eval/train": list(),
            "ann/eval/test": list(),
            "snn/diff/train": list(),
            "snn/diff/test": list(),
            "experiment/timesteps": list(),
            "experiment/epochs": list(),
            "experiment/norm_percentile": list(),
            "experiment/threshold": list(),
            "experiment/beta": list(),
            "experiment/img_size": list(),
            "experiment/tau": list()
        }

    def save_metrics(self, timestep_analysis=False):
        pd.DataFrame(self.scalar_summary).to_csv(
            f'/home/ubuntu/testlogs_csv/run{self.start_time}{self.tensorboard_title}')
        if timestep_analysis:
            with open(f'/home/ubuntu/testlogs_csv/run{self.start_time}_abt_{self.tensorboard_title}', "wb") as handle:
                pickle.dump(self.accuracy_by_timestep_list, handle)

    def run_repeat(self, timestep_analysis=False):
        if self.interval is None:
            try:
                for i in range(self.repetitions):
                    if timestep_analysis:
                        metrics, accuracy_per_timestep = self.run(timestep_analysis=timestep_analysis)
                        self.accuracy_by_timestep_list.append(accuracy_per_timestep)
                    else:
                        metrics = self.run(timestep_analysis=timestep_analysis)
                    for elem in metrics.keys():
                        self.scalar_summary[elem].append(metrics[elem])
                    tf.keras.backend.clear_session()
                self.save_metrics(timestep_analysis=timestep_analysis)
            except:
                t, v, tb = sys.exc_info()
                self.save_metrics(timestep_analysis=timestep_analysis)
                raise t(v).with_traceback(tb)
        else:
            try:
                for i in range(self.interval[0], self.interval[1] + 1):
                    self.timesteps = i
                    if timestep_analysis:
                        metrics, accuracy_per_timestep = self.run(timestep_analysis=timestep_analysis)
                        self.accuracy_by_timestep_list.append(accuracy_per_timestep)
                    else:
                        metrics = self.run(timestep_analysis=timestep_analysis)
                    for elem in metrics.keys():
                        self.scalar_summary[elem].append(metrics[elem])
                    tf.keras.backend.clear_session()
                self.save_metrics(timestep_analysis=timestep_analysis)
            except:
                t, v, tb = sys.exc_info()
                self.save_metrics(timestep_analysis=timestep_analysis)
                raise t(v).with_traceback(tb)


def print_spent_time(time_breakpoint, part_name):
    time_diff = int(time.time()) - time_breakpoint
    print("%%% Time spent with " + part_name + ": " + str(time_diff) + "s")
    return int(time.time())


def experiment_base():
    # based on Rueckauer and Liu 2018
    # setup experiment hyperparameters
    exp = Experiment(name="experiment_base",
                     tensorboard_title="test",
                     batch_size_ann_training=128,
                     batch_size_snn_eval=32,
                     timesteps=120,
                     max_rate=2,
                     epochs=2,
                     norm_percentile=99.0,
                     tau=999999.0,
                     threshold=10,
                     beta=0.1,
                     encoding="ttfs",
                     image_size_rescale=[28, 28],
                     image_depth_increase=False,
                     dataset="mnist",
                     training_function=train_network_base,
                     force_training=True,
                     max_images=None)
    # hint: if encoding is rate, use threshold of 0.1, if ttfs, threshold of 10, if ttfs_dyn_thresh about 5ish

    exp.run()


def experiment1():
    # setup experiment hyperparameters
    exp = Experiment(name="experiment1",
                     batch_size_ann_training=16,
                     batch_size_snn_eval=64,
                     timesteps=100,
                     max_rate=2,
                     epochs=1,
                     norm_percentile=99.99,
                     tau=10.0,
                     threshold=10,
                     encoding="ttfs")

    time_breakpoint = int(time.time())
    ds_train, ds_test = prepare_dataset_ann(exp, dataset="mnist")
    ds_train_spike, ds_test_spike = prepare_dataset_spike(exp, dataset="mnist")
    time_breakpoint = print_spent_time(time_breakpoint, "loading data")

    model = train_network_denseonly1(exp, ds_train, ds_test, force_training=True)
    time_breakpoint = print_spent_time(time_breakpoint, "training model")

    model_snn = convert_ann_to_snn(exp, model, ds_train)
    time_breakpoint = print_spent_time(time_breakpoint, "converting ann -> snn")

    #ann_accuracy(model, ds_train, ds_test, experiment=exp, max_images=200)
    time_breakpoint = print_spent_time(time_breakpoint, "calculating accuracy ann")
    snn_accuracy(model_snn, ds_train_spike, ds_test_spike, exp)
    time_breakpoint = print_spent_time(time_breakpoint, "calculating accuracy snn")

    layer_outs_snn = ann_and_snn_activations_per_layer(model, model_snn, ds_test, ds_test_spike)
    time_breakpoint = print_spent_time(time_breakpoint, "activations debugging")

    plot_spikes_as_eventplot(layer_outs_snn, 0, 0)
    # print(who_spiked_when(layer_outs_snn, 4, 0))
    # spiking_intensity_per_timestep(layer_outs_snn, 0)
    a = 1


def experiment2():
    # setup experiment hyperparameters
    exp = Experiment(name="experiment2",
                     tensorboard_title="ttfs-clamped-cifar100(32x32)-convonly",
                     batch_size_ann_training=128,
                     batch_size_snn_eval=1,
                     timesteps=120,
                     max_rate=2,
                     epochs=8,
                     norm_percentile=99.0,
                     tau=10.0,
                     threshold=0.08,
                     beta=0.1,
                     encoding="ttfs_clamped",
                     image_size_rescale=[32, 32],
                     image_depth_increase=False)
    # hint: if encoding is rate, use threshold of 0.1, if ttfs, threshold of 10, if ttfs_dyn_thresh about 5ish

    time_breakpoint = int(time.time())
    ds_train, ds_test = prepare_dataset_ann(exp, dataset="cifar10")
    ds_normalize_train, ds_normalize_test = prepare_dataset_ann(exp, dataset="cifar10")
    ds_normalize = ds_normalize_train.concatenate(ds_normalize_test)
    ds_train_spike, ds_test_spike = prepare_dataset_spike(exp, dataset="cifar10")
    time_breakpoint = print_spent_time(time_breakpoint, "loading data")

    model = train_network_batchnorm_etal(exp, ds_train, ds_test, force_training=True)
    #model = train_network_vgg16(exp, ds_train, ds_test, force_training=True)
    time_breakpoint = print_spent_time(time_breakpoint, "training model")

    model_snn = convert_ann_to_snn(exp, model, ds_normalize)
    time_breakpoint = print_spent_time(time_breakpoint, "converting ann -> snn")

    #ann_accuracy(model, ds_train, ds_test, experiment=exp, max_images=200)
    time_breakpoint = print_spent_time(time_breakpoint, "calculating accuracy ann")
    snn_accuracy(model_snn, ds_train_spike, ds_test_spike, exp, max_images=200)
    time_breakpoint = print_spent_time(time_breakpoint, "calculating accuracy snn")

    layer_outs_snn = ann_and_snn_activations_per_layer(model, model_snn, ds_test, ds_test_spike)
    time_breakpoint = print_spent_time(time_breakpoint, "activations debugging")

    test = snn_activation_analysis(model, model_snn, ds_test_spike, exp)

    #plot_spikes_as_eventplot(layer_outs_snn, 0, 0)
    # print(who_spiked_when(layer_outs_snn, 4, 0))
    # spiking_intensity_per_timestep(layer_outs_snn, 0)
    a = 1


def experiment2_optim():
    opt_timesteps = OptimizableValue("timesteps", DiscreteValues([120,]))
    #opt_max_rate = OptimizableValue("max_rate", DiscreteValues([1, 2, 3, 4]))
    opt_norm_percentile = OptimizableValue("norm_percentile", DiscreteValues([99.0, 99.5, 99.9, 99.95, 99.99, 99.999, 99.9999]))
    #opt_threshold = OptimizableValue("threshold", DiscreteValues([0.1, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 10.0, 18.0, 25.0]))
    opt_threshold = OptimizableValue("threshold", RealInterval("uniform", low=2.0, high=20.0))
    #opt_beta = OptimizableValue("threshold", RealInterval("uniform", low=8.5422e-3, high=8.5422e-3))

    log_filename = "experiment2_optim" + str(int(time.time())) + ".txt"
    experiment_counter = 0
    ann_accuracy_train_max = 0
    ann_accuracy_test_max = 0
    snn_accuracy_train_max = 0
    snn_accuracy_test_max = 0
    try:
        while True:
            #tf.random.set_seed(42)
            tf.keras.backend.clear_session()
            tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
            opt_timesteps_value = opt_timesteps.get()
            opt_threshold_value = opt_threshold.get()
            opt_norm_percentile_value = opt_norm_percentile.get()
            #opt_beta_value = opt_beta.get()
            exp = Experiment(name="experiment2_optim",
                             tensorboard_title="ttfsclamped-optim2-thresh-verify",
                             batch_size_ann_training=128,
                             batch_size_snn_eval=64,
                             timesteps=opt_timesteps_value,
                             max_rate=2,
                             epochs=10,
                             norm_percentile=opt_norm_percentile_value,
                             tau=10.0,
                             threshold=opt_threshold_value,
                             beta=0.1,
                             encoding="ttfs",
                             image_size_rescale=[28, 28],
                             image_depth_increase=False,
                             dataset="mnist",
                             training_function=train_network_base,
                             force_training=True,
                             max_images=None)
            # hint: if encoding is rate, use threshold of 0.1, if ttfs, threshold of 10

            ds_train, ds_test = prepare_dataset_ann(exp, dataset="cifar10")
            ds_normalize_train, ds_normalize_test = prepare_dataset_ann(exp, dataset="cifar10")
            ds_normalize = ds_normalize_train.concatenate(ds_normalize_test)
            ds_train_spike, ds_test_spike = prepare_dataset_spike(exp, dataset="cifar10")

            model = train_network_batchnorm_etal(exp, ds_train, ds_test, force_training=True)

            model_snn = convert_ann_to_snn(exp, model, ds_normalize)

            ann_accuracy(model, ds_train, ds_test, experiment=exp, max_images=200)
            snn_accuracy(model_snn, ds_train_spike, ds_test_spike, experiment=exp, max_images=200)

            test = snn_activation_analysis(model, model_snn, ds_test_spike, exp)

            with open(log_filename, "a") as logfile:
                logfile.write("###### EXPERIMENT " + str(experiment_counter) + " ######" + "\r\n")
                logfile.write("Setup:" + "\r\n")
                logfile.write(" " + opt_timesteps.name + ": " + str(opt_timesteps_value) + "\r\n")
                #logfile.write(" " + opt_max_rate.name + ": " + str(opt_max_rate_value) + "\r\n")
                #logfile.write(" " + opt_norm_percentile.name + ": " + str(opt_norm_percentile_value) + "\r\n")
                logfile.write(" " + opt_threshold.name + ": " + str(opt_threshold_value) + "\r\n")
                logfile.write("Result:" + "\r\n")
                #logfile.write(" ANN Acc Train: " + str(exp.ann_accuracy_train) + "\r\n")
                logfile.write(" SNN Acc Train: " + str(exp.snn_accuracy_train) + "\r\n")
                #logfile.write(" ANN Acc Test: " + str(exp.ann_accuracy_test) + "\r\n")
                logfile.write(" SNN Acc Test: " + str(exp.snn_accuracy_test) + "\r\n")

            #ann_accuracy_train_max = max(ann_accuracy_train_max, exp.ann_accuracy_train)
            snn_accuracy_train_max = max(snn_accuracy_train_max, exp.snn_accuracy_train)
            #ann_accuracy_test_max = max(ann_accuracy_test_max, exp.ann_accuracy_test)
            snn_accuracy_test_max = max(snn_accuracy_test_max, exp.snn_accuracy_test)

            print("...finished experiment " + str(experiment_counter) + "!" + "\r\n")
            experiment_counter += 1

    except KeyboardInterrupt:
        with open(log_filename, "a") as logfile:
            logfile.write("!!!!!! END SUMMARY !!!!!!" + "\r\n")
            logfile.write("ANN Accuracy Train Max: " + str(ann_accuracy_train_max) + "\r\n")
            logfile.write("SNN Accuracy Train Max: " + str(snn_accuracy_train_max) + "\r\n")
            logfile.write("ANN Accuracy Test Max: " + str(ann_accuracy_test_max) + "\r\n")
            logfile.write("SNN Accuracy Test Max: " + str(snn_accuracy_test_max) + "\r\n")
            logfile.write("" + "\r\n")
            logfile.write("Optimization ended." + "\r\n")
        raise


def experiment3():
    # setup experiment hyperparameters
    exp = Experiment(name="experiment3",
                     batch_size_ann_training=16,
                     batch_size_snn_eval=64,
                     timesteps=100,
                     max_rate=2,
                     epochs=1,
                     norm_percentile=99.99,
                     tau=10.0,
                     threshold=10.0,
                     encoding="ttfs")
    # hint: if encoding is rate, use threshold of 0.1, if ttfs, threshold of 10

    time_breakpoint = int(time.time())
    ds_train, ds_test = prepare_dataset_ann(exp, dataset="mnist")
    ds_train_spike, ds_test_spike = prepare_dataset_spike(exp, dataset="mnist")
    time_breakpoint = print_spent_time(time_breakpoint, "loading data")

    model = train_network_denseonly1(exp, ds_train, ds_test, force_training=True)
    time_breakpoint = print_spent_time(time_breakpoint, "training model")

    model_snn = convert_ann_to_snn(exp, model, ds_train)
    time_breakpoint = print_spent_time(time_breakpoint, "converting ann -> snn")

    ann_accuracy(model, ds_train, ds_test)
    time_breakpoint = print_spent_time(time_breakpoint, "calculating accuracy ann")
    snn_accuracy(model_snn, ds_train_spike, ds_test_spike)
    time_breakpoint = print_spent_time(time_breakpoint, "calculating accuracy snn")

    layer_outs_snn = ann_and_snn_activations_per_layer(model, model_snn, ds_test, ds_test_spike)
    time_breakpoint = print_spent_time(time_breakpoint, "activations debugging")

    plot_spikes_as_eventplot(layer_outs_snn, 0, 0)
    # print(who_spiked_when(layer_outs_snn, 4, 0))
    # spiking_intensity_per_timestep(layer_outs_snn, 0)
    a = 1


if __name__ == '__main__':

    # print("ATTENTION RETURN SEQUENCE IN LAST LAYER ENABLED")

    #tf.random.set_seed(13)
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

    #for i in [0, 2 ** -20, 2 ** -16, 2 ** -12, 2 ** -8, 2 ** -4]:
    #for i in [2 ** -4]:

    #tf.random.set_seed(13)
    expa = RepeatedExperiment(name="experiment_base",
                              tensorboard_title=f"valid_exp25_dynthresh_450timesteps_thresh30",
                              batch_size_ann_training=64,
                              batch_size_snn_eval=8,
                              timesteps=450,
                              max_rate=2,
                              epochs=2,
                              norm_percentile=99.99,
                              tau=1024,
                              threshold=30,
                              beta=2 ** -16,
                              encoding="ttfs_dyn_thresh",
                              image_size_rescale=[64, 64],  # 64 64
                              image_depth_increase=True,
                              dataset="mnist",
                              # training_function=train_network_base,
                              training_function=train_network_lenet5_maxpool_extensions,
                              force_training=True,
                              max_images=None,
                              disable_training_accuracy=True,
                              repetitions=25)
    expa.run_repeat(timestep_analysis=False)

    expa = RepeatedExperiment(name="experiment_base",
                              tensorboard_title=f"valid_exp26_dynthresh_500timesteps_thresh30",
                              batch_size_ann_training=64,
                              batch_size_snn_eval=8,
                              timesteps=500,
                              max_rate=2,
                              epochs=2,
                              norm_percentile=99.99,
                              tau=1024,
                              threshold=30,
                              beta=2 ** -16,
                              encoding="ttfs_dyn_thresh",
                              image_size_rescale=[64, 64],  # 64 64
                              image_depth_increase=True,
                              dataset="mnist",
                              # training_function=train_network_base,
                              training_function=train_network_lenet5_maxpool_extensions,
                              force_training=True,
                              max_images=None,
                              disable_training_accuracy=True,
                              repetitions=25)
    expa.run_repeat(timestep_analysis=False)

    expa = RepeatedExperiment(name="experiment_base",
                              tensorboard_title=f"valid_exp27_dynthresh_530timesteps_thresh30",
                              batch_size_ann_training=64,
                              batch_size_snn_eval=8,
                              timesteps=530,
                              max_rate=2,
                              epochs=2,
                              norm_percentile=99.99,
                              tau=1024,
                              threshold=30,
                              beta=2 ** -16,
                              encoding="ttfs_dyn_thresh",
                              image_size_rescale=[64, 64],  # 64 64
                              image_depth_increase=True,
                              dataset="mnist",
                              # training_function=train_network_base,
                              training_function=train_network_lenet5_maxpool_extensions,
                              force_training=True,
                              max_images=None,
                              disable_training_accuracy=True,
                              repetitions=25)
    expa.run_repeat(timestep_analysis=False)

    #for i in [86., 86.1, 86.2, 86.3, 86.4, 86.5, 86.6, 86.7, 86.8, 86.9, 87., 87.1, 87.2, 87.3, 87.4, 87.5, 87.6,
    #          87.7, 87.8, 87.9, 88., 88.1, 88.2, 88.3, 88.4, 88.5, 88.6, 88.7, 88.8, 88.9, 89., 89.1, 89.2, 89.3,
    #          89.4, 89.5, 89.6, 89.7, 89.8, 89.9,
    #          90., 90.1, 90.2, 90.3, 90.4, 90.5, 90.6, 90.7, 90.8, 90.9, 91., 91.1, 91.2, 91.3, 91.4, 91.5, 91.6, 91.7,
    #          91.8, 91.9, 92., 92.1, 92.2, 92.3, 92.4, 92.5, 92.6, 92.7, 92.8, 92.9, 93., 93.1, 93.2, 93.3, 93.4, 93.5,
    #          93.6, 93.7, 93.8, 93.9, 94., 94.1, 94.2, 94.3, 94.4, 94.5, 94.6, 94.7, 94.8, 94.9, 95., 95.1, 95.2, 95.3,
    #          95.4, 95.5, 95.6, 95.7, 95.8, 95.9, 96., 96.1, 96.2, 96.3, 96.4, 96.5, 96.6, 96.7, 96.8, 96.9, 97., 97.1,
    #          97.2, 97.3, 97.4, 97.5, 97.6, 97.7, 97.8, 97.9, 98., 98.1, 98.2, 98.3, 98.4, 98.5, 98.6, 98.7, 98.8, 98.9,
    #          99., 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9, 100.0]:
