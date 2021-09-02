import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob


def plot_spikes_as_eventplot(layer_outs_snn, n, layer):
    spike_pattern = layer_outs_snn[layer][0][n].T
    spike_timesteps = [[] for i in range(len(spike_pattern))]
    for elem in np.argwhere(spike_pattern):
        spike_timesteps[elem[0]].append(elem[1])
    plt.eventplot(spike_timesteps)
    plt.show()


def total_spikes_per_layer(layer_outs_snn, n):
    # TODO leave out last layer? Right now that's not a "real" spiking layer
    spikes_per_layer = np.empty(0)
    for i, elem in enumerate(layer_outs_snn):
        spikes_per_layer = np.concatenate((spikes_per_layer, [np.count_nonzero(elem[0][n])]))
    return spikes_per_layer


def total_spikes(layer_outs_snn, n):
    return total_spikes_per_layer(layer_outs_snn, n).sum()


def who_spiked_when(layer_outs_snn, layer, batch):
    return np.argwhere(layer_outs_snn[layer][0][batch])


def spiking_intensity_per_timestep(layer_outs_snn, batch):
    for i, elem in enumerate(layer_outs_snn):
        print("Layer " + str(i) + " spikes at " + str(np.argwhere(elem[0][batch])[:, 0]))


def repeated_experiment_plots(start_time: str, tensorboard_title: str):
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    metrics_pd = pd.read_csv(f"./logs/run{start_time}{tensorboard_title}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].boxplot([list(metrics_pd["ann/eval/train"]),
                    list(metrics_pd["snn/eval/train"]),
                    list(metrics_pd["ann/eval/test"]),
                    list(metrics_pd["snn/eval/test"])],
                   labels=["ANN Train", "SNN Train", "ANN Test", "SNN Test"])
    axs[0].set_title("Accuracy")
    axs[1].boxplot([list(metrics_pd["snn/diff/train"]), list(metrics_pd["snn/diff/test"])], labels=["Train", "Test"])
    axs[1].set_title("Difference Accuracy (ANN - SNN)")
    fig.subplots_adjust(hspace=0.2)
    plt.savefig("./data_accuracy_per_threshold/acc_snn_25times.pdf")
    plt.show()

    #plt.boxplot(list(metrics_pd["snn/diff/train"]))
    #list(metrics_pd["snn/diff/test"])


def repeated_experiment_plots_diff_only(start_time: str, tensorboard_title: str):
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    metrics_pd = pd.read_csv(f"./logs/run{start_time}{tensorboard_title}")

    fig, axs = plt.subplots(1, 1, figsize=(8, 4))

    #axs[0].boxplot([list(metrics_pd["ann/eval/train"]),
    #                list(metrics_pd["snn/eval/train"]),
    #                list(metrics_pd["ann/eval/test"]),
    #                list(metrics_pd["snn/eval/test"])],
    #               labels=["ANN Train", "SNN Train", "ANN Test", "SNN Test"])
    #axs[0].set_title("Accuracy")
    axs.boxplot([list(metrics_pd["snn/diff/train"]), list(metrics_pd["snn/diff/test"])], labels=["Train", "Test"])
    axs.set_title("Difference Accuracy (ANN - SNN)")
    fig.subplots_adjust(hspace=0.2)
    #plt.savefig("./data_accuracy_per_threshold/acc_snn_25times_fixedann.pdf")
    plt.show()

    #plt.boxplot(list(metrics_pd["snn/diff/train"]))
    #list(metrics_pd["snn/diff/test"])


def repeated_experiment_plots_side_by_side(start_time1: str, tensorboard_title1: str,
                                           start_time2: str, tensorboard_title2: str):
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    metrics_pd1 = pd.read_csv(f"./logs/run{start_time1}{tensorboard_title1}")
    metrics_pd2 = pd.read_csv(f"./logs/run{start_time2}{tensorboard_title2}")

    fig, axs = plt.subplots(1, 1, figsize=(8, 4))

    #axs[0].boxplot([list(metrics_pd["ann/eval/train"]),
    #                list(metrics_pd["snn/eval/train"]),
    #                list(metrics_pd["ann/eval/test"]),
    #                list(metrics_pd["snn/eval/test"])],
    #               labels=["ANN Train", "SNN Train", "ANN Test", "SNN Test"])
    #axs[0].set_title("Accuracy")
    axs.boxplot([list(metrics_pd1["snn/diff/test"]), list(metrics_pd2["snn/diff/test"])], labels=["Baseline", "Larger Network"])
    axs.set_title("Difference Accuracy (ANN - SNN)")
    fig.subplots_adjust(hspace=0.2)
    #plt.savefig("./data_accuracy_per_threshold/acc_snn_25times_fixedann.pdf")
    plt.show()

    #plt.boxplot(list(metrics_pd["snn/diff/train"]))
    #list(metrics_pd["snn/diff/test"])


def repeated_experiment_plots_variant_comparison():
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    run_list = glob.glob(".\\logs\\run??????????valid_test28_base_*")

    measurements_snn_eval_test = list()
    plt.figure(figsize=(11, 6))
    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        measurements_snn_eval_test.append(list(metrics_pd["snn/diff/test"]))
    plt.boxplot(measurements_snn_eval_test, labels=["TTFS base", "TTFS dyn thresh", "TTFS clamped"])

    # plt.xlim(99.0, 100.01)
    # plt.ylim(-0.005, 0.0125)
    # plt.hlines([0], 0, 22, linestyles="dotted")
    #plt.xlabel("Normalization percentile")
    plt.ylabel("Diff (ANN - SNN)")
    # plt.xscale("log")

    # plt.legend()
    plt.savefig("./data_accuracy_per_threshold/acc_variant_comparison2.pdf")
    plt.show()


def repeated_experiment_plots_scaling():
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    run_list = glob.glob(".\\logs\\run??????????valid_test29_base_*") + glob.glob(".\\logs\\run??????????valid_test30_*")

    measurements_snn_eval_test = list()
    plt.figure(figsize=(11, 6))
    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        measurements_snn_eval_test.append(list(metrics_pd["ann/eval/test"]))
    plt.boxplot(measurements_snn_eval_test, labels=["base", "scale1", "scale2"])

    # plt.xlim(99.0, 100.01)
    # plt.ylim(-0.005, 0.0125)
    # plt.hlines([0], 0, 22, linestyles="dotted")
    #plt.xlabel("Normalization percentile")
    plt.ylabel("Diff (ANN - SNN)")
    # plt.xscale("log")

    # plt.legend()
    #plt.savefig("./data_accuracy_per_threshold/acc_variant_comparison.pdf")
    plt.show()


def repeated_experiment_plots_conv_maxpool():
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    run_list = glob.glob(".\\logs\\run??????????valid_test29_base*") + glob.glob(".\\logs\\run??????????valid_test31_*")

    measurements_snn_eval_test = list()
    plt.figure(figsize=(11, 6))
    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        measurements_snn_eval_test.append(list(metrics_pd["snn/diff/test"]))
    plt.boxplot(measurements_snn_eval_test, labels=["conv", "base", "maxpool"])

    # plt.xlim(99.0, 100.01)
    # plt.ylim(-0.005, 0.0125)
    # plt.hlines([0], 0, 22, linestyles="dotted")
    #plt.xlabel("Normalization percentile")
    plt.ylabel("Diff (ANN - SNN)")
    # plt.xscale("log")

    # plt.legend()
    #plt.savefig("./data_accuracy_per_threshold/acc_variant_comparison.pdf")
    plt.show()


def repeated_experiment_plots_lenet5():
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    run_list = glob.glob(".\\logs\\run??????????valid_test29_base_*") + \
               glob.glob(".\\logs\\run??????????valid_testH32_*") + \
               glob.glob(".\\logs\\run1619619320valid_test40_lenet5_dynthresh_thresh15_timestep175_norm99.999_25times") + \
               glob.glob(".\\logs\\run1619629562valid_test43_lenet5_clamped_thresh10_timestep125_norm99.99_beta1.52587890625e-05_25times") + \
               glob.glob(".\\logs\\run1620744038valid_exp8_clamped_125ts_thresh15_2-6")

    measurements_snn_eval_test = list()
    plt.figure(figsize=(13, 6))
    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        measurements_snn_eval_test.append(list(metrics_pd["snn/diff/test"]))
    plt.boxplot(measurements_snn_eval_test, labels=["Baseline (base)", "LN-5 (base)", "LN-5 (dyn thresh)", "LN-5 (dyn thresh), updated", "LN-5 (clamped)", "LN-5 (clamped), updated"])

    # plt.xlim(99.0, 100.01)
    plt.ylim(-0.005, 0.2)
    # plt.hlines([0], 0, 22, linestyles="dotted")
    #plt.xlabel("Normalization percentile")
    plt.ylabel("Diff (ANN - SNN)")
    # plt.xscale("log")

    # plt.legend()
    plt.savefig("./data_accuracy_per_threshold/acc_lenet5_variants_baseline_parameters3.pdf")
    plt.show()


def repeated_experiment_plots_lenet5ext():
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    run_list = glob.glob(".\\logs\\run??????????valid_exp20_*") + \
               glob.glob(".\\logs\\run??????????valid_exp21_*") + \
               glob.glob(".\\logs\\run??????????valid_exp26_*")

    measurements_snn_eval_test = list()
    plt.figure(figsize=(13, 6))
    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        measurements_snn_eval_test.append(list(metrics_pd["snn/diff/test"]))
    plt.boxplot(measurements_snn_eval_test, labels=["LeNetExtended (TTFS base)",
                                                    "LeNetExtended (TTFS dyn thresh)",
                                                    "LeNetExtended (TTFS clamped)"])

    # plt.xlim(99.0, 100.01)
    #plt.ylim(-0.005, 0.2)
    # plt.hlines([0], 0, 22, linestyles="dotted")
    #plt.xlabel("Normalization percentile")
    plt.ylabel("Diff (ANN - SNN)")
    # plt.xscale("log")

    # plt.legend()
    plt.savefig("./data_accuracy_per_threshold/acc_lenetext_variant_boxplot.pdf")
    plt.show()


def repeated_experiment_plots_input_variants():
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    run_list = glob.glob(".\\logs\\run??????????TEST_baseline_base_other_input") + \
               glob.glob(".\\logs\\run??????????TEST_baseline_base_normal_input")

    measurements_snn_eval_test = list()
    plt.figure(figsize=(12, 4))
    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        measurements_snn_eval_test.append(list(metrics_pd["snn/diff/test"]))
    plt.boxplot(measurements_snn_eval_test, labels=["Poisson Rate Input",
                                                    "Constant Current Input"])

    # plt.xlim(99.0, 100.01)
    #plt.ylim(-0.005, 0.2)
    # plt.hlines([0], 0, 22, linestyles="dotted")
    #plt.xlabel("Normalization percentile")
    plt.ylabel("Diff (ANN - SNN)")
    # plt.xscale("log")

    # plt.legend()
    plt.savefig("./data_accuracy_per_threshold/acc_input_variants.pdf")
    plt.show()


def repeated_experiment_timestep_plot(start_time: str, tensorboard_title: str, disable_training_accuracy=False):
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    metrics_pd = pd.read_csv(f"./logs/run{start_time}{tensorboard_title}")

    if not disable_training_accuracy:
        plt.plot(list(metrics_pd["experiment/timesteps"]), list(metrics_pd["snn/eval/train"]))
    plt.plot(list(metrics_pd["experiment/timesteps"]), list(metrics_pd["snn/eval/test"]))
    plt.ylim(0.95, 0.98)
    plt.xlabel("Time steps")
    plt.ylabel("Accuracy")
    plt.savefig("./data_accuracy_per_threshold/acc_t_thresh_10_zoomed.pdf")
    plt.show()

    #fix, axs = plt.subplots(1, 2, figsize=(12, 6))


def repeated_experiment_timestep_plot_lenet5ext():
    #metrics_pd = pd.read_csv(f"/home/ubuntu/testlogs_csv/run{start_time}{tensorboard_title}")
    run_list_44 = glob.glob(".\\logs\\run??????????valid_test44_*")
    run_list_45 = glob.glob(".\\logs\\run??????????valid_test45_*")
    measurement_snn_eval_test_44 = list()
    measurement_timesteps_44 = list()
    measurement_snn_eval_test_45 = list()
    measurement_timesteps_45 = list()

    for elem in run_list_44:
        metrics_pd = pd.read_csv(elem)
        measurement_snn_eval_test_44.append(metrics_pd["snn/eval/test"])
        measurement_timesteps_44.append(metrics_pd["experiment/timesteps"])

    for elem in run_list_45:
        metrics_pd = pd.read_csv(elem)
        measurement_snn_eval_test_45.append(metrics_pd["snn/eval/test"])
        measurement_timesteps_45.append(metrics_pd["experiment/timesteps"])

    plt.plot(measurement_timesteps_44, measurement_snn_eval_test_44)
    plt.plot(measurement_timesteps_45, measurement_snn_eval_test_45)

    #plt.ylim(0.95, 0.98)
    plt.xlabel("Time steps")
    plt.ylabel("Accuracy")
    #plt.savefig("./data_accuracy_per_threshold/acc_t_thresh_10_zoomed.pdf")
    plt.show()

    #fix, axs = plt.subplots(1, 2, figsize=(12, 6))


def repeated_experiment_timestep_plot_multiple_thresholds(disable_training_accuracy=False):
    #run_list = glob.glob(".\\logs\\run??????????valid_test7*")
    #run_list = glob.glob(".\\logs\\run??????????valid_test13_base_dynthresh_thresh*")
    #run_list = glob.glob(".\\logs\\run??????????valid_test38_*")
    run_list = glob.glob(".\\logs\\run??????????valid_test27_*")

    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        if not disable_training_accuracy:
            plt.plot(list(metrics_pd["experiment/timesteps"]), list(metrics_pd["snn/eval/train"]))
        plt.plot(list(metrics_pd["experiment/timesteps"]), list(metrics_pd["snn/eval/test"]), label=elem.split("_")[-4][6:])
    plt.ylim(0.92, 0.98)
    #plt.xlim(1, 200)
    plt.xlabel("Time steps")
    plt.ylabel("Accuracy")

    plt.legend()
    #plt.savefig("./data_accuracy_per_threshold/acc_t_per_thresh_zoomed.pdf")
    #plt.savefig("./data_accuracy_per_threshold/acc_dynthresh_t_per_thresh_zoomed.pdf")
    #plt.savefig("./data_accuracy_per_threshold/acc_lenet5_dynthresh_t_per_thresh_zoomed.pdf")
    #plt.savefig("./data_accuracy_per_threshold/acc_clamped_t_per_thresh_zoomed.pdf")
    plt.show()

    #fix, axs = plt.subplots(1, 2, figsize=(12, 6))


def repeated_experiment_timestep_plot_multiple_taus(disable_training_accuracy=False):
    run_list = glob.glob(".\\logs\\run??????????valid_test11_base_thresh10_timestep125_tau*")
    measurements_tau = list()
    measurements_snn_eval_train = list()
    measurements_snn_eval_test = list()
    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        if not disable_training_accuracy:
            measurements_snn_eval_train.append(metrics_pd["snn/eval/train"].values[0])
        measurements_tau.append(metrics_pd["experiment/tau"].values[0])
        measurements_snn_eval_test.append(metrics_pd["snn/eval/test"].values[0])

    if not disable_training_accuracy:
        plt.plot(measurements_tau, measurements_snn_eval_train)
    plt.plot(measurements_tau, measurements_snn_eval_test)
    plt.ylim(0.972, 0.976)
    plt.xlabel("Tau")
    plt.ylabel("Accuracy")
    plt.xscale("log")

    #plt.legend()
    plt.savefig("./data_accuracy_per_threshold/acc_per_tau_zoomed.pdf")
    #plt.show()

    #fix, axs = plt.subplots(1, 2, figsize=(12, 6))


def repeated_experiment_timestep_plot_multiple_normalization(disable_training_accuracy=False):
    #run_list = glob.glob(".\\logs\\run??????????valid_test12_base_thresh10_timestep125_norm*")
    #run_list = glob.glob(".\\logs\\run??????????valid_test23_base_dynthresh_thresh5_timestep125_norm*")
    run_list = glob.glob(".\\logs\\run??????????valid_test39_*")
    measurements_norm_percentile = list()
    measurements_snn_eval_train = list()
    measurements_snn_eval_test = list()
    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        if not disable_training_accuracy:
            measurements_snn_eval_train.append(metrics_pd["snn/eval/train"].values[0])
        measurements_norm_percentile.append(metrics_pd["experiment/norm_percentile"].values[0])
        measurements_snn_eval_test.append(metrics_pd["snn/eval/test"].values[0])
    if not disable_training_accuracy:
        plt.plot(measurements_norm_percentile, measurements_snn_eval_train)
    plt.plot(measurements_norm_percentile, measurements_snn_eval_test)

    #plt.xlim(99.0, 100.01)
    #plt.ylim(0.974, 0.975)
    plt.xlabel("Normalization percentile")
    plt.ylabel("Accuracy")
    #plt.xscale("log")

    #plt.legend()
    #plt.savefig("./data_accuracy_per_threshold/acc_per_normalized_zoomed.pdf")
    # plt.savefig("./data_accuracy_per_threshold/acc_dynthresh_per_normalized.pdf")
    plt.show()


def repeated_experiment_timestep_plot_multiple_normalization_boxplots(disable_training_accuracy=False):
    #run_list = glob.glob(".\\logs\\run??????????valid_test12_base_thresh10_timestep125_norm*")
    run_list = glob.glob(".\\logs\\run??????????valid_test25_base_dynthresh_thresh5_timestep125_norm*")
    measurements_norm_percentile = list()
    measurements_snn_eval_train = list()
    measurements_snn_eval_test = list()
    plt.figure(figsize=(11, 6))
    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        if not disable_training_accuracy:
            measurements_snn_eval_train.append(metrics_pd["snn/eval/train"].values[0])
        measurements_norm_percentile.append(metrics_pd["experiment/norm_percentile"].values[0])
        measurements_snn_eval_test.append(list(metrics_pd["snn/diff/test"]))
    if not disable_training_accuracy:
        plt.plot(measurements_norm_percentile, measurements_snn_eval_train)
    plt.boxplot(measurements_snn_eval_test, labels=measurements_norm_percentile)

    #plt.xlim(99.0, 100.01)
    #plt.ylim(-0.005, 0.0125)
    #plt.hlines([0], 0, 22, linestyles="dotted")
    plt.xlabel("Normalization percentile")
    plt.ylabel("Diff (ANN - SNN)")
    #plt.xscale("log")

    #plt.legend()
    #plt.savefig("./data_accuracy_per_threshold/acc_per_normalized_zoomed.pdf")
    #plt.savefig("./data_accuracy_per_threshold/acc_dynthresh_per_normalized.pdf")
    plt.show()


def repeated_experiment_timestep_plot_multiple_beta(disable_training_accuracy=False):
    #run_list = glob.glob(".\\logs\\run??????????valid_test26_base_clamped_thresh5_beta*")
    run_list = glob.glob(".\\logs\\run??????????valid_exp3_clamped_*")
    measurements_beta = list()
    measurements_snn_diff_train = list()
    measurements_snn_diff_test = list()
    plt.figure(figsize=(11, 6))
    for elem in run_list:
        metrics_pd = pd.read_csv(elem)

        if not disable_training_accuracy:
            measurements_snn_diff_train.append(metrics_pd["snn/eval/train"].values[0])
        measurements_beta.append(metrics_pd["experiment/beta"].values[0])
        measurements_snn_diff_test.append(list(metrics_pd["snn/eval/test"]))
    if not disable_training_accuracy:
        plt.boxplot(measurements_snn_diff_train, labels=measurements_beta)
    #labels = [r'0', r'$2^{-18}$', r'$2^{-17}$', r'$2^{-16}$', r'$2^{-15}$', r'$2^{-14}$', r'$2^{-13}$',
    #          r'$2^{-12}$', r'$2^{-11}$', r'$2^{-10}$', r'$2^{-9}$', r'$2^{-8}$', r'$2^{-7}$',
    #          r'$2^{-6}$', r'$2^{-5}$', r'$2^{-4}$', r'$2^{-3}$']
    labels = [r'0', r'$2^{-20}$', r'$2^{-16}$', r'$2^{-12}$', r'$2^{-8}$', r'$2^{-4}$']
    plt.boxplot(measurements_snn_diff_test, labels=labels)

    #plt.xlim(0, 0.1)
    #plt.ylim(0.0, 0.007)
    plt.xlabel("Beta")
    plt.ylabel("Diff (ANN - SNN)")
    #plt.xscale("log")

    #plt.legend()
    #plt.savefig("./data_accuracy_per_threshold/acc_clamped_per_beta.pdf")
    plt.show()


def repeated_experiment_baseline_vs_fullyconv():
    metrics_pd_fc = pd.read_csv(f"./logs/run1619363046valid_test33_fullyconv_thresh10_timestep125_norm99.99_25times")
    metrics_pd_base = pd.read_csv(f"./logs/run1619276820valid_test29_base_thresh10_timestep125_25times")

    fig, axs = plt.subplots(1, 2, figsize=(13, 6))

    axs[0].boxplot([list(metrics_pd_base["ann/eval/test"]),
                    list(metrics_pd_fc["ann/eval/test"])],
                   labels=["Baseline", "Fully Conv"])
    axs[0].set_title("Accuracy (ANN)")
    axs[1].boxplot([list(metrics_pd_base["snn/eval/test"]),
                    list(metrics_pd_fc["snn/eval/test"])],
                   labels=["Baseline", "Fully Conv"])
    axs[1].set_title("Accuracy (SNN)")

    fig.subplots_adjust(hspace=0.2)
    plt.savefig("./data_accuracy_per_threshold/acc_baseline_vs_fullyconv.pdf")
    plt.show()


def plot_relu_clampedrelu_sigmoid():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    def relu(x, beta=0.):
        if beta > 0 and x <= beta:
            return 0
        return max(0, x)

    def logistic_func(x, L=1, k=1, x0=0):
        return L / (1 + np.exp(-k * (x - x0)))

    x = np.linspace(-5, 5, 5000)
    axs[0].plot(x, list(map(relu, x)))
    axs[1].plot(x, list(map(lambda x: relu(x, beta=0.5), x)))
    axs[2].plot(x, list(map(logistic_func, x)))
    axs[0].set_title("ReLU")
    axs[1].set_title("Clamped ReLU with $\\beta=0.5$")
    axs[2].set_title(r"Logistic function with $L=1$,$k=1$,$x_0=0$")
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    plt.savefig("./data_accuracy_per_threshold/activation_functions.pdf", bbox_inches="tight")
    plt.show(bbox_inches="tight")


def test_timestep_pickle_plot():
    run_list = glob.glob(".\\logs\\run??????????_abt_valid_exp9*")
    #run_list = glob.glob(".\\logs\\run??????????_abt_valid_exp7_*")
    result_list = list()
    for elem in run_list:
        with open(elem, "rb") as handle:
            result_list.append(pickle.load(handle))

    for i, elem in enumerate(result_list):
        if int(run_list[i][-2:]) in [20, 25, 30, 35, 40, 50, 60, 70]:
            plt.plot(list(elem[0].values()), label=run_list[i][-2:])
    #plt.xlim(200, 1000)
    #plt.ylim(0.73, 0.79)
    plt.legend()
    # plt.savefig("./data_accuracy_per_threshold/acc_lenetext_base_per_timesteps_zoomed.pdf")
    # plt.savefig("./data_accuracy_per_threshold/acc_lenetext_dynthresh_per_timesteps_zoomed.pdf")
    plt.show()

    a = 1


if __name__ == '__main__':
    #repeated_experiment_timestep_plot("1618797275", "valid_test7_base_timesteps_1_200_thresh10_fixedann", True)
    #repeated_experiment_timestep_plot_multiple_taus(True)
    #repeated_experiment_timestep_plot_multiple_normalization(True)
    #repeated_experiment_plots("1618939422", "valid_test10_base_thresh10_timestep125_25_times")
    #repeated_experiment_plots_diff_only("1618937615", "valid_test10_base_thresh10_timestep125_25_times_stickyann")
    #repeated_experiment_timestep_plot_multiple_thresholds(True)
    #repeated_experiment_timestep_plot_multiple_beta(True)
    #repeated_experiment_timestep_plot_multiple_normalization(True)
    #repeated_experiment_timestep_plot_multiple_normalization_boxplots(True)
    #repeated_experiment_plots_side_by_side("1619168185", "valid_test18_base_thresh10_timestep125_multiple25",
    #                                       "1619168778", "valid_test18_base_dynthresh_thresh5_timestep125_multiple25")
    #repeated_experiment_plots_side_by_side("1619191952", "valid_test21_base_thresh10_timestep125_25times",
    #                                       "1619192557", "valid_test21_basescale1_thresh10_timestep125_25times")
    #repeated_experiment_plots_variant_comparison()
    #repeated_experiment_plots_scaling()
    #repeated_experiment_plots_conv_maxpool()
    #repeated_experiment_plots_lenet5()
    #repeated_experiment_baseline_vs_fullyconv()
    #repeated_experiment_timestep_plot_lenet5ext()
    #plot_relu_clampedrelu_sigmoid()
    #test_timestep_pickle_plot()
    repeated_experiment_plots_lenet5ext()
    #repeated_experiment_plots_input_variants()