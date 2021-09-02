# TensorFlow Spiking Neural Network Toolbox

##### This toolbox offers a variety of tools for researching spiking neural networks, mainly a full conversion from a trained TensorFlow model to a working Spiking Neural Networks, implemented completely within TensorFlow using RNNs.  

---

### Main features
* Training of ANN
* Conversion of ANN -> SNN
* Multiple SNN encoding (rate encoding, time-to-first-spike and variants)
* Performance evaluation of ANN vs. SNN
* Various plotting of SNN networks 
* Gain insights into the SNN via TensorBoard

---

### Code layout
#### Spiking Module
Contains the definition of spiking RNN cells, split into rate encoding (rate.py), time-to-first-spike
encoding (ttfs.py) and time-to-first-spike encoding with dynamic threshold (ttfs_dynthresh.py).
Every file contains:
* **IntegratorNeuronCell**: Integrates the current of all timesteps, useful for the last layer
* **LifNeuronCell**: Standard spiking cell used for Dense layers
* **LifNeuronCellConv2D**: Spiking cell implementing convolutions
* **LifNeuronCellMaxPool2D**: Spiking cell implementing max pooling

The rest of the code is split into:
* **ann_training.py**: Loading the datasets and training the ANN
* **conversion.py**: Everything regarding conversion of an ANN to a SNN, including generating the SNN
according to the ANN layout, assigning the weights to the SNN and normalization
* **experiment.py**: The layouts of the different experiments are defined here, including the experiment class
* **optimization.py**: Helpers for hyperparameter optimization
* **plotting.py**: Mostly unused, however there are some functions to calculate total numbers of spikes etc.

---
### Conversion

#### Allowed inputs

The ANNs used as a basis are defined as a sequential keras model, e.g.:

    model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(784, activation='relu'),
                tf.keras.layers.Dense(500, activation='relu'),
                tf.keras.layers.Dense(84, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

### Implementation overview
Usually the activation function has to be ReLU, otherwise conversion won't work.
#### Rate Encoding ("rate")
* tf.keras.layers.Dense: YES
* tf.keras.layers.Conv2D: YES
* tf.keras.layers.Flatten: YES
#### Time To First Spike Encoding ("ttfs")
* tf.keras.layers.Dense: YES
* tf.keras.layers.Conv2D: YES
* tf.keras.layers.Flatten: YES
* tf.keras.layers.MaxPool2D: YES
#### Time To First Spike Encoding with dynamic threshold ("ttfs_dyn_thresh")
* tf.keras.layers.Dense: YES
* tf.keras.layers.Conv2D: YES
* tf.keras.layers.Flatten: YES
* tf.keras.layers.MaxPool2D: YES
#### Time To First Spike Encoding with clamped ReLU ("ttfs_clamped")
* tf.keras.layers.Dense: YES
* tf.keras.layers.Conv2D: YES
* tf.keras.layers.Flatten: YES
* tf.keras.layers.MaxPool2D: YES

---

### Optimization

Under development.

To optimize a variable, define a OptimizeableValue with the name and from which distribution the variable should get its values.
Available distributions to choose from:
* DiscreteInterval(start, stop, step), e.g. DiscreteInterval(1, 5, 1) uniformly draws from the list [1, 2, 3, 4]
* RealInterval(distribution_name, **kwargs), with normal, uniform, poisson and beta distributions. Expects keywords the respective numpy distribution functions would expect.
* DiscreteValues(list), e.g. list=["val1", "val2", "val3"] uniformly draws a value from that list.

An optimizable value is created like the following:

    opt_timesteps = OptimizableValue("timesteps", DiscreteValues([80, 100, 120, 140]))


With the defined optimizable values, create a loop that inserts the value into the experiment:

    while True:
        opt_timesteps_values = opt_timesteps.get()
        exp = Experiment(...
                         timesteps=opt_timesteps_value
                         ...)
    

---
### TensorBoard

Various statistics like:
* average/variance of number of spikes per layer
* average/variance of timestep of spikes per layer
* weight distributions of ANN and SNN
* distribution of spike timesteps for every layer

and more. To start tensorboard execute `tensorboard --logdir ./tblogs/` in the repository directory.

The metrics are created if you call snn_activation_analysis() in your experiment, some of the 
measurements also need other functions called before, e.g. if you want the ANN/SNN
accuracy, ann_accuracy() and snn_accuracy() have to be executed before.

There are no checks yet if all functions were called before and no documentation exists yet.

---

### Theory

* Rueckauer, Bodo, et al. "Conversion of continuous-valued deep networks to efficient event-driven networks for image classification." Frontiers in neuroscience 11 (2017): 682.
* Rueckauer, Bodo, and Shih-Chii Liu. "Conversion of analog to spiking neural networks using sparse temporal coding." 2018 IEEE International Symposium on Circuits and Systems (ISCAS). IEEE, 2018.

---

### Master Thesis
#### Performance of Time to First Spike EncodedSpiking Neural Networks
This repository serves as a codebase for my master's thesis. The thesis is currently not available online, if you
are interested contact me.

#### State of the repository

This repository is mostly in the same state as it was during writing the thesis. It is not optimized for open-source
use by anyone (yet) which you might notice when working with it, as some functions might not be perfectly
documented. Feel free to open issues!

#### Abstract
Analog neural networks (ANNs) are used for a variety of tasks and can even outperform humans in image classification 
tasks. However, inference with ANNs on edge devices is still a challenging task. Spiking neural networks (SNNs) 
tackle this issue by using spikes to communicate between neurons, requiring fewer computations than ANNs. SNNs with 
a rate-based encoding achieve almost similar accuracy results than their ANN counterpart, but neurons with high firing 
rates still produce many avoidable spikes. Time-to-first-spike encoding solves the issue of avoidable spikes by only 
allowing one spike per neuron. This thesis proposes a way to use recurrent neural networks to simulate SNNs and 
analyzes the performance of time-to-first-spike encoded SNNs. The experiments show that with small networks, SNNs 
can slightly improve their accuracy when using time-to-first-spike encoding, and the best LeNet-5 accuracy of an 
SNN is just 0.35 percentage points worse than the ANN. However, when using deeper network architectures, the 
performance drops significantly. With state-of-the-art network architectures, the SNN could not perform better 
than a random agent.

---

### Authors
* Simon Klimaschka