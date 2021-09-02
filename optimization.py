import numpy as np
import tensorflow as tf
from random import randint


# TODO there is a memory leak when running optimizations
# TODO make optimization more automated, return a summary of all runs, maybe include in tensorboard somehow

class OptimizerInput:

    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass


class DiscreteInterval(OptimizerInput):

    def __init__(self, start, stop, step):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        self.iterator = range(self.start, self.stop, self.step).__iter__()
        return self

    def __next__(self):
        return self.iterator.__next__()


class RealInterval(OptimizerInput):

    def __init__(self, distribution_name: str, **kwargs):
        super().__init__()
        allowed_distributions = ["normal", "uniform", "poisson", "beta"]
        if distribution_name not in allowed_distributions:
            raise ValueError("The distribution " + distribution_name + " is not allowed. Please choose from the "
                                                                       "following: " + str(allowed_distributions))
        self.distribution_name = distribution_name
        self.kwargs = kwargs

    def __iter__(self):
        if self.distribution_name == "normal":

            def distribution_function():
                return np.random.normal(loc=self.kwargs.get("loc", 0.0), scale=self.kwargs.get("scale", 1.0))

            self.distribution_function = distribution_function
        elif self.distribution_name == "uniform":
            if "low" not in self.kwargs.keys() or "high" not in self.kwargs.keys():
                raise ValueError("If using uniform distribution, keywords low and high have to be provided.")

            def distribution_function():
                return np.random.uniform(low=self.kwargs.get("low"), high=self.kwargs.get("high"))

            self.distribution_function = distribution_function
        elif self.distribution_name == "poisson":
            if "lam" not in self.kwargs.keys():
                raise ValueError("If using poisson distribution, keyword lam has to be provided.")

            def distribution_function():
                return np.random.poisson(lam=self.kwargs.get("lam"))

            self.distribution_function = distribution_function
        elif self.distribution_name == "beta":
            if "a" not in self.kwargs.keys() or "b" not in self.kwargs.keys():
                raise ValueError("If using beta distribution, keywords a and b have to be provided.")

            def distribution_function():
                return np.random.beta(a=self.kwargs.get("a"), b=self.kwargs.get("b"))

            self.distribution_function = distribution_function

        return self

    def __next__(self):
        return self.distribution_function()


class DiscreteValues(OptimizerInput):
    def __init__(self, values: list or np.array):
        super().__init__()
        self.values = values

    def __iter__(self):
        return self

    def __next__(self):
        return self.values[randint(0, len(self.values) - 1)]


class OptimizableValue:
    def __init__(self, name: str, input_values: OptimizerInput):
        self.name = name
        self.input_values = input_values.__iter__()

    def get(self):
        return self.input_values.__next__()


class Optimization:
    def __init__(self, optimizable_value_list: list):
        self.optimizable_value_list = optimizable_value_list

    @property
    def get(self):
        return
