"""
Reference paper: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.40.3735&rep=rep1&type=pdf
https://github.com/handcraftsman/GeneticAlgorithmsWithPython
"""

import utils
import os
from random import random


class Country:
    def __init__(self, t=0, s=0, m=0):
        self.t = t
        self.s = s
        self.m = m


class RateConstants:
    def __init__(self, k_11=0, k_12=0, k_13=0, k_22=0, k_23=0, k_33=0):
        self.k_11 = k_11
        self.k_12 = k_12
        self.k_13 = k_13
        self.k_22 = k_22
        self.k_23 = k_23
        self.k_33 = k_33


class Log:
    def __init__(self):
        self.threat_history = {}


def creator(parameters=None, init_threat=None):
    if init_threat is None:
        init_threat = [0, 0, 0]
    if parameters is None:
        parameters = [0.5, 0.2, 0.6, 0.5, 0.4, 0.6, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2]

    #print("Parameters: ", parameters)
    x = Country(init_threat[0], parameters[0], parameters[1])
    y = Country(init_threat[1], parameters[2], parameters[3])
    z = Country(init_threat[2], parameters[4], parameters[5])
    k = RateConstants(parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11])
    t_history = dict()
    t_history['x'] = []
    t_history['y'] = []
    t_history['z'] = []

    for i in range(50):
        x_t, y_t, z_t = x.t, y.t, z.t
        # print(f"Iteration: {i} - {x.t, y.t, z.t}")
        t_history['x'].append(x_t)
        t_history['y'].append(y_t)
        t_history['z'].append(z_t)

        x.t = x_t + (k.k_11 * (x.s - x_t) + k.k_23 * (y_t + z_t)) * (x.m - x_t)
        y.t = y_t + (k.k_22 * (y.s - y_t) + k.k_13 * (x_t - z_t)) * (y.m - y_t)
        z.t = z_t + (k.k_33 * (z.s - z_t) + k.k_12 * (x_t - y_t)) * (z.m - z_t)

    fitness = x.t - (y.t + z.t)
    #print("Fitness: ", fitness)
    # Plot individual iteration history
    # utils.plot_country_fitnesses(t_history)

    return t_history, fitness


def main(n_generations):
    generations = range(n_generations)
    log = Log()

    for generation in generations:
        parameters = [round(random(), 4), round(random(), 4), round(random(), 4), round(random(), 4),
                      round(random(), 4), round(random(), 4), round(random(), 4), round(random(), 4),
                      round(random(), 4), round(random(), 4), round(random(), 4), round(random(), 4)]
        t_history, fitness = creator(parameters= parameters)

        log.threat_history[generation] = {"parameters": parameters,
                                          "fitness": fitness,
                                          "threats": t_history}

        print("Generation {} has fitness {}:".format(generation, fitness))

    for generation, log in log.threat_history.items():
        if log["fitness"] > 0:
            utils.plot_country_fitnesses(generation, log["threats"], save_figure=True)


if __name__ == '__main__':
    main(n_generations=50)
