from matplotlib import pyplot as plt
from datetime import datetime
import os
import numpy.random as npr


def plot_country_fitness(generation, threats, save_figure):
    plt.plot(range(1, len(threats['x']) + 1), threats['x'], color="blue", label="Country X")
    plt.plot(range(1, len(threats['y']) + 1), threats['y'], color="green", label="Country Y")
    plt.plot(range(1, len(threats['z']) + 1), threats['z'], color="red", label="Country Z")
    plt.title("Individual: {} threats for countries X, Y, Z".format(generation))
    plt.xlabel("time")
    plt.ylabel("threat potential")
    plt.legend(loc="best")
    if save_figure:
        folder = os.path.join("./plots", datetime.now().strftime('%Y-%m-%d_%H-%M'))
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig("{}/generation_{}_threats_over_time.png".format(folder, generation))
        plt.close()
    else:
        plt.show()


def plot_country_threats_over_time(generation, threats, fitnesses, save_figure):

    n_population = threats.shape[0]

    for individual in range(n_population):
        i_threats = list(threats.iloc[individual])
        x_threats, y_threats, z_threats = [], [], []
        fitness = fitnesses.iloc[individual]

        for i in range(0, len(i_threats), 3):
            x_threats.append(i_threats[i])
            y_threats.append(i_threats[i+1])
            z_threats.append(i_threats[i+2])

        plt.plot(range(1, len(x_threats) + 1), x_threats, color="blue", label="Country X")
        plt.plot(range(1, len(y_threats) + 1), y_threats, color="green", label="Country Y")
        plt.plot(range(1, len(z_threats) + 1), z_threats, color="red", label="Country Z")
        plt.title("Generation : {} Individual: {} Fitness: {}".format(generation, individual, fitness))
        plt.xlabel("time")
        plt.ylabel("threat potential")
        plt.legend(loc="best")
        if save_figure:
            folder = os.path.join("./plots", datetime.now().strftime('%Y-%m-%d_%H-%M') + "_Gen_" + str(generation))
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig("{}/individual_{}_fitness_{}_threats_over_time.png".format(folder, individual, fitness))
            plt.close()
        else:
            plt.show()


def plot_fitnesses(fitnesses, save_figure):
    l_fitness = list(fitnesses)

    plt.plot(range(0, len(l_fitness)), l_fitness, color="green")
    plt.title("Fitnesses for all individual")
    plt.xlabel("individuals")
    plt.ylabel("fitness values")

    if save_figure:
        folder = os.path.join("./plots", datetime.now().strftime('%Y-%m-%d_%H-%M') + "_Fitness")
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig("{}/all_fitnesses.png".format(folder))
        plt.close()
    else:
        plt.show()


def weighted_random_choice(choices):
    population = choices.values()
    max = sum([1/x for x in population if str(x) != 'nan'])
    selection_probs = [(1/(c*max)) for c in population if str(c) != 'nan']

    return npr.choice(len(population), p=selection_probs)