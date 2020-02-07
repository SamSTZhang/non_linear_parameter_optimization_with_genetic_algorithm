from matplotlib import pyplot as plt
from datetime import datetime
import os


def plot_country_fitnesses(generation, threats, save_figure):
    plt.plot(range(1, len(threats['x']) + 1), threats['x'], color="blue", label="Country X")
    plt.plot(range(1, len(threats['y']) + 1), threats['y'], color="green", label="Country Y")
    plt.plot(range(1, len(threats['z']) + 1), threats['z'], color="red", label="Country Z")
    plt.title("Generation: {} threats for countries X, Y, Z".format(generation))
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
