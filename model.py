import utils
import numpy as np
import pandas as pd
import logging
import random
import copy

LIFECYCLE_LENGTH = 50


class Log:
    def __init__(self):
        self.threat_history = {}


def generation_lifecycle(chromosomes, init_threat):
    """
    Takes a population through it's lifecycle following Richardson's equation
    :param chromosomes: initial population parameters
    :param init_threat: initial Threat values
    :return: Threat history values and calculated fitness
    """
    if not init_threat:
        chromosomes['x_t'], chromosomes['y_t'], chromosomes['z_t'] = 0, 0, 0

    t_history_list = []

    for _ in range(LIFECYCLE_LENGTH):
        chromosomes['x_t'] = chromosomes['x_t'] + (chromosomes['k11'] * (chromosomes['x_s'] - chromosomes['x_t'])
                                                   + chromosomes['k23']*(chromosomes['y_t'] + chromosomes['z_t'])
                                                   )*(chromosomes['x_m'] - chromosomes['x_t'])

        chromosomes['y_t'] = chromosomes['y_t'] + (chromosomes['k22'] * (chromosomes['y_s'] - chromosomes['y_t'])
                                                   + chromosomes['k13']*(chromosomes['x_t'] - chromosomes['z_t'])
                                                   )*(chromosomes['y_m'] - chromosomes['y_t'])
        chromosomes['z_t'] = chromosomes['z_t'] + (chromosomes['k33'] * (chromosomes['z_s'] - chromosomes['z_t'])
                                                   + chromosomes['k12']*(chromosomes['x_t'] - chromosomes['y_t'])
                                                   )*(chromosomes['z_m'] - chromosomes['z_t'])

        # alliance shift check
        for i in range(chromosomes.shape[0]):
            if chromosomes.at[i, 'x_t'] < chromosomes.at[i, 'y_t']:
                chromosomes.at[i, 'x_t'], chromosomes.at[i, 'y_t'] = \
                    chromosomes.at[i, 'y_t'], chromosomes.at[i, 'x_t']
                chromosomes.at[i, 'x_s'], chromosomes.at[i, 'y_s'] = \
                    chromosomes.at[i, 'y_s'], chromosomes.at[i, 'x_s']
                chromosomes.at[i, 'x_m'], chromosomes.at[i, 'y_m'] = \
                    chromosomes.at[i, 'y_m'], chromosomes.at[i, 'x_m']
                chromosomes.at[i, 'k11'], chromosomes.at[i, 'k22'] = \
                    chromosomes.at[i, 'k22'], chromosomes.at[i, 'k11']
                chromosomes.at[i, 'k23'], chromosomes.at[i, 'k13'] = \
                    chromosomes.at[i, 'k13'], chromosomes.at[i, 'k23']

            if chromosomes.at[i, 'x_t'] < chromosomes.at[i, 'z_t']:
                chromosomes.at[i, 'x_t'], chromosomes.at[i, 'z_t'] = \
                    chromosomes.at[i, 'z_t'], chromosomes.at[i, 'x_t']
                chromosomes.at[i, 'x_s'], chromosomes.at[i, 'z_s'] = \
                    chromosomes.at[i, 'z_s'], chromosomes.at[i, 'x_s']
                chromosomes.at[i, 'x_m'], chromosomes.at[i, 'z_m'] = \
                    chromosomes.at[i, 'z_m'], chromosomes.at[i, 'x_m']
                chromosomes.at[i, 'k11'], chromosomes.at[i, 'k33'] = \
                    chromosomes.at[i, 'k33'], chromosomes.at[i, 'k11']
                chromosomes.at[i, 'k23'], chromosomes.at[i, 'k12'] = \
                    chromosomes.at[i, 'k12'], chromosomes.at[i, 'k23']

        t_history_list.append(list(chromosomes['x_t']))
        t_history_list.append(list(chromosomes['y_t']))
        t_history_list.append(list(chromosomes['z_t']))

    fitnesses = abs(chromosomes['x_t'] - (chromosomes['y_t'] + chromosomes['z_t']))

    timeline_range = np.array(list(range(1,LIFECYCLE_LENGTH+1))*3)
    timeline_range.sort()

    countries_range = np.array(['x', 'y', 'z']*LIFECYCLE_LENGTH)
    t_history_list = np.array(t_history_list)
    t_history = pd.DataFrame(data=t_history_list.T, columns=pd.MultiIndex.from_tuples(zip(timeline_range, countries_range)))

    return t_history, fitnesses


def selection(chromosomes, fitnesses, method):
    """
    Implements selection amongst generations
    :param chromosomes: population
    :param fitnesses: fitness values
    :param method: Choice between different selection strategies
    :return: new population
    """

    if method is None:
        method = 'fitness_proportionate'

    dict_fitnesses = fitnesses.to_dict()
    n_population = chromosomes.shape[0]
    selected_individuals = pd.DataFrame(data=None, columns=chromosomes.columns)

    # Or use df.sample(weights='a')
    if method == 'fitness_proportionate':
        for _ in range(n_population):
            loc = utils.weighted_random_choice(dict_fitnesses)
            selected_individuals = selected_individuals.append(chromosomes.iloc[loc])
    else:
        selected_individuals = chromosomes.copy()
    
    return selected_individuals.reset_index(drop=True)


def mutate(chromosomes, prob):
    """
    Implements mutation amongst generations
    :param chromosomes: population
    :param prob: mutation probability p_m
    :return: mutated population
    """
    idx_to_mutate = list(chromosomes.sample(frac=prob, replace=True, random_state=1).index)

    columns = list(chromosomes.columns)
    len_cols = chromosomes.shape[1]-1

    for id in idx_to_mutate:
        chromosomes.at[id, columns[random.randint(1, len_cols)]] = random.random()

    return chromosomes


def crossover(chromosomes, prob):
    """
    Implements single point crossover
    :param chromosomes: chromosomes
    :param prob: probability of crossover
    :return: new population
    """
    idx_to_crossover = list(chromosomes.sample(frac=prob, replace=True).index)
    len_cols = chromosomes.shape[1] - 1

    for i in range(0, len(idx_to_crossover), 2):
        crossover_point = random.randint(1, len_cols)
        temp_i = copy.deepcopy(chromosomes.iloc[idx_to_crossover[i]][:crossover_point])
        temp_i1 = copy.deepcopy(chromosomes.iloc[idx_to_crossover[i+1]][:crossover_point])

        chromosomes.iloc[idx_to_crossover[i]][:crossover_point] = temp_i1
        chromosomes.iloc[idx_to_crossover[i+1]][:crossover_point] = temp_i

    return chromosomes


def main(n_generations, n_population):
    generations = range(n_generations)
    generation = 1
    curr_fitnesses_sum = 100
    prev_fitnesses_sum = 0
    log = Log()
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M %p',
                        level=logging.DEBUG)

    # Initialize population
    chromosomes = pd.DataFrame(np.random.random(size=(n_population, 12)),
                               columns=['x_s', 'x_m', 'y_s', 'y_m', 'z_s', 'z_m',
                                        'k11', 'k22', 'k33', 'k12', 'k13', 'k23'])
    fitnesses = pd.Series(np.linspace(3, 33, 3))

    while abs(curr_fitnesses_sum - prev_fitnesses_sum) > 0.01:

        prev_fitnesses_sum = sum(fitnesses)

        t_history, fitnesses = generation_lifecycle(chromosomes=chromosomes, init_threat=False)

        new_gen_chromosomes = selection(chromosomes=chromosomes, fitnesses=fitnesses, method="fitness_proportionate")

        new_gen_chromosomes = mutate(chromosomes=new_gen_chromosomes, prob=0.6)

        new_gen_chromosomes = crossover(chromosomes=new_gen_chromosomes, prob=0.4)

        chromosomes = new_gen_chromosomes
        curr_fitnesses_sum = sum(fitnesses)
        logging.debug("Generation %d: #Uniques %d, sumdiff: %.6f", generation, len(fitnesses.unique()),
                      abs(curr_fitnesses_sum - prev_fitnesses_sum))
        generation += 1

    # Plotting
    utils.plot_country_threats_over_time(generation, t_history, fitnesses, save_figure=True)
    utils.plot_fitnesses(fitnesses, save_figure=True)


if __name__ == '__main__':
    main(n_generations=50, n_population=100)
