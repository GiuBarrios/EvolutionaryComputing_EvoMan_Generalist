# import from deap
import array
import random
import numpy
import pandas as pd

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# imports framework
import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from evoman_framework.demo_controller import player_controller

# imports other libs
import numpy as np
import os

experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
# number of weights for multilayer with 10 hidden neurons
n_vars = (21) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

dom_u = 1
dom_l = -1
npop = 50
gens = 30
mutation = 0.05
last_best = 0
mate = 0.5

######### 1) CHOSE ENEMY GROUP HERE ##########
# hard (EG1)
# enemies_group = [4, 6, 7]

# easy (EG2)
# enemies_group = [1, 2, 3]

# diverse / mixed (EG3)
enemies_group = [1, 2, 3, 4, 6, 7]

#### DEAP ###########
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
# weights uniform, no integer
toolbox.register("attribute", random.uniform, -1, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")

    # tests saved demo solutions for each enemy
    f_array = []
    individual_gain_array = []

    for en in enemies_group:
        # Update the enemy
        env.update_parameter('enemies', [en])
        f, p, e, t = env.play(pcont=np.array(individual))
        f_array.append(f)
        individual_gain_array.append(p - e)

    individual_array.append(individual)
    individual_gain.append(np.mean(individual_gain_array))
    f_mean = np.mean(f_array)
    return f_mean,


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.05)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=mutation)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=mate, mutpb=mutation, ngen=gens,
                                   stats=stats, halloffame=hof, verbose=True)

    # logs
    data = [[i for i in item.values()] for item in log]
    df = pd.DataFrame(data, columns=log.header)
    print(df)

    avg_avg = df['avg']
    std_avg = df['std']
    avg_max = df['max']
    std_max = df['max']

    # individual gain
    best_ig = max(individual_gain)
    print('best_ig:', best_ig)
    best_individual = individual_array[individual_gain.index(max(individual_gain))]
    print('best_individual:', best_individual)

    return avg_avg, std_avg, avg_max, std_max, best_ig, best_individual


if __name__ == "__main__":
    avg_avg = []
    print(type(avg_avg))
    std_avg = []
    avg_max = []
    std_max = []
    best_ig_array = []
    best_individual_array = []
    run = []
    print(type(best_individual_array))

    i = 1
    nr_runs = 11

    while i < nr_runs:
        # initialize array of seeds randomly
        # seeds = np.random.randint(1, 101, 10)
        # for s in seeds:
        random.seed(i**2)
        # clean after every run
        individual_gain = []
        individual_array = []

        print('############# THIS IS RUN NUMBER', i, ' #############')

        avg_avg_new, std_avg_new, avg_max_new, std_max_new, best_ig_new, best_individual_new = main()

        if i == 1:
            avg_avg = avg_avg_new
            std_avg = std_avg_new
            avg_max = avg_max_new
            std_max = std_max_new
            run = i
            best_ig_array = best_ig_new
            best_individual_array = best_individual_new

        else:
            run = np.append(run, i)
            avg_avg = np.vstack([avg_avg, avg_avg_new])
            std_avg = np.vstack([std_avg, std_avg_new])
            avg_max = np.vstack([avg_max, avg_max_new])
            std_max = np.vstack([std_max, std_max_new])

            best_ig_array = np.append(best_ig_array, best_ig_new)
            best_individual_array = np.vstack([best_individual_array, best_individual_new])
            print('best_ig_array: ', best_ig_array)
            print('best_individual_array: ', best_individual_array)
        i += 1

    print('avg_avg:', avg_avg)
    print('std_avg:', std_avg)
    print('avg_max:', avg_max)
    print('std_max:', std_max)

    d_logs = {'avg_avg': avg_avg.mean(axis=0), 'std_avg': std_avg.mean(axis=0), 'avg_max': avg_max.mean(axis=0),
              'std_max': std_max.std(axis=0)}
    df_logs = pd.DataFrame(data=d_logs)
    print(df_logs)
    df_logs.to_csv('General_Logbook_Uniform_EG3.csv')

    # final best solutions for each of 10 individual runs
    import scipy.sparse as sparse

    arr = sparse.coo_matrix(best_individual_array)

    d_final = {'run': run, 'best_ig': best_ig_array, 'best_individual': arr.toarray().tolist()}
    df_final = pd.DataFrame(data=d_final)
    print(df_final)

    ######### 2) ADAPT NAMING TO ENEMY GROUP ##########
    df_final.to_csv('General_BestSol_Uniform_EG3.csv')
