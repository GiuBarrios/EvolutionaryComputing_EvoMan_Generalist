import pandas as pd
import scipy.sparse as sparse
import sys
import random

sys.path.insert(0, 'evoman')
from environment import Environment
from evoman_framework.demo_controller import player_controller

# imports other libs
import numpy as np
import glob, os

experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# genetic algorithm params
# number of weights for multilayer with 10 hidden neurons
n_vars = (21) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

dom_u = 1
dom_l = -1
npop = 50
gens = 30
mutation = 0.2
last_best = 0
mate = 0.5

#### EVOMAN ###########
individual_gain = []
individual_array = []
run = []
individual_nr = []
enemy_type = []

# load in list on individuals
data = pd.read_csv("General_BestSol_TwoPoint_EG3.csv")
print(data.head())
# get all individuals in df
best_individual_matrix = data.best_individual


def main(individual, en):
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=en,
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")

    f, p, e, t = env.play(pcont=np.array(individual))
    individual_gain = p - e
    individual_array = individual
    return individual_gain, individual_array


if __name__ == "__main__":

    j = 0
    # load all best individuals
    for individual in best_individual_matrix:
        j += 1

        print('start...')
        print('############# THIS IS INDIVIDUAL NUMBER', j, 'out of 10 #############')

        individual = individual.strip("[]")
        individual = individual.split(",")
        individual = [float(i) for i in individual]

        # loop over all enemies
        for en in range(1, 9):
            print('# THIS IS ENEMY ', en, 'out of 8#')

            # run each enemy 5 times
            i = 1
            nr_runs = 6
            while i < nr_runs:
                random.seed(i**2)
                print('# This is run number ', i, 'out of 5#')
                individual_gain, individual_array = main(individual, [en])

                if j == 1 and i == 1:
                    run = i
                    individual_nr = j
                    enemy_type = en
                    best_ig_array = individual_gain
                    best_individual_array = individual_array

                else:
                    run = np.append(run, i)
                    individual_nr = np.append(individual_nr, j)
                    enemy_type = np.append(enemy_type, en)
                    best_ig_array = np.append(best_ig_array, individual_gain)
                    best_individual_array = np.vstack([best_individual_array, individual_array])
                i += 1


    # combine into array
    arr = sparse.coo_matrix(best_individual_array)

    d_final = {'individual_nr': individual_nr, 'run': run, 'enemy_type': enemy_type, 'best_ig': best_ig_array, 'best_individual': arr.toarray().tolist()}
    df_final = pd.DataFrame(data=d_final)
    print(df_final)
    df_final.to_csv('General_IGs_TwoPoint_EG3.csv')
