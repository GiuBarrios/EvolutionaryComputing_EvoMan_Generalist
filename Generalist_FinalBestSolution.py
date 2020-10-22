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
best_individual_array = []

# insert all 4 IG_BestInd_Generalist Ouputs = [10 x 10] x 2 = 40 lines in total
# identify BEST SOLUTION
data1 = pd.read_csv("General_BestSol_TwoPoint_EG1.csv")
best_ig = max(data1['best_ig'])
print('best_ig:', best_ig)
index_bi = data1[data1['best_ig'] == best_ig].index.values[-1]
best_individual = data1.best_individual[index_bi]
best_individual_array = best_individual

# identify BEST SOLUTION
data2 = pd.read_csv("General_BestSol_TwoPoint_EG2.csv")
best_ig = max(data2['best_ig'])
print('best_ig:', best_ig)
index_bi = data2[data2['best_ig'] == best_ig].index.values[-1]
best_individual = data2.best_individual[index_bi]
best_individual_array = np.vstack([best_individual_array, best_individual])

# identify BEST SOLUTION
data3 = pd.read_csv("General_BestSol_TwoPoint_EG3.csv")
best_ig = max(data3['best_ig'])
print('best_ig:', best_ig)
index_bi = data3[data3['best_ig'] == best_ig].index.values[-1]
best_individual = data3.best_individual[index_bi]
best_individual_array = np.vstack([best_individual_array, best_individual])

# identify BEST SOLUTION
data4 = pd.read_csv("General_BestSol_Uniform_EG1.csv")
best_ig = max(data4['best_ig'])
print('best_ig:', best_ig)
index_bi = data4[data4['best_ig'] == best_ig].index.values[-1]
best_individual = data4.best_individual[index_bi]
best_individual_array = np.vstack([best_individual_array, best_individual])

# identify BEST SOLUTION
data5 = pd.read_csv("General_BestSol_Uniform_EG2.csv")
best_ig = max(data5['best_ig'])
print('best_ig:', best_ig)
index_bi = data5[data5['best_ig'] == best_ig].index.values[-1]
best_individual = data5.best_individual[index_bi]
best_individual_array = np.vstack([best_individual_array, best_individual])

# identify BEST SOLUTION
data6 = pd.read_csv("General_BestSol_Uniform_EG3.csv")
best_ig = max(data6['best_ig'])
print('best_ig:', best_ig)
index_bi = data6[data6['best_ig'] == best_ig].index.values[-1]
best_individual = data6.best_individual[index_bi]
best_individual_array = np.vstack([best_individual_array, best_individual])


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
    return individual_gain, f, p, e


ind = 1

if __name__ == "__main__":
    # load best individual
    print('best_individual_array LENGHT', len(best_individual_array))
    for individual in best_individual_array:

        print('start...')
        individual = individual[0]
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
                random.seed(i ** 2)
                print('# This is run number ', i, 'out of 5#')
                individual_gain, f, p, e = main(individual, [en])

                if i == 1:
                    run = i
                    best_ig_array = individual_gain
                    f_array = f
                    p_array = p
                    e_array = e

                else:
                    run = np.append(run, i)
                    best_ig_array = np.append(best_ig_array, individual_gain)
                    f_array = np.append(f_array, f)
                    p_array = np.append(p_array, p)
                    e_array = np.append(e_array, e)

                i += 1

            if en == 1:
                ind_array = ind
                run_final = run
                enemy_type = en
                best_ig_array_final = best_ig_array.mean(axis=0)
                f_array_final = f_array.mean(axis=0)
                print('f_array_final', f_array_final)
                p_array_final = p_array.mean(axis=0)
                print('p_array', p_array_final)
                e_array_final = e_array.mean(axis=0)
                best_individual_array = individual
            else:
                ind_array = np.append(ind_array, ind)
                run_final = np.append(run_final, run)
                enemy_type = np.append(enemy_type, en)
                best_ig_array_final = np.append(best_ig_array_final, best_ig_array.mean(axis=0))
                f_array_final = np.append(f_array_final, f_array.mean(axis=0))
                p_array_final = np.append(p_array_final, p_array.mean(axis=0))
                e_array_final = np.append(e_array_final, e_array.mean(axis=0))
                best_individual_array = np.vstack([best_individual_array, individual])

        if ind == 1:
            print('ind_array1', ind_array)
            ind_array_final = ind_array
            run_final_final = run_final
            enemy_type_final = enemy_type
            best_ig_array_final_final = best_ig_array_final
            f_array_final_final = f_array_final
            p_array_final_final = p_array_final
            e_array_final_final = e_array_final
            best_individual_array_final = best_individual_array
        else:
            ind_array_final = np.append(ind_array_final, ind_array)
            run_final_final = np.append(run_final_final, run_final)
            enemy_type_final = np.append(enemy_type_final, enemy_type)
            best_ig_array_final_final = np.append(best_ig_array_final_final, best_ig_array_final)
            print('best_ig_array_final_final', best_ig_array_final_final)
            f_array_final_final = np.append(f_array_final_final, f_array_final)
            p_array_final_final = np.append(p_array_final_final, p_array_final)
            e_array_final_final = np.append(e_array_final_final, e_array_final)
            best_individual_array_final = np.vstack([best_individual_array_final, best_individual_array])

        ind += 1


    arr = sparse.coo_matrix(best_individual_array_final)
    # combine into array
    d_final = {'ind_nr': ind_array_final, 'enemy_type': enemy_type_final,
               'best_ig': best_ig_array_final_final, 'best_ig_mean': best_ig_array_final_final,
               'f': f_array_final_final,
               'p': p_array_final_final, 'e': e_array_final_final, 'best_individual': arr.toarray().tolist()}
    df_final = pd.DataFrame(data=d_final)
    print(df_final.head(50))
    df_final.to_csv('FinalBestSolution_p_e.csv')

    df_final_mean = df_final.groupby('ind_nr').mean()
    df_final_mean.to_csv('FinalBestSolution_Means.csv')
