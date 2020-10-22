# README

This repository contains five Python files:

#### (1) Generalist_FinalBestSolution
Identifying the final best player, performing best against all 8 enemies in terms of avg fitness after the runs.

#### (2) Generalist_IG_BestInd
Comapring the two algorithms in a test playing against all enemies, testing each enemy 5
times with final best solution for each of the 10 independent runs, and presenting
the resulting Gains in box-plots.

#### (3) Generalist_Plots
Line plots (𝑀 and 𝑆𝐸 (for better visualization) of mean and max
fitness of the GAs across 10 independent runs for 3 EGs) and Boxplots (Gains best solutions, across all enemies (5 runs)). 

#### (4) Generalist_TwoPoint
Implementing the Genetic Algorithm with a Two Point Crossover Method.

#### (5) Generalist_Uniform
Implementing the Genetic Algorithm with a Uniform Crossover Method.

# Evolutionary Computing Task 2: Generalist Agent

## 1. Introduction
Neuroevolution describes a process where deep neural networks are optimized by evolutionary algorithms (\textit{EA}s) allowing artificial agents to take automatic decisions in uncertain environments. Video games like the EvoMan Framework that is based on the Mega Man II game, display appropriate testing environments allowing to verify the capability of an algorithm by evolving an autonomous agent. 
In our previous paper we compared two different crossover mechanisms of a Genetic Algorithm (\textit{GA}) and investigated the influence on the performance of a specialist agent by taking possible interaction effects between population size and crossover method into account. Against the findings of John and Spears, our results neither showed that a uniform crossover method outperforms a 2-point crossover method for smaller populations, nor that a 2-point crossover is beneficial for larger population sizes. Instead, rather search space specific patterns were found, suggesting that the optimal choice of a crossover operator is rather problem specific. 

Building up on these findings, the aim of this paper is to compare the performance of 2-point and uniform crossover (see 3.1) based GAs by taking the influence of the complexity of the search space (represented by different enemy groups) on the player performance into account.
Moreover, we aim to evolve a generalist agent capable of defeating all EvoMan bosses with their different behaviours and level designs. 
The performance of the GAs is evaluated against the baseline paper provided by Miras et al.. We evaluate and compare the evolutionary progress during training against a small group of enemies (easy and difficult groups) and examine how well the solution with the highest fitness score performs playing against all 8 enemies of the EvoMan framework based on (1) the number of successfully defeated enemies and (2) the preserved energy level of the player at the end of the game.

## 2. Theory and Hypothesis
Crossover produces new offspring by recombining the information from the parent population and displays the major exploratory mechanism of GAs. When dealing with different classes of problems, crossover operators show various levels of efficiency in solving those problems. Whereas the two-point crossover operator shows a strong bias against keeping together combinations of genes that are located at opposite ends of the representation (positional bias), the uniform operator is more global but has a strong tendency towards transmitting 50% of the genes from each parent (distributional bias).
Therefore, uniform crossover methods disrupt schema with great probability but search larger search spaces. Two-point crossover operator however prevent schema to be disrupted but when the population becomes homogeneous, the search space becomes smaller. Whereas the behaviour of the uniform GA was found to be beneficial for smaller search spaces in our previous research, the two-point crossover GA yielded higher results in more complex search spaces. Based on this, we hypothesize that \textit{(H1) a uniform crossover GA outperforms a two-point crossover based GA for more complex search spaces (represented by a harder enemy group) }and \textit{(H2) a two-point crossover based GA outperforms a uniform crossover based GA in smaller problem spaces (represented by a less complex enemy group)}. 

## 3. Method
Both implemented algorithms were based on a standard GA representing a simple way to perform Neuroevolution by defining a fixed topology and representing the evolved solutions as a vector of weights. The reproduction crossover operators displayed the only difference between the GAs and are further described in section 3.1. The mutation and crossover parameters were fine-tuned by evaluating the average fitness of the agents for 10 test iterations over all possible configurations and then kept constant during training. The population size as well as the number of generations per evolution cycle was selected based on the estimated run times and the limit in computational resources. For the implementation of both GAs the DEAP Framework was used.

The algorithms comprehended three main steps: (1) Both GAs were initialized with a random population of $n = 50$ individuals sampled from a uniform distribution between -1 and 1. For each generation new offspring was generated by applying a crossover operator on pairs of solutions. Parents were selected in an unbiased approach via a uniform random distribution where every individual had the same probability to be selected.
(2) To further increase diversity, the offspring was randomly mutated with a Gaussian operator of $mu = 0$ and $std = 1$. The chance of mutation was kept constant across all generations at $indpb= 0.05$.
(3) The next generation of individuals was selected among the current population and the offspring and was implemented as a k-tournament selection, where n individuals are randomly sampled and the fittest are chosen to be part of the next generation. The tournament size was set to $tournsize = 3$. Evolution was terminated after 30 generations.

### 3.1 Crossover Methods
As described, the implemented GAs differed in their crossover methods: uniform and 2-point crossover. In the \textit{uniform crossover }method each bit of the chromosome from either parent was chosen with equal probability. Attributes were then swapped with an independent probability of $p = 5%$. On the other hand, the \textit{2-point crossover} method picked two crossover points randomly from the parent chromosomes. The parts in between the two points were then swapped between the parent individuals. 

### 3.2 Fitness Function
For the fitness function the default fitness function provided by the EvoMan Framework was used: $fitness= 0.9(100-enemylife)+0.1playerlife-\log(t)$,
where $e$ and $p$ display the energy level of the player and the enemy in the range of [0, 100] and $t$ the total number of time steps the game lasted. To discourage simple avoidance tactics of the player, the weight of the player's life is lower compared to the enemy. Furthermore, the number of time steps is subtracted to encourage faster wins. 

### 3.3 Controller
For the controller representation a neural network with 10 hidden neurons was used. The network received information from 20 sensors as inputs (16 for distances from the player to the fired projectiles, 2 for the directions the agents were facing, 2 for distances from the player to the enemy). Chromosomes were encoded as real-value weight vectors. Inputs were normalized using min-max scaling and a sigmoid function was selected as activation function.

#### Experimental Set Up
The algorithms were trained against three different enemy groups (EG) based on a strategic selection from the 8 enemies provided by the EvoMan framework differing in their level of difficulty. EG1 consists of simple enemies (FalshMan, AirMan and WoodMan). To succeed against this EG, the agent only has to jump at the same spot while shooting. 
In contrast, the hard enemy group (EG2) is composed of the most difficult enemies (HeatMan, CrashMan and BubbleMan) requiring the player to develop a more advanced policy (special care when jumping due to environmental conditions). A third mixed EG was set up as a control group, in which the player was trained against a combination of the easiest and most difficult opponents ($EG1 + EG2$). 
The individual evolution simulation mode of the framework was used in which the player fights against the selected static enemies. All other environmental parameters were kept to their default values. Due to the stochastic nature of GAs, each algorithm was assessed on 10 independent training runs against each enemy group. The average and maximum fitness, their standard deviations, the player and enemy energy as well as the solution with the maximum fitness were recorded across generations for each run. For the best solution, the total gain was calculated by subtracting the enemy energy from the player energy summed up over all enemies across 5 independent runs: $gain = \sum_{i=1}^{n} (p\_i - e\_i)$. For the comparison of the performance of the GAs, non parametric two-sided Mann-Whitney tests were performed with $\alpha$= .05 across all experiments.


## 4. Results
#### Fitness across 30 generations:
Figure \ref{fig:fitness} shows the time course of the average and maximum fitness with their standard deviations across 30 generations against the 3 different enemy groups. In line with this, Figure \ref{fig:fitnesstable} describes the average maximum and mean fitness across all generations and shows the statistical results of the Mann-Whitney tests. \textit{Comparison between GAs:} in general, the average mean and maximal fitness scores of both GAs converged in a similar trend towards a (local) optimum after 20-30 generations. However, the 2-point GA graph reached a higher fitness level across all enemy groups and numerical results showed significantly higher mean and max fitness values for the easy EG1 and mixed EG3 with p < 0.05 (see Table). In contrast, for EG2 no significant differences between the two GAs were found, however it sparks out that the average maximum fitness of the 2-point GA converges much faster (after ~ generation 5) compared to the uniform GA (after ~ generation 25).This could be interpreted as an advantageous effect of successful exploitation, resulting from the less exploratory behavior of the 2-point GA.

#### Comparison between the EGs:
when comparing the average and maximum fitness for one GA across the different enemy groups, highest fitness results were achieved for the easiest enemy group (2-point $mean\_max\_fit$ = 62.66; uniform $mean\_max\_fit$ = 58,27) followed by the hard enemy group (2-point $mean\_max\_fit$ = 53.11; uniform $mean\_max\_fit$ = 52.50). In the mixed enemy group, the player showed the lowest average max fitness with 2-point GA $mean\_max\_fit$ = 49.90 and Uniform GA $mean\_max\_fit$ = 46.75.

#### Gains of 6 best generalist players:
Figure \ref{fig:gains} shows the gain-box plots for the solutions that achieved maximum fitness in the ten independent training runs, measured in five independent test runs each. \textit{Comparison between GAs}: While EG3, uniform with $gain\_mean = -18.01$ showed the best overall gain, the difference to 2-point EG3 ($gain\_mean = -18.99$) was not significant ($W = 65555.50, p = 0.350$). The only significant ($W = 61677.50, p = 0.047$) effect was found within the easy EG2, where the 2-point GA ($gain\_mean = -20.64$) significantly outperformed uniform crossover ($gain\_mean = -27.98$).\textit{ Comparison between the EGs:} Significant differences between the EGs were only detected for the uniform crossover, easy EG2 - with overrandom differences to the higher difficulty levels EG1 ($W = 60406.50, p = 0.014$) and especially to EG3 ($W = 57068.50, p < 0.001$). No significant differences in performance were found between the three EGs in the two-point crossover, indicating a robust high level of performance.


#### Final best generalist player:
So far we have shown how the agents that were trained against one out of the tree enemy groups performed against these specific EGs. 
In this section, we closer evaluate the player that achieved the highest gain performance when being tested against all 8 opponents provided by the EvoMan Framework.
Table \ref{tab:bestplayer} shows the average energy points of the agents for our best solution, which was achieved with the 2-point crossover GA and EG1, over five runs. This best individual was determined by running the 6 best agents (with the best IGs) from the 6 different experimental groups (2 GAs, 3 EGs) against all 8 enemies (5 repetitions) and selecting the agent with the highest fitness average over all enemies ($f\_mean = 64.5$). This individual showed a high winning rate of 5 out of 8 successfully defeated enemies ($p\_win = 63\%$): Trained on the hard EG1, the player was able to win against all complex enemy types 5 - 8 and even generalizes enough to beat the easy opponent 2.

Compared to the results of the baseline paper, our best-developed agent performs slightly worse in terms of the number of defeated enemies, but better in terms of average gain. The agent of GA10 developed by de Araujo et al, trained on the same enemies (4,6,7), achieved a lower mean gain (baseline: $g=-142$ vs. 2-point, EG1: $g=-79$), but was able to beat two more enemies.
This difference could be explained with the fact that a higher population size was chosen $Pop\_size = 100$ that allowed a higher amount of training iterations and thus a faster convergence of performance.


## 5. Discussion
In this paper we investigated the hypothesis that \textit{(H1) a uniform crossover GA outperforms a two-point crossover based GA for more complex search spaces }and \textit{(H2) a two-point crossover based GA outperforms a uniform crossover based GA in less complex problem spaces}. Moreover, we were interested in developing a generalist agent that is capable of defeating all EvoMan enemies provided by the framework. 
The experiments showed that the 2-point GA achieved a higher performance (higher fitness values during training) when compared to the uniform GA for the easy enemy group. Based on these results we could infer that a two-point crossover based GA indeed outperforms a uniform GA in less complex problem spaces. These findings show that the higher positional bias, resulting in a more exploitative behaviour, was beneficial for our specific problem. In contrast, advantages of the more exploratory behaviour of the uniform crossover could not unfold in any of the search spaces of our problem, but might have disrupted potentially good solutions instead.
With regard to the final best generalist player, the 2-point crossover GA that was trained against the hard EG3, achieved the highest performance (highest number of defeated enemies and highest mean fitness) in the test runs. This shows that a training on difficult enemies is beneficial for developing a behaviour that allows to generalize to new problems and successfully defeat unknown enemies. The experiment however also showed that a general strategy that is capable to defeat ALL enemies proves to remain a challenge that is not yet achieved by the tested algorithms.


## 6. Conclusion
We implemented two GAs that varied in their crossover methods (uniform vs. 2-point) in order to develop a generalist agent, capable of successfully competing against groups of enemies that differ in their problem complexity. We showed that a 2-point crossover GA is beneficial to defeat less complex enemy groups when compared to a uniform GA. Moreover, our 2-point GA was able to develop a general agent reaching a higher mean gain but defeating a lower number of enemies when compared to the baseline algorithm GA10 by Araujo et al.

With regard to possible limitations of the study the fact that the definition of the complexity of the search spaces was  based on the heuristics provided by the baseline paper has to be mentioned.
To derive more generalizable conclusions regarding the influence of different crossover methods and search spaces on the performance of GAs, future studies should therefore quantify the complexity of the investigated problems with explicit measurements (e.g correlation analysis). Moreover, not just additional crossover methods (e.g. statistical based methods) but also more difficult test problems should be tested. Also self-adaptive approaches in which the EA selects its own search operators could be considered.

