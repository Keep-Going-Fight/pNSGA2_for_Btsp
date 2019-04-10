
import logging
import pickle
import itertools
import timeit
import pandas as pd
import math

from matplotlib import pyplot as plt
import numpy as np

from ea_population import TSPPopulation
from utils import Loader  
from ea_problem import TSPProblem


"""
TSPPopulation   -----  a population of  potential solution
Loader          -----  loader datasets
TSPProblem      -----  solve TSP by NSGA 2
"""


class EARunner:
    def __init__(self,run):
        logging.basicConfig(level=logging.DEBUG)
        self.loader = Loader(False)
        #self.run_problem()
        #print type(self.run)
        self.run_true_front(run)
        #self.run_problem(run)
        
        

    def run_problem(self,run):
        distances, costs = self.loader.load_dataset_a()
        problem = TSPProblem(distances, costs,run)
        save_path = str(problem.population_size) + ' ' \
                    + str(problem.generation_limit) + ' ' \
                    + str(problem.crossover_rate) + ' ' \
                    + str(problem.mutation_rate) + ' report'
        self.run(problem, plot=True)
        #self.run(problem, plot=True, save_path="../results/" + save_path)

    def run_true_front(self,run):
        distances, costs = self.loader.load_dataset_b()
        problem = TSPProblem(distances, costs,run)
        #print len(costs)
        self.run(problem, plot=True)

    def load_results(self):
        paths = ["../results/50 4000 0.7 0.05 report-0.pickle",
                 "../results/100 2000 0.8 0.01 report-0.pickle",
                 "../results/200 1000 0.8 0.05 report-1.pickle"]
        self.load_results_stats(paths)
        self.load_results_plot(paths)

    @staticmethod
    def run(problem, true_front=None, plot=True, save_path=None):
        """
        :param problem:
        :param plot:
        :param true_front: actual optimal front (for comparison with discovered/calculated front)
        :param save_path: Save the first front of the final population to file with the given path
        :return:
        """
        # Generate the initial population
        population = TSPPopulation(problem)
        logging.info("Generations: %s, Pop. size: %s, Cross. rate: %s, Mut. rate: %s",
                     problem.generation_limit,
                     problem.population_size,
                     problem.crossover_rate, problem.mutation_rate)
        fronts = []

        def main_loop():
            while population.generation < problem.generation_limit:
                population.generation += 1
                population.evaluate_fitnesses()  # Calculate total cost and total distance for each route/individual
                population.select_adults()
                population.select_parents()
                population.reproduce()
                if population.generation % (problem.generation_limit / 5) == 0:
                    logging.info("\t\t Generation %s/%s", population.generation, problem.generation_limit)
                    fronts.append(population.get_front(0))

        logging.info("\tExecution time: %s", timeit.timeit(main_loop, number=1))
        logging.info("\t(Min/Max) Distance: %s/%s; Cost: %s/%s",
                     TSPPopulation.min_fitness(population.adults, 0).fitnesses[0],
                     TSPPopulation.max_fitness(population.adults, 0).fitnesses[0],
                     TSPPopulation.min_fitness(population.adults, 1).fitnesses[1],
                     TSPPopulation.max_fitness(population.adults, 1).fitnesses[1])

        if save_path:
            with open(save_path + "-" + str(np.random.randint(10)) + '.pickle', 'wb') as f:
                pickle.dump(population.get_front(0), f)
        if plot:
            Pareto_adult  = [(individule.fitnesses[0],individule.fitnesses[1])  for individule in population.adults]
            Pareto_adult = sorted(list(set(Pareto_adult)), key=lambda adult: adult[0])
            
            Pareto_get_front = [(individule.fitnesses[0],individule.fitnesses[1])  for individule in population.get_front(0)]
            Pareto_get_front = sorted(list(set(Pareto_get_front)), key=lambda front: front[0])
            
            Pareto_true_front = ((individule.fitnesses[0],individule.fitnesses[1])  for individule in fronts[0])
            Pareto_true_front = sorted(list(set(Pareto_true_front)), key=lambda front: front[0])
            
            with open(r'Pareto_adult_Fitnesses.txt','a') as f:
                for result_1 in Pareto_adult:
                    f.write(str(result_1[0]) + " " + str(result_1[1]) + "\n")
            with open(r'Pareto_get_front.txt','w') as f:
                for result_2 in Pareto_get_front:
                    f.write(str(result_2[0]) + " " + str(result_2[1]) + "\n")
            with open(r'Pareto_get_front_1.txt','a') as f:
                for result_4 in Pareto_get_front:
                    f.write(str(result_4[0]) + " " + str(result_4[1]) + "\n")
            with open(r'Pareto_true_front.txt','a') as f:
                for result_3 in Pareto_true_front:
                    f.write(str(result_3[0]) + " " + str(result_3[1]) + "\n")
                    
            EARunner.plot([population.adults], save_path=save_path)
            EARunner.plot([population.get_front(0)],
                          name='Fitnesses, final Pareto-front',
                          save_path=save_path)
          #  EARunner.plot(fronts, true_front=true_front, dash=True,
                         # name='Fitnesses, final Pareto-front per 20% progress', save_path=save_path)
            plt.show()

    @staticmethod
    def plot(pools, true_front=None, dash=False, name='Fitnesses', save_path=None):
        """
        :param true_front:
        :param pools: NOT instance of TSPPopulations, but a list of lists of individuals (lists of population.adults)
        :param dash: dash lines between each individual in each pool
        :param name: Plot legend
        :param save_path:
        :return:
        """
        marker = itertools.cycle(('o', ',', '+', '.', '*'))  #line marker
        color = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))
        """
        if dash:
            linestyle = "--"
            for pool_i in range(len(pools)):
                pools[pool_i] = sorted(pools[pool_i],
                                       key=lambda ind: ind.fitnesses[0])
        else:
            linestyle = ""
        """
        linestyle = ""
        plt.figure()
        plt.title(name)
        
        for i, pool in enumerate(pools):
            c = next(color)
            plt.plot([individual.fitnesses[0] for individual in pool],
                     [individual.fitnesses[1] for individual in pool],
                     marker=next(marker), linestyle=linestyle, color=c,
                     label=str((i + 1) * 20) + "%-" + str(len(pool))
                           + "sols-" + str(TSPPopulation.n_unique(pool)) + "uniq")
            min_dist = TSPPopulation.min_fitness(pool, 0).fitnesses
            max_dist = TSPPopulation.max_fitness(pool, 0).fitnesses
            min_cost = TSPPopulation.min_fitness(pool, 1).fitnesses
            max_cost = TSPPopulation.max_fitness(pool, 1).fitnesses
            if not dash:
                c = 'r'
            plt.plot([min_dist[0]], [min_dist[1]], marker='D', color=c)
            plt.plot([max_dist[0]], [max_dist[1]], marker='D', color=c)
            plt.plot([min_cost[0]], [min_cost[1]], marker='D', color=c)
            plt.plot([max_cost[0]], [max_cost[1]], marker='D', color=c)
        if true_front is not None:
            plt.plot([i[0] for i in true_front], [i[1] for i in true_front],
                     linestyle="--", label="True front")
            if dash:
             plt.legend(loc="best")
        plt.xlabel("Distance")
        plt.xticks(np.arange(30000, 120001, 10000))
        plt.ylabel("Cost")
        plt.yticks(np.arange(300, 1401, 100))
        if save_path:
            plt.savefig(save_path + "-" + str(np.random.randint(10)) + ".png")

    @staticmethod
    def load_results_plot(paths):
        populations = []
        for path in paths:
            with open(path, 'rb') as f:
                population = pickle.load(f)
                populations.append(population)
        #print populations
        EARunner.plot(populations, dash=True,
                      name="Final pareto fronts, 3 configurations")
        plt.show()

    @staticmethod
    def load_results_stats(paths):
        for path in paths:
            with open(path, 'rb') as f:
                population = pickle.load(f)
                logging.info("\t(Min/Max) Distance: %s/%s; Cost: %s/%s",
                             TSPPopulation.min_fitness(population, 0).fitnesses[0],
                             TSPPopulation.max_fitness(population, 0).fitnesses[0],
                             TSPPopulation.min_fitness(population, 1).fitnesses[1],
                             TSPPopulation.max_fitness(population, 1).fitnesses[1])

"""
    every module has itself name. when the module is inputed in the first,the main module is running.
    if __name == "__main__"  then running it single.
"""
if __name__ == "__main__":
   for run in range(20):
       runner = EARunner(run)
       temp_sum = []
       temp = pd.read_csv('Pareto_get_front.txt',sep=' ',header=None)
       current_front = temp.values
       front_count = len(current_front)
       #comparision
       #data1 = pd.read_csv('./../datasets/best.euclidAB100.tsp',sep='	',usecols=[0,1],header=None)
       #data1 = pd.read_csv('./../datasets/best.euclidEF100.tsp',sep='	',usecols=[0,1],header=None)
       #data1 = pd.read_csv('./../datasets/best.euclidCD100.tsp',sep='	',usecols=[0,1],header=None)
       #data1 = pd.read_csv('./../datasets/best.randomAB100.tsp',sep='	',usecols=[0,1],header=None)
       #data1 = pd.read_csv('./../datasets/best.randomCD100.tsp',sep='	',usecols=[0,1],header=None)

       #data1 = pd.read_csv('./../datasets/best.3best.100.1',sep=' ',usecols=[0,1],header=None)
       #data1 = pd.read_csv('./../datasets/best.3best.150.1',sep=' ',usecols=[0,1],header=None)
       #data1 = pd.read_csv('./../datasets/best.3best.200.1',sep=' ',usecols=[0,1],header=None)
       
       #data1 = pd.read_csv('./../datasets/best.3AC.100.1',sep=' ',usecols=[0,1],header=None)
       #data1 = pd.read_csv('./../datasets/best.3BC.100.1',sep=' ',usecols=[0,1],header=None)
       data1 = pd.read_csv('./../datasets/best.3CD.100.1',sep=' ',usecols=[0,1],header=None)
       #data1 = pd.read_csv('./../datasets/best.3DE.100.1',sep=' ',usecols=[0,1],header=None)

       comprision = data1.values
       l_front = len(comprision)
       sum_Mdistance = 0
       for j in range(front_count):
           min_distance = math.sqrt(math.pow((current_front[j][0]-comprision[0][0]),2)+math.pow((current_front[j][1]-comprision[0][1]),2))
           for k in range(l_front):
               if min_distance > math.sqrt(math.pow((current_front[j][0]-comprision[k][0]),2)+math.pow((current_front[j][1]-comprision[k][1]),2)):
                   nce = math.sqrt(pow((current_front[j][0]-comprision[k][0]),2)+pow((current_front[j][1]-comprision[k][1]),2))
           sum_Mdistance += min_distance
       M1_star = sum_Mdistance*1/(front_count)
       
       segma = 100
       M2_coount = 0
       for j in range(front_count):
           temp_count = 0
           for k in range(front_count):
               if math.sqrt(pow((current_front[j][0]-current_front[k][0]),2)+pow((current_front[j][1]-current_front[k][1]),2)) > segma:
                   temp_count += 1
           M2_coount += temp_count
       temp_var = front_count-1
       M2_star = M2_coount*1/temp_var
       M3_temp1 = 0
       M3_temp2 = 0
       for j in range(front_count):
          for k in range(front_count):
              if math.fabs(current_front[j][0]-current_front[k][0]) > M3_temp1:
                  M3_temp1 = math.fabs(current_front[j][0]-current_front[k][0])
              if math.fabs(current_front[j][1]-current_front[k][1]) > M3_temp2:
                  M3_temp2 = math.fabs(current_front[j][1]-current_front[k][1])
       M3_star = math.sqrt(M3_temp1+M3_temp2)
        
       with open(r'measurements.txt','a') as f:
           f.write(str(M1_star) + " " + str(M2_star)+ " " + str(M3_star)  + "\n")    
      
              
        
   #print '***'    
   numbers_front = []
   front = []
   '''
   with open ('Pareto_get_front.txt','r') as f:
       data = f.readlines()
       
       for line in data:
           tmp = line.split()
           numbers_front.append(tmp)
           #numbers_front = map(int,tmp)
           #print numbers_front

   '''
   data = pd.read_csv('Pareto_get_front_1.txt',sep=' ',header=None)
   numbers_front = data.values
   #print numbers_front
   
   i = 0  
   while i < len(numbers_front):  
       j = 0  
       while j < len(numbers_front):  
           if i != j:  
               vj1 = numbers_front[j][0]  
               vj2 = numbers_front[j][1]  
               vi1 = numbers_front[i][0]  
               vi2 = numbers_front[i][1]  
               
               
               if (vj1 >= vi1 and vj2 <= vi2) and (vj1 > vi1 or vj2 < vi2):  #no-dominated each other
                   j += 1  
                     
               else:  
                 
                  
                  if (vj1 >= vi1 and vj2 >= vi2) and (vj1 > vi1 or vj2 > vi2):
                      numbers_front[j][0]=10000000
                      numbers_front[j][1]=10000000
                  if (vj1 <= vi1 and vj2 <= vi2) and (vj1 < vi1 or vj2 < vi2):
                      numbers_front[i][0]=10000000
                      numbers_front[i][1]=10000000
                  j +=1  
               if j == len(numbers_front):  
                    #print numbers_front[i]  
                    i += 1     
                    break      
           else:  
                j += 1     
                if i == len(numbers_front)-1 and j == len(numbers_front):  
                    #print numbers_front[i]      
                    i += 1   
   
   #print numbers_front
   
   front = np.unique(numbers_front,axis = 0)
   #print front
   fronts=front.tolist()
   if ([10000000,10000000] in front):
       fronts.remove([10000000,10000000])
   with open(r'Pareto_get_Fitnesses_2.txt','a') as f:
       for result_1 in fronts:
           f.write(str(result_1[0]) + " " + str(result_1[1]) + "\n")
   #front = front[:,-2]
   #front.remove(10000000)
   #print fronts
  
            