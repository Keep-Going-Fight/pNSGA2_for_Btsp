import numpy as np
from scipy import io
import random
"""
define the operational character between individuals
such as dominate,lt,gt,eq....
"""

class TSPGenome:
    __slots__ = ['n_cities', 'fitnesses', 'dominates_list',
                 'inverse_domination_count', 'rank', 'crowding_distance',
                 'genotype']

    """
    A 1-D list of the cities to visit (in order)
    """
    n_objectives = 2
 
    

    def __init__(self, n_cities, run,genotype=None):           
        self.n_cities = n_cities
        self.run = run
        self.fitnesses = np.zeros(self.n_objectives, dtype='uint32')
        self.dominates_list = None  # List of individuals that this individual dominates  Sp
        self.inverse_domination_count = float('-inf')  # Number of individuals that dominate this individual  np
        self.rank = -1  # Member of the n'th pareto front; 0 being the best 
        self.crowding_distance = -1
        if genotype is None:
            #D1_1 = io.loadmat(r'./../datasets/N100Q100totalQuantity.mat')
            #D1_2 = io.loadmat(r'./../datasets/N100Q100totalQuantity_eculidB100.mat')
            #D1_1 = io.loadmat(r'./../datasets/N100Q100totalQuantity_euclidE100.mat')
            #D1_2 = io.loadmat(r'./../datasets/N100Q100totalQuantity_euclidF100.mat')
            #D1_1 = io.loadmat(r'./../datasets/N100Q100totalQuantity_euclidC100.mat')
            #D1_2 = io.loadmat(r'./../datasets/N100Q100totalQuantity_euclidD100.mat')
            #D1_1 = io.loadmat(r'./../datasets/N100Q100totalQuantity_randomA100.mat')
            #D1_2 = io.loadmat(r'./../datasets/N100Q100totalQuantity_randomB100.mat')
            #D1_1 = io.loadmat(r'./../datasets/N100Q100totalQuantity_randomC100.mat')
            #D1_2 = io.loadmat(r'./../datasets/N100Q100totalQuantity_randomD100.mat')

            #D1_1 = io.loadmat(r'./../datasets/N150Q100totalQuantity_kroA150.mat')
            #D1_2 = io.loadmat(r'./../datasets/N150Q100totalQuantity_kroB150.mat')
            #D1_1 = io.loadmat(r'./../datasets/N200Q100totalQuantity_kroA200.mat')
            #D1_2 = io.loadmat(r'./../datasets/N200Q100totalQuantity_kroB200.mat')
            
            #D1_1 = io.loadmat(r'./../datasets/N100Q100totalQuantity_kroA100.mat')
            #D1_1 = io.loadmat(r'./../datasets/N100Q100totalQuantity_kroB100.mat')
            #D1_1 = io.loadmat(r'./../datasets/N100Q100totalQuantity_kroC100.mat')
            D1_1 = io.loadmat(r'./../datasets/N100Q100totalQuantity_kroD100.mat')
            D1_2 = io.loadmat(r'./../datasets/N100Q100totalQuantity_kroE100.mat')
            tempD1_1 = D1_1['totalQuantity']
            tempD1_2 = D1_2['totalQuantity']
            D1 = np.multiply(tempD1_1,tempD1_2)           
            genotype1 = np.zeros(self.n_cities, dtype = np.int32)
            _sum = 0
            row = random.randint(0,self.n_cities-1)
            #print D1['totalQuantity'].shape
            star = int(self.run*100)
            end = int ((self.run + 1)*100)
            #temp_D = np.array(D1['totalQuantity'])[star:end,:]
            temp_D = np.array(D1)[star:end,:]
            #temp_D = np.array(D1['totalQuantity'])[2*100:3*100,:]
            #print temp_D.shape
            #print (self.run+1)*100
            #print temp_D
            genotype1[0] = row
            for j in range(0,self.n_cities):               
                value, row = Sort_Com(temp_D,row,self.n_cities)
                _sum += value
                #row = col
                genotype1[j] = row
                for i in range(0,self.n_cities):
                    temp_D[i][row] = 0
            #print genotype1, genotype1
            self.genotype = genotype1
            
        else:
            #print type(run)
            self.genotype = genotype

        #print self.genotype
    def dominates(self, individual_b):
        """
        The concept of dominates:
        Individual_a dominates individual_f if both:
            a is no worse than b in regards to all fitnesses
            a is strictly better than b in regards to at least one fitness
        Assumes that lower fitness is better, as is the case with cost-distance-TSP.
        :param self:
        :param individual_b:
        :return: True if individual_a dominates individual_b  a<b
                 False if individual_b dominates individual_a or neither dominate each other  b<a
        """
        a_no_worse_b = 0  # a <= b
        a_strictly_better_b = 0  # a < b
        n_objectives = len(self.fitnesses)
        for fitness_i in range(n_objectives):
            f_a = self.fitnesses[fitness_i]
            f_b = individual_b.fitnesses[fitness_i]
            if f_a < f_b:
                a_no_worse_b += 1
                a_strictly_better_b += 1
            elif f_a == f_b:
                a_no_worse_b += 1
            else:
                return False
        return a_no_worse_b == n_objectives and a_strictly_better_b >= 1

    def __lt__(self, other):
        """ Even though A < B, that does not indicate that A.dominates(B),
        as A may have a lower value for fit. func. 1 but greater value for
        fit. func 2 and therefore neither dominate each other. """
        for i in range(len(self.fitnesses)):
            if self.fitnesses[i] < other.fitnesses[i]:
                return True
            if self.fitnesses[i] > other.fitnesses[i]:
                return False
        return False

    def __gt__(self, other):
        for i in range(len(self.fitnesses)):#it can eaual,but at least a>b
            if self.fitnesses[i] > other.fitnesses[i]:
                return True
            if self.fitnesses[i] < other.fitnesses[i]:
                return False
        return False

    def __eq__(self, other):
        for i in range(len(self.fitnesses)):
            if self.fitnesses[i] != other.fitnesses[i]:
                return False
        return True

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __ne__(self, other):
        for i in range(len(self.fitnesses)):
            if self.fitnesses[i] != other.fitnesses[i]:
                return True
        return False
def Sort_Com(D,row,length):   
   temp = D[row][0]
   temp_col = 0
           
   for i in range(1,length):
       if D[row][i] > temp:
           temp = D[row][i]
           temp_col = i
   #print temp,temp_col
   return temp,temp_col