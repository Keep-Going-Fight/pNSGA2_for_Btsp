#import scipy.io as sio   
import numpy as np
import random
from scipy import io

def Sort_Com(D,row,length):   
   temp = D[row][0]
   temp_col = 0
           
   for i in range(1,length):
       if D[row][i] > temp:
           temp = D[row][i]
           temp_col = i
   #print temp,temp_col
   return temp,temp_col     


if __name__ == "__main__":
    D1 = io.loadmat(r'./../datasets/N100Q100totalQuantity.mat')
    A1 = [[1,2,9],[2,3,0]]
    genotype1 = np.zeros(100, dtype = np.int32)
    _sum = 0
    row = random.randint(0,99)
    temp_D = D1['totalQuantity'][3*100:(3+1)*100]
    genotype1[0] = row
    print temp_D
    '''
    for j in range(0,100):               
        value, row = Sort_Com(temp_D,row,100)
        _sum += value
        #row = col
        genotype1[j] = row
        for i in range(0,100):
            temp_D[i][row] = 0
    #print genotype1, genotype1
    '''

 
    
        