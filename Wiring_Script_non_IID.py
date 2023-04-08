# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 19:03:20 2022

@author: Alex
"""

import numpy as np
from lib.Common_Functions import quantize_nearest_power_2, quantize_nearest_power_2_finite_set, SBF_set_integer_exponents, Greedy_set_integer_exponents,\
    Mean_centered_integer_exponents,HWES_set_integer_exponents,EWdA_set_integer_exponents
    
from lib.Wiring_Algorithm_Finite_Set import wiring_matrices_finite_set
from lib.Wiring_Algorithm import wiring_matrices
from tqdm import trange




N = 8
K = 2**N
num_wiring = 10
sparsity = 3
#specifies the size of the set of integer coefficients for the wiring matrix entries except the 0 integer exponent which is always in support
cardinality = 4
iterations_averaging = 1000

wiring_exponent_initial_codebook = np.arange(-6,3,dtype=float)

used_algorithms = [HWES_set_integer_exponents,EWdA_set_integer_exponents,SBF_set_integer_exponents,Greedy_set_integer_exponents]

    

#initialize variables for wiring wiring matrices and targets matrices
Wiring_Mats = np.zeros((iterations_averaging,num_wiring,K,K))
Target_Mats = np.random.randn(iterations_averaging,N,K)*1/np.sqrt(N)*np.random.rand(iterations_averaging,K)[:,None,:]*10


#quantize Target_Mats to the next power of 2 (with storage demand of 5+1 bit per entry)
B = quantize_nearest_power_2_finite_set(Target_Mats,wiring_exponent_initial_codebook)


#compute unconstrained wiring as benchmark
for i in trange(iterations_averaging):
    Wiring_Mats[i]= wiring_matrices(B[i], Target_Mats[i], num_wiring, sparsity)
       


np.savez_compressed(f'Target_Wiring_UNN_Constr_Quant_C_Unconstrained_Wiring_sparsity_{sparsity}_non_IID.npz',Wiring_Mats=Wiring_Mats,Target_Mats=Target_Mats)
del Wiring_Mats


    
#compute exponential based algorithms

for alg in used_algorithms:
    #initialize variables for wiring wiring matrices and targets matrices
    Wiring_Mats = np.zeros((iterations_averaging,num_wiring,K,K))
    current_factorization = np.copy(B)
    Wiring_layer_exact = np.zeros((iterations_averaging,K,K))
    
    for j in trange(num_wiring):
        for i in range(iterations_averaging):
            #compute at first depth one wiring matrix decomposition without the constraint on the entries of wiring matrix to assess the mean of the wiring coefficients
            Wiring_layer_exact[i] = wiring_matrices(current_factorization[i], Target_Mats[i], 1, sparsity)
       
        #compute chosen set of wiring exponents by algorithm
        
        set_integer_exponents = alg(Wiring_layer_exact,current_factorization,cardinality)
        
        #ensure that both sets are ordered in descending order
        set_integer_exponents = -np.sort(-set_integer_exponents)
        
        print(set_integer_exponents)
        
        for i in range(iterations_averaging):
            Wiring, current_factorization[i] = wiring_matrices_finite_set(current_factorization[i],Target_Mats[i], 1,  sparsity, set_integer_exponents)
            Wiring_Mats[i,j] = Wiring[0]
            
        print(f'Error: {np.linalg.norm(current_factorization-Target_Mats)**2}')
    
           
    
    
    np.savez_compressed(f'Target_Wiring_UNN_Constr_Quant_C_{alg.__name__}_sparsity_{sparsity}_non_IID.npz',Wiring_Mats=Wiring_Mats,Target_Mats=Target_Mats)
    
    #delete wiring mats to free memory
    del Wiring_Mats
    
    
#compute exponential based algorithms without codebook weighted histogram

for alg in used_algorithms:
    #initialize variables for wiring wiring matrices and targets matrices
    Wiring_Mats = np.zeros((iterations_averaging,num_wiring,K,K))
    current_factorization = np.copy(B)
    Wiring_layer_exact = np.zeros((iterations_averaging,K,K))
    
    for j in trange(num_wiring):
        for i in range(iterations_averaging):
            #compute at first depth one wiring matrix decomposition without the constraint on the entries of wiring matrix to assess the mean of the wiring coefficients
            Wiring_layer_exact[i] = wiring_matrices(current_factorization[i], Target_Mats[i], 1, sparsity)
       
        #compute chosen set of wiring exponents by algorithm
        
        set_integer_exponents = alg(Wiring_layer_exact,current_factorization,cardinality,codebook_weighted=False)
        
        #ensure that both sets are ordered in descending order
        set_integer_exponents = -np.sort(-set_integer_exponents)
        
        print(set_integer_exponents)
        
        for i in range(iterations_averaging):
            Wiring, current_factorization[i] = wiring_matrices_finite_set(current_factorization[i],Target_Mats[i], 1,  sparsity, set_integer_exponents)
            Wiring_Mats[i,j] = Wiring[0]
            
        print(f'Error: {np.linalg.norm(current_factorization-Target_Mats)**2}')
    
           
    
    
    np.savez_compressed(f'Target_Wiring_UNN_Constr_Quant_C_{alg.__name__}_not_codebook_weighted_sparsity_{sparsity}_non_IID.npz',Wiring_Mats=Wiring_Mats,Target_Mats=Target_Mats)
    
    #delete wiring mats to free memory
    del Wiring_Mats
    
    
    



