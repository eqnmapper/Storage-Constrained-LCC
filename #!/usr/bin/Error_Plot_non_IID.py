#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 20:42:33 2021

@author: alex
"""


import numpy as np
import matplotlib.pyplot as plt
from lib.Common_Functions import quantize_nearest_power_2,quantize_nearest_power_2_finite_set

#list of decision algorithms to be plotted
used_algorithms = ['Unconstrained_Wiring','HWES_set_integer_exponents',
                   'EWdA_set_integer_exponents','EWdA_set_integer_exponents_not_codebook_weighted',
                   'SBF_set_integer_exponents','SBF_set_integer_exponents_not_codebook_weighted',
                   'Greedy_set_integer_exponents','Greedy_set_integer_exponents_not_codebook_weighted']

#determine the file name of each algorithm

data_paths = []
for alg in used_algorithms:
    data_paths.append(f'Target_Wiring_UNN_Constr_Quant_C_{alg}_sparsity_3_non_IID.npz')


wiring_depth = 10
number_datasets = len(data_paths)
number_simulations = 1000

wiring_exponent_initial_codebook = np.arange(-6,3,dtype=float)

MSE = np.zeros((number_datasets,wiring_depth+1))



for idx, path in enumerate(data_paths):

    simulation_data = np.load(path)

    wiring_mats = simulation_data.get('Wiring_Mats')

    target_mats = simulation_data.get('Target_Mats')

    #calculate norm of error between columns of target matrix and approximated target matrix
    N = target_mats.shape[1]
    K = target_mats.shape[2]

    Squared_Error = np.zeros((number_simulations,wiring_depth+1,N,K))

    for i in range(number_simulations):
        B = quantize_nearest_power_2_finite_set(target_mats[i],wiring_exponent_initial_codebook)
            
        T_approx = B
        Squared_Error[i,0] = (target_mats[i]-T_approx)**2
        for j in range(wiring_depth):
            T_approx = T_approx@wiring_mats[i,j]
            #norm_column_error[i,j+1] = np.linalg.norm((target_mats[i]-T_approx),axis = 0)
            Squared_Error[i,j+1] = (target_mats[i]-T_approx)**2

    MSE[idx,:] = np.mean(Squared_Error,axis=(0,2,3))

    
    del target_mats

    del wiring_mats




plt.figure(figsize=(16,9))
layers = np.arange(0,wiring_depth+1)

for idx, alg in enumerate(used_algorithms):
    plt.plot(layers,np.log2(MSE[idx]),label=alg)

plt.ylabel('$\mathrm{log_2}(\mathrm{MSE})$')
plt.xlabel('Wiring Layer')
plt.xticks(np.arange(0,11,1),np.arange(0,11,1))
plt.legend()
plt.savefig('MSE_non_IID_sparsity_3', dpi=200)


