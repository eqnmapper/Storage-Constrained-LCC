#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:39:31 2021

@author: alex
"""
import numpy as np
from numba import njit, prange
import itertools
from scipy import special

@njit(cache=True)
def ReLU(x):
    return (x+np.abs(x))/2

@njit(cache=True)
def quantize_nearest_power_2(x): 
    shape = x.shape
    x = x.ravel()
    abs_x = np.abs(x)
    biased_round = np.zeros_like(x)
    biased_round[abs_x>0] = np.log2(abs_x[abs_x>0])
    nearest_power_2 = 2**np.rint(biased_round -(np.log2(1.5)-0.5))
    x = np.sign(x)*nearest_power_2
    return x.reshape(shape)

@njit(cache=True)
def quantize_nearest_power_2_finite_set(x,set_integer_exponents):
    shape = x.shape
    x = x.ravel()
    abs_x = np.abs(x)
    #concatenate the vals with each sign 
    vals = np.concatenate((2**set_integer_exponents,-2**set_integer_exponents))
    #concatenate 0 to the vals to compare with
    vals = np.concatenate((vals,np.array([0])))
    smallest_error_idx = np.argmin(np.abs(np.expand_dims(vals,1)-x),axis = 0)
    y = vals[smallest_error_idx]
    
    return y.reshape(shape)

def get_wiring_exponent_support(wiring_entries):
    wiring_entries = np.abs(wiring_entries)
    wiring_entries = wiring_entries[wiring_entries!=0]
    wiring_entries = np.log2(wiring_entries)
    wiring_entries_max = np.rint(1+np.max(wiring_entries))
    wiring_entries_min = np.rint(np.min(wiring_entries)-1)
    range_integer = np.arange(wiring_entries_min,wiring_entries_max+1,1).astype(float)
    return range_integer

def calc_integer_exponent_histogram(wiring_mats):
    wiring = np.copy(wiring_mats)
    wiring_entries = wiring.flatten()
    range_integer = get_wiring_exponent_support(wiring_entries)
    wiring_entries = np.abs(wiring_entries)
    wiring_entries = wiring_entries[wiring_entries!=0]
    wiring_entries = np.log2(wiring_entries)
    histogram = np.zeros_like(range_integer)
    for idx,val in enumerate(range_integer):
        histogram[idx]= np.sum(wiring_entries == val)

    #flip histogram and range_integer such that biggest exponent is in the first entry of the array
    range_integer = np.flip(range_integer)
    histogram = np.flip(histogram)
    
    return histogram,range_integer

def calc_integer_exponent_histogram_codebook_squared_norm_weighted(wiring_mats,codebook_mats):
    codebook_norms_2 = np.sum(codebook_mats**2,axis=1)

    range_integer = get_wiring_exponent_support(wiring_mats.flatten())
    
    wiring_exponent_mats = np.abs(wiring_mats)
    #set zero entries to infinity as they are going to be ignored when checking against the wiring exponents in the support of the histogram
    wiring_exponent_mats[wiring_exponent_mats==0] = +np.inf
    wiring_exponent_mats = np.log2(wiring_exponent_mats)
    
    histogram = np.zeros_like(range_integer)

    for idx,val in enumerate(range_integer):
        for r in range(wiring_mats.shape[0]):
            for k in range(wiring_mats.shape[1]):
                histogram[idx]+=codebook_norms_2[r,k]*np.sum(wiring_exponent_mats[r,k,:] == val)
    
    #flip histogram and range_integer such that biggest exponent is in the first entry of the array
    range_integer = np.flip(range_integer)
    histogram = np.flip(histogram)
    
    return histogram,range_integer


def Mean_centered_integer_exponents(wiring_mats,cardinality):
    wiring_mats_without_zero_exponent = wiring_mats[(np.abs(wiring_mats)<=0.5)]
    histogram,range_integer = calc_integer_exponent_histogram(wiring_mats_without_zero_exponent)

    #normalize histogram such that it sums to one
    histogram = histogram/np.sum(histogram)

    mean_wiring_exponent = np.sum(range_integer*histogram)

    max_exp = int(np.rint(mean_wiring_exponent)+np.ceil((cardinality-2)/2))
    min_exp = int(np.rint(mean_wiring_exponent)-np.floor((cardinality-2)/2))

    return np.concatenate((np.arange(min_exp,max_exp+1,dtype=float),np.array([0],dtype=float)))



# def HWES_set_integer_exponents(wiring_mats,cardinality):
#     wiring_mats_without_zero_exponent = wiring_mats[(np.abs(wiring_mats)<=0.5)]
#     histogram,range_integer = calc_integer_exponent_histogram(wiring_mats_without_zero_exponent)

#     return np.concatenate((range_integer[:cardinality-1],np.array([0])))
def HWES_set_integer_exponents(wiring_mats,current_factorization,cardinality,codebook_weighted=True):
    histogram,range_integer = calc_integer_exponent_histogram(wiring_mats)
    #check successively beginning with the highest wiring exponents, whether it is in support of the histogram and return the obtained list of wiring exponents
    wiring_exponent_set = []
    num_wiring_exponents = 0
    for i in range(len(range_integer)):
        if histogram[i]>0:
            wiring_exponent_set.append(range_integer[i])
            num_wiring_exponents +=1
        if num_wiring_exponents == cardinality:
            break
    
    return np.array(wiring_exponent_set)
    

def EWdA_set_integer_exponents(wiring_mats,current_factorization,cardinality,codebook_weighted=True):
    
    if codebook_weighted:
        histogram,range_integer = calc_integer_exponent_histogram_codebook_squared_norm_weighted(wiring_mats,current_factorization)
    else:
        histogram,range_integer = calc_integer_exponent_histogram(wiring_mats)

    #calculate the exponential weighted histogram
    exp_histogram = np.zeros_like(histogram)
    for i in range(len(range_integer)):
        exp_histogram[i]=2**(-i)*histogram[i]

    cardinality = min(histogram.shape[0],cardinality)
    #determine cardinality biggest components of histogram and return the corresponding wiring exponents
    return range_integer[np.argpartition(exp_histogram, -cardinality)[-cardinality:]]




def SBF_set_integer_exponents(wiring_mats,current_factorization,cardinality,codebook_weighted=True):

    if codebook_weighted:
        histogram,range_integer = calc_integer_exponent_histogram_codebook_squared_norm_weighted(wiring_mats,current_factorization)
    else:
        histogram,range_integer = calc_integer_exponent_histogram(wiring_mats)
    
    histogram_length = histogram.shape[0]
    
    #cardinality should be at most the histogram length
    
    cardinality =  min(histogram_length,cardinality)
    
    subsets = np.array([list(tup) for tup in itertools.combinations(range_integer,cardinality)],dtype=float)
    num_subsets = special.binom(histogram_length,cardinality)
    
    error_reduction = np.zeros(int(num_subsets))
    best_set_integer_exponents = None
    
    for idx,subset in enumerate(subsets):
        for j in range(histogram_length):
            #test for each integer exponent in the subset, which one is the best to reduce the error for the current integer exponent for the histogram
                current_error_reduction = ReLU((2**(2*range_integer[j])-(2**subset-2**range_integer[j])**2))*histogram[j]
                optimal_exponent = np.argmax(current_error_reduction)
                error_reduction[idx] += current_error_reduction[optimal_exponent]
                
    best_set_integer_exponents = subsets[np.argmax(error_reduction)]
    
    return best_set_integer_exponents 


def Greedy_set_integer_exponents(wiring_mats,current_factorization,cardinality,codebook_weighted=True):
    if codebook_weighted:
        histogram,range_integer = calc_integer_exponent_histogram_codebook_squared_norm_weighted(wiring_mats,current_factorization)
    else:
        histogram,range_integer = calc_integer_exponent_histogram(wiring_mats)

    histogram_length = histogram.shape[0]

    #cardinality should be at most the histogram length

    cardinality =  min(histogram_length,cardinality)


    chosen_set_wiring_exponents = np.array([])
    for i in range(cardinality):
        added_wiring_exponent_objective = np.zeros(range_integer.size)
        for j in range(range_integer.size):
            for k in range(histogram_length):
                current_error_reduction = ReLU((2**(2*range_integer[k])-(2**np.concatenate((chosen_set_wiring_exponents,range_integer[j:j+1]))-2**range_integer[k])**2))*histogram[k]
                optimal_exponent = np.argmax(current_error_reduction)
                added_wiring_exponent_objective[j] += current_error_reduction[optimal_exponent]
        chosen_set_wiring_exponents=np.concatenate((chosen_set_wiring_exponents,np.array([range_integer[np.argmax(added_wiring_exponent_objective)]])))
        print(i)

    return chosen_set_wiring_exponents






    
    
    
    
    


