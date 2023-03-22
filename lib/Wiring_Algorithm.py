#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 13:21:24 2021

@author: alex
"""
import numpy as np
from .Common_Functions import quantize_nearest_power_2
from numba import njit, prange

@njit(parallel = True, cache=True)
def wiring_matrices(B,T,num,s):
    K = T.shape[1]
    N = T.shape[0]
    W_array = np.zeros((num,K,K))
    T_approx = B
    for i in range(num):
        T_copy = T.copy()
        for col in prange(K):
            W_array[i,:,col] = greedy_wiring_vector(T_approx,T_copy[:,col],s)
        T_approx = T_approx@W_array[i,:,:]
    return W_array
@njit(cache=True)
def greedy_wiring_vector(T_approx,T_col,s):
    K = T_approx.shape[1]
    w = np.zeros(K)
    for i in range(s):
        idx,a = closest_codebook_vector(T_approx,T_col)
        T_col = T_col-a*T_approx[:,idx]
        del_w = np.zeros(K)
        del_w[idx] = a
        w = w + del_w
    return w
            
@njit(cache=True)
def closest_codebook_vector(T_approx,T_col):
    K = T_approx.shape[1]
    d = np.zeros(K)
    a = np.zeros(K)
    T_approx_norm = np.sqrt(np.sum(T_approx**2,axis = 0))
    a[T_approx_norm > 0] = (T_approx.T@T_col)[T_approx_norm >0]/T_approx_norm[T_approx_norm > 0]**2
    a = quantize_nearest_power_2(a)   
    squared_distance = (a*T_approx).T-T_col
    squared_distance = np.sum(squared_distance**2,axis = 1)
    idx = np.argmin(squared_distance)
    return idx,a[idx]
