#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from gp import *

def read_dataset(path):
    return pd.read_csv(path, sep = ',', names = ['x', 'y', 'f'])

def get_nrmse(dataset):
    def nrmse(individual):
        pred = dataset.apply(lambda row: individual.evaluate(**row), axis = 1)
        truth = dataset['f']
        
        residuals = np.square(truth - pred)
        normalizer = np.square(dataset['f'] - dataset['f'].mean())
        
        return np.sqrt(np.divide(np.sum(residuals), np.sum(normalizer)))
        
    return nrmse

def get_batch_nrmse(dataset):
    def batch_nrmse(individuals):
        pred = pd.DataFrame(list(map(
            lambda ind: dataset.apply(
                lambda row: ind.evaluate(**row), 
                axis = 1), 
            individuals))).T
        truth = dataset['f']
        
        residuals = np.square((truth - pred.T).T)
        normalizer = np.square(dataset['f'] - dataset['f'].mean())
        
        return np.sqrt(np.divide(
            np.sum(residuals, axis = 0), 
            np.sum(normalizer)))
    
    return batch_nrmse

def random_constant(start, end):
    return ConstantTerminal(np.random.randint(start, end))

def gaussian_constant(mean, std_var):
    return ConstantTerminal(np.random.normal(mean, std_var))

def random_pop_gen(N, max_depth, operators, terminals):
    pop = []
    
    for _ in range(N):
        ind = np.random.choice(operators)()
        
        for _ in range(ind.arity):
            ind.children.append(np.random.choice(terminals)())
            
        pop.append(ind)
    
    return pop

def full_pop_gen(N, max_depth, operators, terminals):
    pop = []
    
    def full_ind_gen(max_depth):
        if max_depth == 1:
            return np.random.choice(terminals)()
        else:
            node = np.random.choice(operators)()
            
            for _ in range(node.arity):
                node.children.append(full_ind_gen(max_depth - 1))
            
            return node
            
    for _ in range(N):
        pop.append(full_ind_gen(max_depth))
        
    return pop

def selection(pop, fitness, amount = 1):
    p = normalize((1 / fitness).reshape(1, -1), norm = 'l1')
    return np.random.choice(pop, p = p[0], size = amount)

def crossover(ind1, ind2):
    # hardcoded
    points = list(np.random.choice([0, 1], size = (2,)))
    new_ind1, new_ind2 = ind1.copy(), ind2.copy()
    
    new_ind1.children[points[0]], new_ind2.children[points[1]] = \
        new_ind2.children[points[1]], new_ind1.children[points[0]]

    return new_ind1, new_ind2

def mutation(ind, operators, terminals):
    # hardcoded
    point = np.random.choice([0, 1])
    new_ind = ind.copy()
    
    new_ind.children[point] = np.random.choice(terminals)()
        
    return [new_ind]

operators = [lambda: Operator(lambda x, y: x + y, 2, '{} + {}')]
terminals = [lambda: VariableTerminal('x'), 
        lambda: VariableTerminal('y'),
        lambda: gaussian_constant(0, 10)]

train = read_dataset('data/synth1/synth1-train.csv')

model = GeneticProgramming(operators, terminals, 
            full_pop_gen, get_nrmse(train), selection, 
            crossover, mutation, get_batch_nrmse(train))

result = model.run(N = 10, max_depth = 3, max_gen = 15, 
            p_cross = 0.7, p_mut = 0.3)

print(result)