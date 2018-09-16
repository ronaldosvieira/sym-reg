#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
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

def random_population(N):
    pop = []
    elts = [lambda: VariableTerminal('x'), 
        lambda: VariableTerminal('y'),
        lambda: gaussian_constant(0, 10)]
    
    for _ in range(N):
        pop.append(BinaryOperator(lambda x, y: x + y,
                np.random.choice(elts)(),
                np.random.choice(elts)(),
                str_rep = '{} + {}'))
    
    return pop

def crossover(ind1, ind2):
    # hardcoded
    points = list(np.random.choice([0, 1], size = (2,)))
    child1, child2 = ind1.copy(), ind2.copy()
    
    if points == [0, 0]:
        child1.left, child2.left = child2.left, child1.left
    elif points == [0, 1]:
        child1.left, child2.right = child2.right, child1.left
    elif points == [1, 0]:
        child1.right, child2.left = child2.left, child1.right
    elif points == [1, 1]:
        child1.right, child2.right = child2.right, child1.right

    return child1, child2

train = read_dataset('data/synth1/synth1-train.csv')

tree1, tree2 = random_population(2)
print(tree1)
print(tree2)

print(list(map(str, crossover(tree1, tree2))))
