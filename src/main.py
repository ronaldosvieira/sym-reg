#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from gp import *

def read_dataset(path):
    return pd.read_csv(path, sep = ',', names = ['x', 'y', 'f'])

def eval_fitness(individual, dataset):
    pred = dataset.apply(lambda row: individual.evaluate(**row), axis = 1)
    truth = dataset['f']
    
    residuals = np.square(truth - pred)
    normalizer = np.square(dataset['f'] - dataset['f'].mean())
    
    return np.sqrt(np.divide(np.sum(residuals), np.sum(normalizer)))

def eval_fitness_batch(individuals, dataset):
    pred = pd.DataFrame(list(map(
        lambda ind: dataset.apply(lambda row: ind.evaluate(**row), axis = 1), 
        individuals))).T
    truth = dataset['f']
    
    residuals = np.square((truth - pred.T).T)
    normalizer = np.square(dataset['f'] - dataset['f'].mean())
    
    return np.sqrt(np.divide(np.sum(residuals, axis = 0), np.sum(normalizer)))

train = read_dataset('data/synth1/synth1-train.csv')

tree = BinaryOperator(lambda x, y: x + y,
        VariableTerminal('x'),
        ConstantTerminal(5))
        
print(eval_fitness(tree, train))
print(eval_fitness_batch([tree, tree, tree], train))