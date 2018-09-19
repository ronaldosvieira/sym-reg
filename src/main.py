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

operators = [lambda: Operator(lambda x, y: x + y, 2, '{} + {}')]
terminals = [lambda: VariableTerminal('x'), 
        lambda: VariableTerminal('y'),
        lambda: gaussian_constant(0, 10)]

def random_pop_gen(N, max_depth):
    pop = []
    
    for _ in range(N):
        ind = np.random.choice(operators)()
        
        for _ in range(ind.arity):
            ind.children.append(np.random.choice(terminals)())
            
        pop.append(ind)
    
    return pop

def full_pop_gen(N, max_depth):
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
    
def grow_pop_gen(N, max_depth):
    pop = []
    
    def full_ind_gen(max_depth):
        if max_depth == 1:
            return np.random.choice(terminals)()
        else:
            node = np.random.choice(terminals + operators)()
            
            try:
                for _ in range(node.arity):
                    node.children.append(full_ind_gen(max_depth - 1))
            except:
                pass
            
            return node
    
    for _ in range(N):
        pop.append(full_ind_gen(max_depth))
        
    return pop

def roulette_selection(pop, fitness, amount = 1):
    p = normalize((1 / fitness).reshape(1, -1), norm = 'l1')
    return np.random.choice(pop, p = p[0], size = amount)

def find_point(node, parent, point):
        if point == 0:
            return node, parent
        else:
            accum = 0
            
            for child in node.children:
                if point <= child.size() + accum:
                    return find_point(child, node, point - accum - 1)
                    
                accum += child.size()

def subtree_crossover(ind1, ind2, params):
    ind1, ind2 = ind1.copy(), ind2.copy()
    points = [np.random.randint(0, ind1.size()),
        np.random.randint(0, ind2.size())]
                    
    if points[0] == 0:
        point2, _ = find_point(ind2, None, np.random.randint(0, ind2.size()))
        
        return [point2]
    else:
        point1, parent1 = find_point(ind1, None, points[0])
        point2, _ = find_point(ind2, None, points[1])
    
        index = parent1.children.index(point1)
        parent1.children[index] = point2
    
        return [ind1]

def subtree_mutation(ind, params):
    random_ind = grow_pop_gen(1, params['max_depth'] - ind.depth())[0]
    
    return subtree_crossover(ind, random_ind, params)

train = read_dataset('data/synth1/synth1-train.csv')

model = GeneticProgramming(full_pop_gen, get_nrmse(train), roulette_selection, 
            subtree_crossover, subtree_mutation, get_batch_nrmse(train))

result = model.run(N = 10, max_depth = 3, max_gen = 50, 
            p_cross = 0.7, p_mut = 0.3)

print(result)