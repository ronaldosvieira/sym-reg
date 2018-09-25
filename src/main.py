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
        pred = individuals['ind'].apply(
            lambda ind: dataset.apply(
                lambda row: ind.evaluate(**row), 
                axis = 1))
        truth = dataset['f']
        
        residuals = np.square((truth - pred))
        normalizer = np.square(dataset['f'] - dataset['f'].mean())
        
        return np.sqrt(np.divide(
            np.sum(residuals, axis = 1), 
            np.sum(normalizer)))
    
    return batch_nrmse

def random_constant(start, end):
    return ConstantTerminal((np.random.rand() * (end - start)) + start)

def gaussian_constant(mean, std_var):
    return ConstantTerminal(np.random.normal(mean, std_var))

operators = [lambda: Operator(lambda x, y: x + y, 2, '{} + {}'),
        lambda: Operator(lambda x, y: x * y, 2, '{} * {}'),
        lambda: Operator(lambda x, y: x / y if y != 0 else 0, 2, "{} / {}"),
        lambda: Operator(lambda x: np.sin(x), 1, "sin({})")]
terminals = [lambda: VariableTerminal('x'), 
        lambda: VariableTerminal('y'),
        lambda: random_constant(-1, 1)]

def random_pop_gen(params):
    pop = []
    
    for _ in range(params['N']):
        ind = np.random.choice(operators)()
        
        for _ in range(ind.arity):
            child = np.random.choice(terminals)()
            child.parent = ind

            ind.children.append(child)
            
        pop.append(ind)
    
    return pd.DataFrame(data = pop, columns = ['ind'])

def full_pop_gen(params):
    pop = []
    max_depth = params['init_max_depth']
    
    def full_ind_gen(max_depth):
        if max_depth <= 1:
            return np.random.choice(terminals)()
        else:
            node = np.random.choice(operators)()
            
            for _ in range(node.arity):
                child = full_ind_gen(max_depth - 1)
                child.parent = node

                node.children.append(child)
            
            return node
            
    for _ in range(params['N']):
        pop.append(full_ind_gen(max_depth))
        
    return pd.DataFrame(data = pop, columns = ['ind'])
    
def grow_pop_gen(params):
    pop = []
    max_depth = params['init_max_depth']
    
    def grow_ind_gen(max_depth):
        if max_depth <= 1:
            return np.random.choice(terminals)()
        else:
            node = np.random.choice(terminals + operators)()
            
            for _ in range(node.arity):
                child = grow_ind_gen(max_depth - 1)
                child.parent = node
                
                node.children.append(child)
            
            return node
    
    for _ in range(params['N']):
        pop.append(grow_ind_gen(max_depth))
        
    return pd.DataFrame(data = pop, columns = ['ind'])

def roulette_selection(pop, params, amount = 1):
    return pop.sample(n = amount, weights = (1 / pop['fitness']), replace = True)

def tournament_selection(pop, params, amount = 1):
    return pd.concat(pop.sample(n = params['k'])
        .sort_values('fitness')
        .head(1) for _ in range(amount))

def find_point(node, point):
        if point == 0:
            return node
        else:
            accum = 0
            
            for child in node.children:
                if point <= child.size() + accum:
                    return find_point(child, point - accum - 1)
                    
                accum += child.size()

def subtree_crossover(ind1, ind2, params):
    ind1, ind2 = ind1.copy(), ind2.copy()
    points = [np.random.randint(0, ind1.size()),
        np.random.randint(0, ind2.size())]
                    
    if points[0] == 0:
        point2 = find_point(ind2, np.random.randint(0, ind2.size()))
        point2.parent = None
        
        return point2
    else:
        point1 = find_point(ind1, points[0])
        point2 = find_point(ind2, points[1])

        index = point1.parent.children.index(point1)
        point1.parent.children[index] = point2
        point2.parent = point1.parent
    
        return ind1

def subtree_mutation(ind, params):
    random_ind = grow_pop_gen(params)['ind'][0]
    
    return subtree_crossover(ind, random_ind, params)

def point_mutation(ind, params):
    point = np.random.randint(0, ind.size())
    node = find_point(ind, point)
    
    compatible = list(filter(
        lambda n: n.arity == node.arity, 
        map(lambda n: n(), operators + terminals)))
        
    if len(compatible) == 0:
        return ind
    
    new_node = np.random.choice(compatible)
    
    if new_node.arity > 0:
        new_node.children = node.children
    
    if node.parent is not None:
        index = node.parent.children.index(node)
        node.parent.children[index] = new_node
        
    return new_node

def all_mutations(ind, params):
    mutations = [subtree_mutation, point_mutation]
    
    return np.random.choice(mutations)(ind, params)

def prune_tree(ind, params):
    def find_depth(node, depth = 0):
        if depth == params['max_depth'] - 1:
            return [node]
        else:
            nodes = []

            for i in range(node.arity):
                nodes.extend(find_depth(node.children[i], depth + 1))

            return nodes

    new_ind = ind.copy()
    nodes = find_depth(new_ind)

    for node in nodes:
        subst = np.random.choice(terminals)()
        subst.parent = node.parent

        index = node.parent.children.index(node)
        node.parent.children[index] = subst

    return new_ind

def dump_ind(ind):
    s = str(ind)
    s += " parent = " + str(ind.parent is not None) + "\n"

    for i in range(ind.arity):
        s += str(ind.depth()) + " " + dump_ind(ind.children[i]) + "\n"

    return s

train = read_dataset('data/synth1/synth1-train.csv')

model = GeneticProgramming(grow_pop_gen, get_nrmse(train), 
            tournament_selection, subtree_crossover, all_mutations, 
            batch_fitness = get_batch_nrmse(train),
            tree_pruning = prune_tree)

result = model.run(N = 10, init_max_depth = 3, max_gen = 10, 
            p_cross = 0.9, p_mut = 0.05, max_depth = 7, k = 2, elitism = 1)

print(result.drop(['pop'], axis = 1))
print(result['pop'].iloc[10])