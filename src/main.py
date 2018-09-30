#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from gp import *
import math

def read_dataset(path):
    dataset = pd.read_csv(path, sep = ',')

    columns = ['x' + str(i) for i in range(1, len(dataset.columns))] + ['y']
    return dataset.rename(columns = dict(zip(dataset.columns, columns)))

def nrmse(dataset, individual):
    pred = dataset.apply(lambda row: individual.evaluate(**row), axis = 1)
    truth = dataset['y']
    
    residuals = np.square(truth - pred)
    normalizer = np.square(dataset['y'] - dataset['y'].mean())
    
    return np.sqrt(np.divide(np.sum(residuals), np.sum(normalizer)))

def batch_nrmse(dataset, individuals):
    pred = individuals['ind'].apply(
        lambda ind: dataset.apply(
            lambda row: ind.evaluate(**row), 
            axis = 1))
    truth = dataset['y']
    
    residuals = np.square((truth - pred))
    normalizer = np.square(dataset['y'] - dataset['y'].mean())
    
    return np.sqrt(np.divide(
        np.sum(residuals, axis = 1), 
        np.sum(normalizer)))

def random_constant(start, end):
    return ConstantTerminal((np.random.rand() * (end - start)) + start)

def gaussian_constant(mean, std_var):
    return ConstantTerminal(np.random.normal(mean, std_var))

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

def ramped_pop_gen(params):
    pop = []

    amount_grow = math.floor(params['N'] // 2)
    amount_full = math.ceil(params['N'] // 2)

    size = 1
    amount_per_size = amount_grow / params['init_max_depth']

    for i in range(amount_grow):
        while i >= size * amount_per_size:
            size += 1

        pop1 = grow_pop_gen({
            'N': amount_grow,
            'init_max_depth': size})

    size = 1
    amount_per_size = amount_full / params['init_max_depth']

    for i in range(amount_full):
        while i >= size * amount_per_size:
            size += 1

        pop2 = full_pop_gen({
            'N': amount_full,
            'init_max_depth': size})
    
    return pd.concat([pop1, pop2])

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

dataset = 'synth1'

train = read_dataset('data/{0}/{0}-train.csv'.format(dataset))
test = read_dataset('data/{0}/{0}-test.csv'.format(dataset))

variables = ['x' + str(i) for i in range(1, len(train.columns))]

operators = [lambda: Operator(lambda x, y: x + y, 2, '{} + {}'),
        lambda: Operator(lambda x, y: x * y, 2, '{} * {}'),
        lambda: Operator(lambda x, y: x / y if y != 0 else 0, 2, "{} / {}"),
        lambda: Operator(lambda x: np.sin(x), 1, "sin({})"),
        lambda: Operator(lambda x: np.square(x), 1, "({})^2")]
terminals = [lambda: VariableTerminal(np.random.choice(variables)), 
        lambda: random_constant(-1, 1)]

model = GeneticProgramming(train, test, ramped_pop_gen, nrmse, 
            tournament_selection, subtree_crossover, all_mutations, 
            batch_fitness = batch_nrmse,
            tree_pruning = prune_tree)

seeds = range(30)
params = {'N': 10, 'init_max_depth': 3, 'max_gen': 10, 
                'p_cross': 0.9, 'p_mut': 0.05, 'max_depth': 7, 
                'k': 2, 'elitism': 1}

results = []

for seed in seeds:
    print("run seed = {}".format(seed))

    result = model.run(**params, seed = seed)

    results.append(result)

stats = pd.concat(results, keys = range(1, len(results) + 1))

best_inds = pd.concat(list(stats.xs(params['max_gen'], level = 1)['pop'])) \
        .sort_values('test_fitness') \
        .head(50)

filename = '{}_{}_{}_{}_{}'.format(dataset, params['N'], params['max_gen'], 
        params['p_cross'], params['k'])

stats.drop('pop', axis = 1).to_csv(
        'results/{}_stats.csv'.format(filename), 
        sep = ',', encoding = 'utf-8')
stats.drop('pop', axis = 1).mean(level = 1).to_csv(
        'results/{}_stats_mean.csv'.format(filename), 
        sep = ',', encoding = 'utf-8')
best_inds.to_csv(
        'results/{}_pop.csv'.format(filename),
        sep = ',', encoding = 'utf-8')
