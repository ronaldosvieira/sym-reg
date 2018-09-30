# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from operator import methodcaller

class Node:
    pass

class Operator(Node):
    def __init__(self, f, arity, str_rep, children = None, parent = None):
        self.f = f
        self.arity = arity
        self.str_rep = str_rep
        self.children = [] if children is None else children[0:arity]
        self.parent = parent
        
    def evaluate(self, **values):
        return self.f(*map(lambda c: c.evaluate(**values), self.children))
        
    def __str__(self):
        return self.str_rep.format(*map(str, self.children))
        
    def copy(self):
        op = Operator(self.f, self.arity, self.str_rep)
        op.children = list(map(lambda c: c.copy(), self.children))

        for child in op.children:
            child.parent = op

        return op
            
    def size(self):
        return sum(map(lambda c: c.size(), self.children)) + 1
        
    def depth(self):
        return max(map(lambda c: c.depth(), self.children)) + 1

class ConstantTerminal(Node):
    def __init__(self, constant, parent = None):
        self.constant = constant
        self.arity = 0
        self.parent = parent
    
    def evaluate(self, **values):
        return self.constant
        
    def __str__(self):
        return str(self.constant)
        
    def copy(self):
        return ConstantTerminal(self.constant)
        
    def size(self):
        return 1
        
    def depth(self):
        return 1

class VariableTerminal(Node):
    def __init__(self, variable, parent = None):
        self.variable = variable
        self.arity = 0
        self.parent = parent
        
    def evaluate(self, **values):
        try:
            return values[self.variable]
        except:
            raise ValueError("%s value not found" % self.variable)
            
    def __str__(self):
        return str(self.variable)
        
    def copy(self):
        return VariableTerminal(str(self.variable))
        
    def size(self):
        return 1
        
    def depth(self):
        return 1

class Log:
    def __init__(self):
        self.data = pd.DataFrame([], 
                columns = ['best', 'worst', 'mean', 'mean_depth', 'mean_size',
                    'duplicated', 'cross', 'mut', 'reprod', 'bc_cross', 'bc_mut',
                    'pop'])

    def log(self, info, generation):
        self.data.loc[generation] = info

class GeneticProgramming:
    def __init__(self, train, test, pop_gen, fitness, 
        selection, crossover, mutation, batch_fitness = None, 
        tree_pruning = lambda ind, params: ind):
        self.train = train
        self.test = test
        self.pop_gen = pop_gen
        self.fitness = fitness
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.batch_fitness = batch_fitness
        self.tree_pruning = tree_pruning

        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            'crossovers': 0,
            'mutations': 0,
            'reproductions': 0,
            'bc_crossover': 0,
            'bc_mutation': 0,
            'op_elitisms': 0
        }

    def return_best(self, parents, child, stat_key):
        family = pd.concat([parents, child])
        best_of_family = family.sort_values('fitness').head(1)

        if best_of_family.iloc[0]['ind'] == child.iloc[0]['ind']:
            self.stats[stat_key] += 1

        return best_of_family

    def run(self, **params):
        try:
            if 'seed' in params:
                np.random.seed(params['seed'])

            generation = 0

            # generates initial pop
            population = self.pop_gen(params)
            population['fitness'] = self.batch_fitness(self.train, population)
            population = population.sort_values('fitness')

            info = Log()

            info.log({
                'best': population.iloc[0]['fitness'],
                'worst': population.iloc[-1]['fitness'],
                'mean': population['fitness'].mean(),
                'mean_depth': population['ind']
                        .apply(methodcaller('depth')).mean(),
                'mean_size': population['ind']
                        .apply(methodcaller('size')).mean(),
                'duplicated': sum(population['ind'].apply(str).duplicated()),
                'cross': 0, 'mut': 0, 'reprod': 0, 'bc_cross': 0, 'bc_mut': 0,
                'pop': population
            }, 0)

            if params.get('op_elitism', False):
                op_elitism = lambda child, best: best
            else:
                op_elitism = lambda child, best: child
            
            while generation < params['max_gen']:
                # checks for elitism
                try:
                    elite = params['elitism']
                    new_population = population.sort_values('fitness').head(elite)
                    new_population['fitness'].iloc[0]
                except KeyError:
                    new_population = pd.DataFrame([], columns = ['ind', 'fitness'])

                # generates rest of new pops
                while len(new_population) < params['N']:
                    draw = np.random.random()
                    
                    # crossover?
                    if (draw <= params['p_cross']):
                        self.stats['crossovers'] += 1
                        parents = self.selection(population, params, amount = 2)

                        child = self.crossover(*parents['ind'], params = params)
                        child = pd.DataFrame(
                            [[child, self.fitness(self.train, child)]], 
                            columns = ['ind', 'fitness'])

                        best = self.return_best(parents, child, 'bc_crossover')
                    
                    # mutation?
                    elif (draw <= params['p_cross'] + params['p_mut']):
                        self.stats['mutations'] += 1
                        parents = self.selection(population, params, amount = 1)
                        
                        child = self.mutation(*parents['ind'], params = params)
                        child = pd.DataFrame(
                            [[child, self.fitness(self.train, child)]], 
                            columns = ['ind', 'fitness'])
                        best = self.return_best(parents, child, 'bc_mutation')

                    # reproduction
                    else:
                        self.stats['reproductions'] += 1
                        parents = self.selection(population, params, amount = 1)

                        child = parents
                        best = parents
                    
                    # operator elitism
                    new_individual = op_elitism(child, best)
                    
                    new_population = pd.concat([new_population, new_individual])
                    
                population = new_population.reset_index(drop = True)
                population['ind'] = population['ind'].apply(
                    lambda i: self.tree_pruning(i, params))
                generation += 1

                population = population.sort_values('fitness')

                # adds info to log
                info.log({
                    'best': population.iloc[0]['fitness'],
                    'worst': population.iloc[-1]['fitness'],
                    'mean': population['fitness'].mean(),
                    'mean_depth': population['ind']
                        .apply(methodcaller('depth')).mean(),
                    'mean_size': population['ind']
                        .apply(methodcaller('size')).mean(),
                    'duplicated': sum(population['ind'].apply(str).duplicated()),
                    'cross': self.stats['crossovers'], 
                    'mut': self.stats['mutations'], 
                    'reprod': self.stats['reproductions'],
                    'bc_cross': self.stats['bc_crossover'],
                    'bc_mut': self.stats['bc_mutation'],
                    'pop': population
                }, generation)

                self.reset_stats()

            population['test_fitness'] = self.batch_fitness(self.test, population)
            population = population.sort_values('fitness')

            info.data.at[generation, 'pop'] = population

            return info.data
            
        except KeyboardInterrupt:
            return info.data
