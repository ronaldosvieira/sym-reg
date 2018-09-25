# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

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
                columns = ['best', 'worst', 'mean', 'duplicated', 
                    'cross', 'mut', 'reprod', 'op_el', 'pop'])

    def log(self, info, generation):
        self.data.loc[generation] = info

class GeneticProgramming:
    def __init__(self, pop_gen, fitness, 
        selection, crossover, mutation, batch_fitness = None, 
        tree_pruning = lambda ind, params: ind):
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
            'op_elitisms': 0
        }

    def return_best(self, parents, child):
        family = pd.concat([parents, pd.DataFrame(
            [[child, self.fitness(child)]], 
            columns = ['ind', 'fitness'])])
        best_of_family = family.sort_values('fitness').head(1)

        if best_of_family.iloc[0]['ind'] != child:
            self.stats['op_elitisms'] += 1

        return best_of_family

    def run(self, **params):
        try:
            generation = 0

            # generates initial pop
            population = self.pop_gen(params)
            population['fitness'] = self.batch_fitness(population)
            population = population.sort_values('fitness')

            info = Log()

            info.log({
                'best': population.iloc[0]['fitness'],
                'worst': population.iloc[-1]['fitness'],
                'mean': population['fitness'].mean(),
                'duplicated': sum(population['ind'].apply(str).duplicated()),
                'cross': 0, 'mut': 0, 'reprod': 0, 'op_el': 0,
                'pop': population
            }, 0)

            if params.get('op_elitism', False):
                op_elitism = self.return_best
            else:
                op_elitism = lambda parents, child: pd.DataFrame(
                    [[child, float("inf")]],
                    columns = ['ind', 'fitness'])
            
            while generation < params['max_gen']:
                # checks for elitism
                try:
                    elite = params['elitism']
                    new_population = population.sort_values('fitness').head(elite)
                except:
                    new_population = pd.DataFrame([], columns = ['ind', 'fitness'])

                # generates rest of new pops
                while len(new_population) < params['N']:
                    draw = np.random.random()
                    
                    # crossover?
                    if (draw <= params['p_cross']):
                        self.stats['crossovers'] += 1
                        parents = self.selection(population, params, amount = 2)

                        child = self.crossover(*parents['ind'], params = params)
                    
                    # mutation?
                    elif (draw <= params['p_cross'] + params['p_mut']):
                        self.stats['mutations'] += 1
                        parents = self.selection(population, params, amount = 1)
                        
                        child = self.mutation(*parents['ind'], params = params)

                    # reproduction
                    else:
                        self.stats['reproductions'] += 1
                        parents = self.selection(population, params, amount = 1)

                        child = parents.iloc[0]['ind']
                    
                    # operator elitism
                    new_individual = op_elitism(parents, child)
                    
                    new_population = pd.concat([new_population, new_individual])
                    
                population = new_population.reset_index(drop = True)
                population['ind'] = population['ind'].apply(
                    lambda i: self.tree_pruning(i, params))
                generation += 1

                # calculates fitness
                population['fitness'] = self.batch_fitness(population)
                population = population.sort_values('fitness')

                # adds info to log
                info.log({
                    'best': population.iloc[0]['fitness'],
                    'worst': population.iloc[-1]['fitness'],
                    'mean': population['fitness'].mean(),
                    'duplicated': sum(population['ind'].apply(str).duplicated()),
                    'cross': self.stats['crossovers'], 
                    'mut': self.stats['mutations'], 
                    'reprod': self.stats['reproductions'],
                    'op_el': self.stats['op_elitisms'],
                    'pop': population
                }, generation)

                self.reset_stats()

            return info.data
            
        except Exception as e:
            #print("Generation: {}".format(generation))
            #print("Population: {}".format(population))
            raise e
