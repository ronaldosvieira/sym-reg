# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

class Node:
    pass

class Operator(Node):
    def __init__(self, f, arity, str_rep, children = None):
        self.f = f
        self.arity = arity
        self.str_rep = str_rep
        self.children = [] if children is None else children[0:arity]
        
    def evaluate(self, **values):
        return self.f(*map(lambda c: c.evaluate(**values), self.children))
        
    def __str__(self):
        return self.str_rep.format(*map(str, self.children))
        
    def copy(self):
        return Operator(self.f, self.arity, self.str_rep, 
            list(map(lambda c: c.copy(), self.children)))
            
    def size(self):
        return sum(map(lambda c: c.size(), self.children)) + 1
        
    def depth(self):
        return max(map(lambda c: c.depth(), self.children)) + 1

class ConstantTerminal(Node):
    def __init__(self, constant):
        self.constant = constant
        self.arity = 0
    
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
    def __init__(self, variable):
        self.variable = variable
        self.arity = 0
        
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

class GeneticProgramming:
    def __init__(self, pop_gen, fitness, 
        selection, crossover, mutation, batch_fitness = None):
        self.pop_gen = pop_gen
        self.fitness = fitness
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.batch_fitness = batch_fitness
        
    def run(self, **params):
        try:
            generation = 0

            # generates initial pop
            population = self.pop_gen(params)
            population['fitness'] = self.batch_fitness(population)
            population = population.sort_values('fitness')

            info = pd.DataFrame([], 
                columns = ['best', 'worst', 'mean', 'duplicated', 
                    'cross', 'mut', 'reprod', 'op_el', 'pop'])

            info.loc[0] = {
                'best': population.iloc[0]['fitness'],
                'worst': population.iloc[-1]['fitness'],
                'mean': population['fitness'].mean(),
                'duplicated': sum(population.duplicated()),
                'cross': 0, 'mut': 0, 'reprod': 0, 'op_el': 0,
                'pop': population
            }
            
            while generation < params['max_gen']:
                # checks for elitism
                try:
                    elite = params['elitism']
                    new_population = population.sort_values('fitness').head(elite)
                except:
                    new_population = pd.DataFrame([], columns = ['ind', 'fitness'])

                count = [0, 0, 0, 0]

                # generates rest of new pops
                while len(new_population) < params['N']:
                    draw = np.random.random()
                    
                    # crossover?
                    if (draw <= params['p_cross']):
                        count[0] += 1
                        parents = self.selection(population, params, amount = 2)

                        child = self.crossover(*parents['ind'], params = params)
                    
                    # mutation?
                    elif (draw <= params['p_cross'] + params['p_mut']):
                        count[1] += 1
                        parents = self.selection(population, params, amount = 1)
                        
                        child = self.mutation(*parents['ind'], params = params)

                    # reproduction
                    else:
                        count[2] += 1
                        parents = self.selection(population, params, amount = 1)

                        child = parents.iloc[0]['ind']
                    
                    # operator elitism
                    family = pd.concat([parents, pd.DataFrame(
                        [[child, self.fitness(child)]], 
                        columns = ['ind', 'fitness'])])
                    best_of_family = family.sort_values('fitness').head(1)

                    if best_of_family.iloc[0]['ind'] != child:
                        count[3] += 1
                    
                    new_population = pd.concat([new_population, best_of_family])
                    
                population = new_population.reset_index(drop = True)
                generation += 1

                # calculates fitness
                population['fitness'] = self.batch_fitness(population)
                population = population.sort_values('fitness')

                # adds info to log
                info.loc[generation] = {
                    'best': population.iloc[0]['fitness'],
                    'worst': population.iloc[-1]['fitness'],
                    'mean': population['fitness'].mean(),
                    'duplicated': sum(population.duplicated()),
                    'cross': count[0], 'mut': count[1], 'reprod': count[2],
                    'op_el': count[3],
                    'pop': population
                }

            return info
            
        except Exception as e:
            #print("Generation: {}".format(generation))
            #print("Population: {}".format(list(map(str, population))))
            raise e
