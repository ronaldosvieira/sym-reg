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
            population = self.pop_gen(params)
            
            while generation < params['max_gen']:
                # calculates fitness
                population['fitness'] = self.batch_fitness(population)
                
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
                        parents = self.selection(population, params, amount = 2)

                        child = self.crossover(*parents['ind'], params = params)
                    
                    # mutation?
                    elif (draw <= params['p_cross'] + params['p_mut']):
                        parents = self.selection(population, params, amount = 1)
                        
                        child = self.mutation(*parents['ind'], params = params)
                    
                    # operator elitism
                    family = pd.concat([parents, pd.DataFrame(
                        [[child, self.fitness(child)]], 
                        columns = ['ind', 'fitness'])])
                    best_of_family = family.sort_values('fitness').head(1)
                    
                    new_population = pd.concat([new_population, best_of_family])
                    
                population = new_population
                generation += 1
            
            population['fitness'] = self.batch_fitness(population)
            
            return population.sort_values('fitness')
            
        except Exception as e:
            #print("Generation: {}".format(generation))
            #print("Population: {}".format(list(map(str, population))))
            raise e
