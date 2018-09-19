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
                fitness = self.batch_fitness(population)
                
                new_population = []
                
                while len(new_population) < params['N']:
                    draw = np.random.random()
                    
                    if (draw <= params['p_cross']):
                        parents = self.selection(population, fitness, amount = 2)
                        
                        children = self.crossover(*parents, params = params)
                        
                    elif (draw <= params['p_cross'] + params['p_mut']):
                        parent = self.selection(population, fitness)
                        
                        children = self.mutation(*parent, params = params)
                    
                    new_population.extend(children)
                    
                population = new_population[0:params['N']]
                generation += 1
            
            return pd.DataFrame({
                'ind': list(map(str, population)),
                'fitness': self.batch_fitness(population)
            }).sort('fitness')
            
        except Exception as e:
            print("Generation: {}".format(generation))
            print("Population: {}".format(list(map(str, population))))
            raise e
