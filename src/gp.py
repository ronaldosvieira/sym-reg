# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

class Node:
    pass

class UnaryOperator(Node):
    def __init__(self, f, children = None, str_rep = 'unop({})'):
        self.f = f
        self.arity = 1
        self.children = [] if children is None else children[0]
        self.str_rep = str_rep
        
    def evaluate(self, **values):
        return self.f(map(lambda c: c.evaluate(**values), self.children))
        
    def __str__(self):
        return self.str_rep.format(*map(str, self.children))
        
    def copy(self):
        return UnaryOperator(self.f, 
            list(map(lambda c: c.copy(), self.children)), self.str_rep)

class BinaryOperator(Node):
    def __init__(self, f, children = None, str_rep = 'binop({}, {})'):
        self.f = f
        self.arity = 2
        self.children = [] if children is None else children[0:2]
        self.str_rep = str_rep
    
    def evaluate(self, **values):
        return self.f(*list(map(
            lambda c: c.evaluate(**values), 
            self.children)))

    def __str__(self):
        return self.str_rep.format(*map(str, self.children))
        
    def copy(self):
        return BinaryOperator(self.f, 
            list(map(lambda c: c.copy(), self.children)), self.str_rep)

class ConstantTerminal(Node):
    def __init__(self, constant):
        self.constant = constant
    
    def evaluate(self, **values):
        return self.constant
        
    def __str__(self):
        return str(self.constant)
        
    def copy(self):
        return ConstantTerminal(self.constant)

class VariableTerminal(Node):
    def __init__(self, variable):
        self.variable = variable
        
    def evaluate(self, **values):
        try:
            return values[self.variable]
        except:
            raise ValueError("%s value not found" % self.variable)
            
    def __str__(self):
        return str(self.variable)
        
    def copy(self):
        return VariableTerminal(str(self.variable))

class GeneticProgramming:
    def __init__(self, pop_gen, fitness, selection, crossover, mutation, 
        batch_fitness = None):
        self.pop_gen = pop_gen
        self.fitness = fitness
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.batch_fitness = batch_fitness
        
    def run(self, **params):
        try:
            population = self.pop_gen(params['N'])
            generation = 0
            
            while generation < params['max_gen']:
                fitness = self.batch_fitness(population)
                
                new_population = []
                
                while len(new_population) < params['N']:
                    draw = np.random.random()
                    
                    if (draw <= params['p_cross']):
                        parents = self.selection(population, fitness, amount = 2)
                        children = self.crossover(*parents)
                    elif (draw <= params['p_cross'] + params['p_mut']):
                        parent = self.selection(population, fitness)
                        children = self.mutation(*parent)
                    
                    new_population.extend(children)
                    
                population = new_population
                generation += 1
            
            return pd.DataFrame({
                'ind': list(map(str, population)),
                'fitness': self.batch_fitness(population)
            }).sort('fitness')
        except Exception as e:
            print("Generation: {}".format(generation))
            print("Population: {}".format(list(map(str, population))))
            raise e
