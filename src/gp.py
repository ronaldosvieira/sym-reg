# -*- coding: utf-8 -*-
import numpy as np

class Node:
    pass

class UnaryOperator(Node):
    def __init__(self, f, child = None):
        self.f = f
        self.arity = 1
        self.child = child
        
    def evaluate(self, **values):
        return self.f(self.child.evaluate(**values))

class BinaryOperator(Node):
    def __init__(self, f, left = None, right = None):
        self.f = f
        self.arity = 2
        self.left = left
        self.right = right
    
    def evaluate(self, **values):
        return self.f(self.left.evaluate(**values), self.right.evaluate(**values))

class ConstantTerminal(Node):
    def __init__(self, constant):
        self.constant = constant
    
    def evaluate(self, **values):
        return self.constant

class VariableTerminal(Node):
    def __init__(self, variable):
        self.variable = variable
        
    def evaluate(self, **values):
        try:
            return values[self.variable]
        except:
            raise ValueError("%s value not found" % self.variable)

def random_constant(start, end):
    return ConstantTerminal(np.random.randint(start, end))
    
def gaussian_constant(mean, std_var):
    return ConstantTerminal(np.random.normal(mean, std_var))