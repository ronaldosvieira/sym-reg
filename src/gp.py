# -*- coding: utf-8 -*-
import numpy as np

class Node:
    pass

class UnaryOperator(Node):
    def __init__(self, f, child = None, str_rep = 'unop({})'):
        self.f = f
        self.arity = 1
        self.child = child
        self.str_rep = str_rep
        
    def evaluate(self, **values):
        return self.f(self.child.evaluate(**values))
        
    def __str__(self):
        return self.str_rep.format(str(self.child))
        
    def copy(self):
        return UnaryOperator(self.f, self.child.copy(), self.str_rep)

class BinaryOperator(Node):
    def __init__(self, f, left = None, right = None, str_rep = 'binop({}, {})'):
        self.f = f
        self.arity = 2
        self.left = left
        self.right = right
        self.str_rep = str_rep
    
    def evaluate(self, **values):
        return self.f(self.left.evaluate(**values), self.right.evaluate(**values))

    def __str__(self):
        return self.str_rep.format(str(self.left), str(self.right))
        
    def copy(self):
        return BinaryOperator(self.f, 
            self.left.copy(), self.right.copy(), self.str_rep)

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

