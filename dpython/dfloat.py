# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 18:25:21 2016

@author: Ajoo
"""

from .autodiff import *

from weakref import WeakKeyDictionary
import numbers
import math

__all__ = [
    'DiffFloat',
    'dfloat'
    ]

#------------------------------------------------------------------------------
#   DiffFloat - DiffObject specialization for representing floats with
#               derivative information
#? Generalize to Complex? DiffNumber or DiffScalar?
class DiffFloat(DiffObject):
    eps = 1e-8
    def __init__(self, value, d=None, name=None):
        if not isinstance(value, numbers.Number):
            raise ValueError("DiffFloat does not support {0} values".format(type(value)))
        self.value = value
        self.name = name
        if d is not None:        
            self.d = d
            #self.track()
        else:
            self.d = {self: 1.}
    
    def track(self, d_self=None):
        '''
        Toggles on tracking for this object
        (Off by default when derivative information is provided)
        '''
        if d_self is None:
            d_self = 1.
        self.d[self] = d_self
            
    def derivative(self, wrt):
        return self.d.get(wrt, 0.)
    
    def chain(self, df):
        d = {k: df*dk for k, dk in self.d.items()}
        return d
  
    def delta(self, eps=None):
        if not eps:
            eps = self.eps
        return [self.value + eps]
    
    def chain_from_delta(self, f, delta, eps=None):
        if not eps:
            eps = self.eps
        df = (delta[0]-f)/eps
        return self.chain(df)
    
    def __repr__(self):
        if self.name:
            return self.name + '(' + repr(self.value) + ')'
        else:
            return 'dfloat(' + repr(self.value) + ')'
    
    def __str__(self):
        return str(self.value)
        
    def __bool__(self):
        return bool(self.value)
        
    def conjugate(self):
        d = {k: v.conjugate() for k, v in self.d.items()}
        return DiffFloat(self.value.conjugate(), d)
    
    @property
    def imag(self):
        d = {k: v.imag for k, v in self.d.viewitems()}
        return DiffFloat(self.value.imag, d)
    
    @property
    def real(self):
        d = {k: v.real for k, v in self.d.viewitems()}
        return DiffFloat(self.value.real, d)
    
    #python 2
    __nonzero__ = __bool__
        
DiffObject._types[float] = DiffFloat

#Factory function for DiffFloat container objects
def dfloat(value):
    try:
        return type(value)(map(DiffFloat, value))
    except:
        return DiffFloat(value)

#float operations not implemented (yet)
ops_not_implemented = ('mod', 'divmod', 'floordiv', 'trunc')
    
#for opname in ops_not_implemented:
#    setattr(DiffFloat, '__{0}__'.format(opname), _not_implemented_func)

ops_act_on_value = ('lt', 'le', 'eq', 'ne', 'ge', 'gt', 'int', 'float')

for op_sname in ops_act_on_value + ops_not_implemented:
    op_lname = '__{0}__'.format(op_sname)
    setattr(DiffFloat, op_lname, cfunction(getattr(float, op_lname)))

#unary operations
def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    elif x == 0:
        return 0.

dirac_1arg = lambda x: 0. if x != 0. else float('inf')
zero_1arg = lambda x: 0.

uops_derivatives = {
    #dabs returns 0 as a subgradient at 0
    'abs': (sign,),
    'neg': (lambda x: -1.,),
    'pos': (lambda x: 1.,),
}

#binary operations   
bops_derivatives = {
    'add': (lambda x, y: 1.,)*2,
    'sub': (lambda x, y: 1., lambda x, y: -1.),
    'mul': (lambda x, y: y, lambda x, y: x),
#    'div': (lambda x, y: 1./y, lambda x, y: -x/y**2),
    'truediv': (lambda x, y: 1./y, lambda x, y: -x/y**2),
    'pow': (lambda x, y: y*x**(y-1.), lambda x, y: math.log(x)*x**y)
}
#TODO mod, floordiv

#add reflected binary ops
def reflected(fun):
    return (lambda x, y: fun(y, x))
    
bops_derivatives.update({'r' + opname: map(reflected, dop) \
    for opname, dop in bops_derivatives.items()})

#add unary and binary ops to DiffFloat dict
for op_sname, dops in list(uops_derivatives.items()) + list(bops_derivatives.items()):
    op_lname = '__{0}__'.format(op_sname)
    setattr(DiffFloat, op_lname, DiffFunction(getattr(float, op_lname), dops))
    #could also use DiffMethod
    
if __name__ == '__main__':
    import numpy as np    
    
    a = DiffFloat(2., name='a')
    b = DiffFloat(3., name='b')
    
    c = a + b

    v = np.array([a,b,c])
    v2 = np.vdot(v, v)    
    
    
    @DiffFunction
    def foo(x1, x2):
        return x1+x2, x1*x2, math.sin(x1)
        
    @foo.derivative(0)
    def foo(x1, x2):
        return 1., x2, math.cos(x1)
        
    @foo.derivative(1)
    def foo(x1, x2):
        return 1., x1, 0.
        
    d, n, m = foo(a,b)
    
    print('Ops not in float: ', set(dir(a))-set(dir(1.)))
    print()
    print('Ops not in dfloat: ', set(dir(1.))-set(dir(a)))