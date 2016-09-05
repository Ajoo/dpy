# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 17:31:59 2016

@author: Ajoo
"""

from collections import Iterable
import types


__all__ = [
    'DiffObject',
    
    'DiffFunctionByList',
    'DiffFunction',
    'dfunction',
    
    'DiffClass',
    
    'ConstFunction',
    'cfunction',
    ]
    
#? Maybe keep track of created objects?
#? maybe replace with factory FUNCTION?
class DiffObject(object):
    '''
    Class that acts as a factory for the different Differentible Object types.
    It uses the type of the value passed to figure out which descendent to 
    instantiate based on the types registered in the static dict _types
    '''
    _types = {}
    #TODO: DiffList and DiffTuple and DiffDict
    def __new__(cls, value, *args, **kwargs):
        if cls is DiffObject:
            try:
                dsc = DiffObject._types[type(value)]
            except:
                raise TypeError("Type {0} not implemented as a differentiable object"\
                    .format(type(value)))
            return super(DiffObject, cls).__new__(dsc)
        
        return super(DiffObject, cls).__new__(cls)
            
        
    def __init__(self, value, d=None, d_self=1.0):
        self.value = value
        if d is not None:        
            self.d = d
        else:
            self.d = {self:d_self}
            
    #? change d to be dictionary specialization that implements chain method?
    def chain(self, df):
        raise NotImplementedError("Chain method not implemented for DiffObject")
    
    def derivative(self, wrt):
        return NotImplemented
        
    def D(self, wrt):
        return self.derivative(wrt)
    
    def __hash__(self):
        return id(self)

#TODO: Metaclass to automatically generate a ValueProxy class for every
     #subclass of DiffClass
class DiffDict(DiffObject):
    pass

class DiffMetaClass(type):
    pass

#TODO: define differentiable methods that provide derivatives wrt self (a DiffClass)
#   This should have a convinient way to define method and dmethods
class DiffClass(DiffObject, metaclass=DiffMetaClass):
    @property
    def params_list(self):
        #TODO: Make a Parameter descriptor class and params_list is instead an
        #attribute populated by these descriptors
        return [k for k in self.__dict__.keys() if isinstance(k, DiffObject)]
    
    @property
    def value(self):
        return ValueProxy(self)
        #return self._value_proxy(self)
    
    def chain(self, df):
        pass
    
    def __setattr__(self, name, value):
        #TODO: update self.value if value is DiffObject
        super(DiffClass, self).__setattr__(name, value)

class ValueProxy(object):
    __slots__ = ["_obj", "__weakref__"]
    
    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)
        
    def __getattribute__(self, name):
        return getattr(object.__getattribute__(self, "_obj"), name)
    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_obj"), name)
    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)
    
    def __nonzero__(self):
        return bool(object.__getattribute__(self, "_obj"))
    def __str__(self):
        return str(object.__getattribute__(self, "_obj"))
    def __repr__(self):
        return repr(object.__getattribute__(self, "_obj"))
        
    

def _not_implemented_func(*args, **kwargs):
    return NotImplemented
NoneFunction = _not_implemented_func

def sum_dicts(*dlist):
    d = {}
    for di in dlist:
        for k, v in di.items():
            d[k] = v + d.get(k, 0.) #v + d[k] if k in d else v
    return d

def is_dobject(x):
    return isinstance(x, DiffObject)


class DiffFunctionBase(object):            
    def __get__(self, instance, cls):
        #return functools.partial(DiffFunction.__call__, self, instance) 
        return types.MethodType(self, instance)
    
    def finite_differences(self, *args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs #TODO: for now can not diff wrt kwargs       
        
        #TODO: won't work for DiffFunctionJoint
        f = self.func(*argvalues, **kwargvalues)
        if not any([isinstance(arg, DiffObject) for arg in args]):
            return f
        
        df = [self.finite_difference(i, arg, f, *argvalues, **kwargvalues) \
            for i, arg in enumerate(args) if isinstance(arg, DiffObject)]
        if type(f) in DiffObject._types:        
            d = sum_dicts(*df)
            return DiffObject(f, d)
        else:
            d = [sum_dicts(*d) for d in zip(*df)]
            return type(f)(map(DiffObject, f, d))

    
    def finite_difference(self, index, darg, f, *args, **kwargs):
        farg = lambda arg: self.func(*(args[0:index]+(arg,)+args[index+1:]), **kwargs)
        d = map(farg, darg.delta())
        if type(f) in DiffObject._types:
            return darg.chain_from_delta(f, d)
        elif isinstance(f, Iterable):
            return [darg.chain_from_delta(fi, di) for fi, di in zip(f,zip(*d))]
            
        raise TypeError('DiffFunction output not implemented as a DiffObject')
    
    fd = finite_differences #alias for convenience

class DiffFunctionByList(DiffFunctionBase):
    def __init__(self, func, request_derivatives=True):#, **kdfun):
        self.func = func
        self.request_derivatives = request_derivatives
        #functools.update_wrapper(self, fun)
        #self.__name__ = fun.__name__
        #self.__doc__ = fun.__doc__
        
    def __call__(self, *args, **kwargs):
        compute_derivatives = [isinstance(arg, DiffObject) for arg in args]
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs #TODO: for now can not diff wrt kwargs       

        if self.request_derivatives:        
            f, df = self.func(*argvalues, **kwargvalues, compute_derivatives=compute_derivatives)
        else:
            f, df = self.func(*argvalues, **kwargvalues)
        
        if not any(compute_derivatives):
            return f
            
        #try to make DiffObject
        if type(f) in DiffObject._types:
            dlist = [arg.chain(dfi) for arg, dfi in zip(args, df) if isinstance(arg, DiffObject)]
            d = sum_dicts(*dlist)
            return DiffObject(f, d)
        elif isinstance(f, Iterable):
            dlist = [[arg.chain(dfij) for dfij in dfi] for arg, dfi in zip(args, df) if isinstance(arg, DiffObject)]
            d = [sum_dicts(*d) for d in zip(*dlist)]
            return type(f)(map(DiffObject, f, d))
            
        raise TypeError('DiffFunction output not implemented as a DiffObject')

class DiffFunction(DiffFunctionBase):
    def __init__(self, func, dfunc=[]):#, **kdfun):
        self.func = func
        #functools.update_wrapper(self, fun)
        #self.__name__ = fun.__name__
        #self.__doc__ = fun.__doc__
        
        self.set_derivatives(dfunc)
        #self.kdfun = kdfun
    
    def set_derivatives(self, dfunc):
        self.dfunc = list(dfunc)
    
    def set_derivative(self, index, dfunc):
        if index < len(self.dfunc):
            self.dfunc[index] = dfunc

        self.dfunc += [NoneFunction]*(index-len(self.dfunc)) + [dfunc]
           
    def derivative(self, index=0):
        '''
        Provides a convenient way to define derivatives by decorating functions
        as @function.derivative(index)
        '''
        def derivative_decorator(dfunc):
            self.set_derivative(index, dfunc)
            return self
        return derivative_decorator
    
    #TODO: use itertools for lazy evaluation and memory efficiency
    def __call__(self, *args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs #TODO: for now can not diff wrt kwargs       
        
        #? should I check is all derivatives are provided?
        #? provide option for numerically computed derivative if not defined?
        f = self.func(*argvalues, **kwargvalues)
        
        if not any([isinstance(arg, DiffObject) for arg in args]):
            return f
            
            #compute df_args
        df = [self.dfunc[i](*argvalues, **kwargvalues) \
             if isinstance(arg, DiffObject) else None \
             for i, arg in enumerate(args)]
            
        #try to make DiffObject
        if type(f) in DiffObject._types:
            dlist = [arg.chain(dfi) for arg, dfi in zip(args, df) if isinstance(arg, DiffObject)]
            d = sum_dicts(*dlist)
            return DiffObject(f, d)
        elif isinstance(f, Iterable):
            dlist = [[arg.chain(dfij) for dfij in dfi] for arg, dfi in zip(args, df) if isinstance(arg, DiffObject)]
            d = [sum_dicts(*d) for d in zip(*dlist)]
            return type(f)(map(DiffObject, f, d))
            
        raise TypeError('DiffFunction output not implemented as a DiffObject')

#class DiffMethod(DiffFunction):    
#    #Makes DiffMethod a non-data descriptor that binds its __call__ method to
#    #the particular instance that calls it
#    def __get__(self, instance, cls):
#        return functools.partial(DiffFunction.__call__, self, instance) 
#        #return types.MethodType(self, instance)

dfunction = DiffFunctionBase #alias

##HACK
#class DictWrapper(object):
#    def __init__(self,d):
#        self.__dict__ = d
##? Alternative implementation?
#def dfunction(fun, *dfun):
#    self = DictWrapper(locals())
#    
#    def wrapped(*args, **kwargs):
#        return DiffFunction.__call__.im_func(self, *args, **kwargs)
#    try:
#        return functools.wraps(fun)(wrapped)
#    except:
#        return wrapped

class ConstFunction(object):
    def __init__(self, func):#, **kdfun):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs #TODO: for now can not diff wrt kwargs
        
        return self.func(*argvalues, **kwargvalues)
        
    def __get__(self, instance, cls):
        #return functools.partial(DiffFunction.__call__, self, instance) 
        return types.MethodType(self, instance)
        
#function wrapper for piecewise constant functions
#the wrapped function will return its normal output (not a DiffObject) with
#all arguments replaced by their values in case these
def cfunction(func):
    '''
    Decorator for piecewise constant functions. The wrapped function will 
    work on DiffObjects arguments by replacing these with their values.
    The output is the regular function output, not a DiffObject
    ''' 
    def wrapped(*args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs
        
        return func(*argvalues, **kwargvalues)
    return wrapped
    
