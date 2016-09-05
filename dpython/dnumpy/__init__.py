# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 18:07:11 2016

@author: Ajoo
"""

from .dnumpy import *
from .ufuncs import *

#import testing

#from .. import numpysupport

print('Initializing autodiff.numpy...')

__all__ = diffnumpy.__all__ + ufuncs.__all__

__version__ = '0.0.0'
__author__ = u'João Ferreira <ajoo@outlook.pt>'


if __name__ == "__main__":
    pass