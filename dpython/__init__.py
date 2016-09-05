# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 17:31:30 2016

@author: Ajoo
"""

from .autodiff import *
from .dfloat import *

print('Initializing dpy...')

__all__ = autodiff.__all__ + dfloat.__all__

__version__ = '0.0.0'
__author__ = u'João Ferreira <ajoo@outlook.pt>'


if __name__ == "__main__":
    pass