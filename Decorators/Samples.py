# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:32:51 2021

@author: egoral
"""

import functools as ft

#  @lru_cache
def factorial(n):
    return n * factorial (n-1) if n else 1

print(factorial (10))