#%%
##Memoization
##Calculating a recursive function using cache using:
##Method 1 - class
##Method 2 - decorator
#%%
#Method 1 - using class
class Fibo:
    def __init__(self):
        self.cache = {1:1, 2:1}

    def fib(self,n):
        if n not in self.cache:
            print(f'Calcualating for {n}')
            self.cache[n] = self.fib(n-1) + self.fib(n-2)
        return self.cache[n]
#%%
f = Fibo()
f.fib(10)
#%%
#Method 2 - using decrator
def memoize(fib): 
    fib_cache = dict()
    def inner(n):
        if n not in fib_cache:
            fib_cache[n] = fib(n)
        return fib_cache[n] 
        
    return inner
        
#%%
@memoize
def fib(n):
    print(f'Calculating fib({n})')
    return 1 if n < 3 else fib(n-1) + fib(n-2)
#%%
fib(10)
#%%
@memoize
def fact(n):
    print(f'Calulating fact({n})')
    return 1 if n<2 else n * fact(n-1)
#%%
fact(6)
#%%
##Method 3: using Python's built-in lru_cache
from functools import lru_cache

@lru_cache(maxsize=10)
def fib(n):
    print(f'Calculationg fib({n})')
    return 1 if n < 3 else fib(n-1) + fib(n-2)
#%%
fib(10)
#%%
fib(11)
#%%
fib(1)
#%%
fib(5)
#%%
