import time

def timer(func):
    def func2(*a):
        t1 = time.time()
        r = func(*a)
        t2 = time.time()
        print(' '*20, f'{func.__name__} took {t2-t1:.3f} s')
        return r
    return func2
