import taichi as ti
ti.init()

@ti.data_oriented
class R():
    def __init__(self, a=0, b=None, c=None):
        self.a=a
        self.b=b
        self.c=c

@ti.kernel
def f():
    x = g()
    print(x.a, x.b)
    
@ti.func
def g():
    x = R(a=1)
    return x


f()