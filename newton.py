
import math
def newtons_method(f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        x0 = x0 - f(x0)/df(x0)
        delta = dx(f, x0)
    print('Root is at: ', x0)
    print('f(x) at root is: ', f(x0))


def f(x):
    return math.sqrt(x*x-4)-20+x*x*x*x-4*x*x*x

def df(x):
    return x/math.sqrt(x*x-4) + 4*x*x*x-12*x*x*x*x


def dx(f, x):
    return abs(0-f(x))

newtons_method(f,df,4.5,1e-5)
