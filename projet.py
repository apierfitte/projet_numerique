# Third-Party Libraries
# ---------------------

# Autograd & Numpy
import autograd
import autograd.numpy as np

# Pandas
import pandas as pd

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10] # [width, height] (inches). 

# Jupyter & IPython
from IPython.display import display

def grad(f):
    g = autograd.grad
    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
    return grad_f

def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f

def f1(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2 

N = 100
eps = 10**(-3)

# On a, d'après la méthode de Newton :
# x_k+1 = x_k - Jf(x_k)^-1 * f(x_k)
# Ce qui donne :
# f(x_k) = Jf(x_k)(x_k - x_k+1)
#On reconnaît un système linéaire

def Newton(F, x0, y0, eps=eps, N=N): 
    for i in range(N):
        # on veut résoudre un système de la forme AX=B
        A = J(F)(x0, y0)
        B = F(x0, y0)
        X = np.linalg.solve(A, B)
        x, y = x0 - X[0], y0 - X[1]
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return x, y
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")

def g1(x1, x2):
    return np.array([f1(x1, x2) - 0.8 , x1 - x2])

x, y = Newton(g1, 0.8, 0.8)

def fonction_level_curve(f, x0, y0, c, delta):
    def g_f(x, y):
        return(np.array([f(x, y) - c, (x - x0)**2 + (y - y0)**2 - delta**2]))
    return (g_f)


def level_curve(f, x0, y0, delta=0.1, N=1000, eps=eps):
    res = np.empty((2, N), dtype=float)
    c = f(x0, y0)
    for i in range(N) :
        gradient = grad(f)(x0, y0)
        # on se place à un nouveau point "dans le sens de grad(f)"
        nouveau_point = np.array([x0, y0]) + delta * gradient
        # on trouve les nouvelles coordonnées, sur le cercle de centre (x0, y0) et de rayon delta
        x , y = Newton(fonction_level_curve(f, x0, y0, c, delta), nouveau_point[0], nouveau_point[1])
        res[0][i], res[1][i] = x, y
        x0, y0 = x, y
    return res

tableau = level_curve(f1, 0.5, 0.8)

plt.plot(tableau[0], tableau[1])
plt.show()