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

tableau = level_curve(f1, 0.3, 0.8)
plt.plot(tableau[0], tableau[1])
plt.show()


# renvoie la position du point c par rapport à la droite AB
# 1 <=> c est "au-dessus" de la droite 
# -1 <=> c est "en-dessous"
# a,b et c sont colinéaires
# a,b et c sont supposés différents
def side(a,b,c):
    d = (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    return 1 if d > 0 else (-1 if d < 0 else 0)

# renvoie True si c est à l'intérieur du segment [a, b], False sinon
# les points a, b et c sont supposés colinéaires
def is_point_in_closed_segment(a, b, c):
    """ Returns True if c is inside closed segment, False otherwise.
        a, b, c are expected to be collinear
    """
    if a[0] < b[0]:
        return a[0] <= c[0] and c[0] <= b[0]
    if b[0] < a[0]:
        return b[0] <= c[0] and c[0] <= a[0]

    if a[1] < b[1]:
        return a[1] <= c[1] and c[1] <= b[1]
    if b[1] < a[1]:
        return b[1] <= c[1] and c[1] <= a[1]

    return a[0] == c[0] and a[1] == c[1]

# Vérifie si les segments [a, b] et [c, d] s'intersectent
def closed_segment_intersect(a,b,c,d):
    if a == b:
        return a == c or a == d
    if c == d:
        return c == a or c == b

    s1 = side(a,b,c)
    s2 = side(a,b,d)

    # Tous les points sont colinéaires
    if s1 == 0 and s2 == 0:
        return \
            is_point_in_closed_segment(a, b, c) or is_point_in_closed_segment(a, b, d) or \
            is_point_in_closed_segment(c, d, a) or is_point_in_closed_segment(c, d, b)

    # Ils ne se touchent pas et c et d sont du même côté de la droite AB
    if s1 and s1 == s2:
        return False

    s1 = side(c,d,a)
    s2 = side(c,d,b)

    # idem
    if s1 and s1 == s2:
        return False

    # arrivé ici, les segments s'intersectent
    return True

def level_curve_question_7(f, x0, y0, delta=0.1, N=1000, eps=eps):
    res = np.empty((2, N), dtype=float)
    c = f(x0, y0)
    # on calcule les bornes du premier segment
    gradient = grad(f)(x0, y0)
    nouveau_point = np.array([x0, y0]) + delta * gradient
    a , b = Newton(fonction_level_curve(f, x0, y0, c, delta), nouveau_point[0], nouveau_point[1])
    x0, y0 = a, b
    for i in range(1, N) :
        gradient = grad(f)(x0, y0)
        # on se place à un nouveau point "dans le sens de grad(f)"
        nouveau_point = np.array([x0, y0]) + delta * gradient
        # on trouve les nouvelles coordonnées, sur le cercle de centre (x0, y0) et de rayon delta
        x , y = Newton(fonction_level_curve(f, x0, y0, c, delta), nouveau_point[0], nouveau_point[1])
        # test d'auto-intersection
        if closed_segment_intersect(a, b, x, y) :
            print("Auto-intersection")
            return res[::,:i]

        res[0][i], res[1][i] = x, y
        x0, y0 = x, y
    return res