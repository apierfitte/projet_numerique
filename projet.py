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

def f2(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2

def f3(x, y):
    return np.sin(x + y) - np.cos(x * y) - 1 + 0.001 * (x * x + y * y) 

N = 1000
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

# on définit la matrice de rotation
mat_rot = np.array(( (0, 1),
                   (-1, 0) )) # matrice de rotation de -pi/2

def level_curve(f, x0, y0, delta=0.1, N=N, eps=eps):
    res = np.empty((2, N), dtype=float)
    c = f(x0, y0)
    res[:,0] = np.array([x0, y0])
    # création du premier point
    for i in range(1, N) :
        gradient = grad(f)(x0, y0)
        # on se place à un nouveau point
        u = mat_rot.dot(gradient)
        norme_u = np.linalg.norm(u)
        nouveau_point = np.array([x0, y0]) + (delta/norme_u) * u
        # on trouve les nouvelles coordonnées, sur le cercle de centre (x0, y0) et de rayon delta
        x , y = Newton(fonction_level_curve(f, x0, y0, c, delta), nouveau_point[0], nouveau_point[1])
        res[:,i] = np.array([x, y])
        x0, y0 = x, y
    return res


# renvoie la position du point c par rapport à la droite AB
# 1 <=> c est "au-dessus" de la droite 
# -1 <=> c est "en-dessous"
# 0 <=> a,b et c sont colinéaires
# a,b et c sont supposés différents
def side(a,b,c):
    d = (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    return 1 if d > 0 else (-1 if d < 0 else 0)

# renvoie True si c est à l'intérieur du segment [a, b], False sinon
# les points a, b et c sont supposés colinéaires
def is_point_in_closed_segment(a, b, c):
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
    if (a == b).all():
        return (a == c).all() or (a == d).all()
    if (c == d).all():
        return (c == a).all() or (c == b).all()

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

def level_curve_question_7(f, x0, y0, delta=0.1, N=N, eps=eps):
    res = np.empty((2, N), dtype=float)
    c = f(x0, y0)
    # on calcule les bornes du premier segment
    a = np.array([x0, y0])
    res[:,0] = a
    gradient = grad(f)(x0, y0)
    # On calcule le nouveau point
    u = mat_rot.dot(gradient)
    norme_u = np.linalg.norm(u)
    nouveau_point = np.array([x0, y0]) + (delta/norme_u) * u
    x , y = Newton(fonction_level_curve(f, x0, y0, c, delta), nouveau_point[0], nouveau_point[1])
    b = np.array([x, y])
    res[:,1] = b
    x0, y0 = x, y

    # on calcule un troisième point
    gradient = grad(f)(x0, y0)
    u = mat_rot.dot(gradient)
    norme_u = np.linalg.norm(u)
    nouveau_point = np.array([x0, y0]) + (delta/norme_u) * u
    x , y = Newton(fonction_level_curve(f, x0, y0, c, delta), nouveau_point[0], nouveau_point[1])
    res[:,2] = np.array([x, y])
    x0, y0 = x, y

    # puis on calcule les suivants
    for i in range(3, N) :
        gradient = grad(f)(x0, y0)
        # on se place à un nouveau point "dans le sens de grad(f)"
        u = mat_rot.dot(gradient)
        norme_u = np.linalg.norm(u)
        nouveau_point = np.array([x0, y0]) + (delta/norme_u) * u
        # on trouve les nouvelles coordonnées, sur le cercle de centre (x0, y0) et de rayon delta
        x , y = Newton(fonction_level_curve(f, x0, y0, c, delta), nouveau_point[0], nouveau_point[1])
        res[:,i] = np.array([x, y])
        # test d'auto-intersection
        if closed_segment_intersect(a, b, res[:,i-1], np.array([x, y])) :
            print(f"Auto-intersection après {i} points.")
            return res[:,:i]
        x0, y0 = x, y
    return res



'''
tache 6
'''

def gamma(t,P1,P2,u1,u2):

    det1 = u1[0]*u2[1] - u2[0]*u1[1]
    det2 = u2[1]*(P2[0]-P1[0]) - u2[0]*(P2[1]-P1[1])
    alpha = (2*det2) / det1
    
    if det1 == 0 or det2 == 0:
        #on fait un chemin lineaire quand les conditions ne sont pas respectées
        xt = (P2[0] - P1[0])*t + P1[0]
        yt = (P2[1]-P1[1])*t + P1[1]
        return np.array([xt,yt]).reshape(2,len(t))
        
    else:
        #on implémente la fonction gamma si les conditions sont respectées
        a = P1[0]
        b = alpha * u1[0]
        c = P2[0] - P1[0] - alpha*u1[0]
        d = P1[1]
        e = alpha*u1[1]
        f = P2[1]-P1[1] - alpha*u1[1]
        
        return np.array((a + b*t + c*(t**2), d +e*t + f*(t**2))).reshape(2,len(t))

# cette fonction marche pour des valeurs vectorielles de t

'''
tache 7
'''

def level_curve_question_8(f, x0, y0, oversampling=1, delta=0.1, N=N, eps=eps):
    if (oversampling == 1):
        return(level_curve_question_7(f, x0, y0))
    else : # on utilise le squelette de la fonction level_curve
        res = np.empty((2, oversampling*N), dtype=float)
        c = f(x0, y0)
        # on calcule les bornes du premier segment
        a = np.array([x0, y0])
        res[:,0] = a
        gradient = grad(f)(x0, y0)
        # On calcule le nouveau point
        u = mat_rot.dot(gradient)
        norme_u = np.linalg.norm(u)
        nouveau_point = np.array([x0, y0]) + (delta/norme_u) * u
        x , y = Newton(fonction_level_curve(f, x0, y0, c, delta), nouveau_point[0], nouveau_point[1])
        b = np.array([x, y])
        res[:,1*oversampling] = b
        x0, y0 = x, y

        # puis on "remplit" entre a et b 
        res[:,1:oversampling] = interpolation(f, a, b, oversampling)

        # on calcule un troisième point
        gradient = grad(f)(x0, y0)
        u = mat_rot.dot(gradient)
        norme_u = np.linalg.norm(u)
        nouveau_point = np.array([x0, y0]) + (delta/norme_u) * u
        x , y = Newton(fonction_level_curve(f, x0, y0, c, delta), nouveau_point[0], nouveau_point[1])
        res[:,2*oversampling] = np.array([x, y])
        x0, y0 = x, y
        
        # on remplit entre b et ce nouveau point
        res[:,1*oversampling+1:2*oversampling] = interpolation(f, b, np.array([x,y]), oversampling)

        # puis on calcule les suivants
        for i in range(3, N) :
            gradient = grad(f)(x0, y0)
            # on se place à un nouveau point "dans le sens de grad(f)"
            u = mat_rot.dot(gradient)
            norme_u = np.linalg.norm(u)
            nouveau_point = np.array([x0, y0]) + (delta/norme_u) * u
            # on trouve les nouvelles coordonnées, sur le cercle de centre (x0, y0) et de rayon delta
            x , y = Newton(fonction_level_curve(f, x0, y0, c, delta), nouveau_point[0], nouveau_point[1])
            res[:,i*oversampling] = np.array([x, y])
            # on remplit res entre ce nouveau point et celui trouvé à l'itération précédente
            res[:,(i-1)*oversampling+1:i*oversampling] = interpolation(f, res[:,(i-1)*oversampling], np.array([x,y]), oversampling)
            # test d'auto-intersection
            if closed_segment_intersect(a, b, res[:,(i-1)*oversampling], np.array([x, y])) :
                print(f"Auto-intersection après {i} points.")
                return res[:,:i*oversampling]
            x0, y0 = x, y
        return res


'''
la fonction suivante retourne un array de dimension (2, oversampling) avec les points d'interpolation entre a et b
'''
def interpolation(f, P1, P2, oversampling):
    u1 = mat_rot.dot(grad(f)(P1[0], P1[1])) #on "suit" le chemin emprunté par la fonction level_curve
    u2 = mat_rot.dot(grad(f)(P2[0], P2[1]))
    t = np.linspace(0, 1, oversampling, False)
    return gamma(t, P1, P2, u1, u2)[:,1:]

tableau1 = level_curve_question_8(f3, -0.3, 0.5, 100)

plt.plot(tableau1[0], tableau1[1], color='red')

plt.show()

# t = np.linspace(0, 1, 1000)
# P1 = np.array([0.2, 0.3])
# P2 = np.array([-0.4, 0.1])
# u1 = grad(f1)(P1[0], P1[1])
# u2 = grad(f1)(P2[0],P2[1])

# res = interpolation(f1, P1, P2, 100)

# print(res)
# print(res.shape)

# plt.plot(res[0], res[1])
# plt.show()


'''
changer question 7 !!!
'''