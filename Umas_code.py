
# Population Dynamics final project - team AZ

import numpy as np
import matplotlib.pyplot as plt

#==============================================
# Input:
#h: change in time
#the populations of each animal:
    #c-cactus, d-dove, s-snake, o-owl, f-fox
#funct: uses Lotka-Volterra equation for each animal

#Output:
#the Euler method of finding the population after change in time, h.
#----------------------------------------------
def integrate_rk4(Nmax):
    y = np.array([2, 2, 2, 2, 2, 2, 2])
    h = 0.01
    t = 0
    populations = []

    def f(t, y):
        c = y[0] - y[0]*y[1]
        d = y[1]*y[0] - y[1] - y[1]*y[2] - y[1]*y[3] - y[1]*y[4]
        s = y[2]*y[1] - y[2] - y[5]*y[2] - y[6]*y[2]
        f = y[3]*y[1] - y[3] - y[5]*y[3] - y[6]*y[3]
        o = y[4]*y[1] - y[4] - y[5]*y[4] - y[6]*y[4]
        p = y[5]*y[2] + y[5]*y[3] + y[5]*y[4] - y[5]
        e = y[6]*y[2] + y[6]*y[3] + y[6]*y[4] - y[6]
        return np.array([c, d, s, f, o, p, e])

    for i in range(Nmax):
        populations.append([t, y[0], y[1], y[2], y[3], y[4], y[5], y[6]])
        y = rk4(y, f, t, h)
        t += h

    return np.array(populations)

'''def test(Nmax, a, b):
    h = 0.1
    for i in range(0, Nmax):
        for j in range(0, Nmax):
            x = integrate_rk4(Nmax, a+h*i, b-h*j)
            if np.average(x[Nmax-100:Nmax,3]) > 1 and np.average(x[Nmax-100:Nmax,4]) > 1 and np.average(x[Nmax-100:Nmax, 5]):
                print("Survives through ", Nmax, "iterations at a= ", a+h*i, "final pop =", 
                x[-1,3], "b =", b-h*j)
                return x
            else:
                print("iterations =", j+1)
    print("Could not find parameter in", Nmax, "iterations. Final a =", a+h*i, "final pop =", x[-1,3], 
    "b = ", b-h*j)
'''

def graph_rk4(Nmax):
    x = integrate_rk4(Nmax)
    plt.plot(x[:,0], x[:,1], label = 'cactus')
    plt.plot(x[:,0], x[:,2], label = 'dove')
    plt.plot(x[:,0], x[:,3], label = 'snake')
    plt.plot(x[:,0], x[:,4], label = 'fox')
    plt.plot(x[:,0], x[:,5], label = 'owl')
    plt.plot(x[:,0], x[:,6], label = 'puma')
    plt.plot(x[:,0], x[:,7], label = 'eagle')
    plt.xlabel('time')
    plt.ylabel('Population density')
    plt.legend(loc = 'best')
    plt.show()

def euler_cactus(h, c, d, s, o, f, cactus):
    """Euler integrator.
    Specifically for prey.
    Returns new x at t+h.
    """
    return c + h * cactus(c, d, s, o, f)

def euler_dove(h, c, d, s, o, f, dove):
    return d + h * dove(c, d, s, o, f)

def euler_snake(h, c, d, s, o, f, snake):
    return s + h * snake(c, d, s, o, f)

def euler_owl(h, c, d, s, o, f, owl):
    return o + h * owl(c, d, s, o, f)

def euler_fox(h, c, d, s, o, f, fox):
    return f + h * fox(c, d, s, o, f)

def rk4(y, f, t, h):
    """Runge-Kutta RK4
    Returns new y at t+h
    """
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

#==============================================
# Lotka-Volterrs functions with built in competition
#----------------------------------------------

def cactus(c, d, s, o, f):
    '''Lotka-Volterra cactus equation.
    Parameters
    ----------
    c, d, s, o, f: current population of all the animals
    alpha:         natural growth rate of cactus
    beta:          rate of death due to dove predation

    Returns
    ----------
    Updated cactus population
    '''
    alpha=beta=1
    return (alpha - beta*d)*c

def dove(c, d, s, o, f):
    '''Lotka-Volterra predator equation.
    Parameters
    ----------
    c, d, s, o, f: current population of all the animals
    alpha:         natural growth rate of doves
    beta:          natural death rate of doves
    gamma:         rate of death due to snake predation
    rho:           rate of death due to fox predation
    mu:            rate of death due to owl predation

    Returns
    ----------
    Updated dove population
    '''
    alpha=beta=gamma=rho=mu=1
    return (-alpha + beta*c - gamma*s - rho*f - mu*o)*d

def snake(c, d, s, o, f):
    '''Lotka-Volterra predator equation.
    Parameters
    ----------
    c, d, s, o, f:  current population of all the animals
    alpha:          natural growth rate
    beta:           natural death rate
    gamma:          rate of death due to fox predation

    Returns
    ----------
    Updated snake population
    '''
    alpha=1
    beta=1
    gamma = 0.1
    return (-alpha*s + beta*d*s) - gamma*f*s

def owl(c, d, s, o, f):
    '''Lotka-Volterra predator equation.
    Parameters
    ----------
    c, d, s, o, f: current population of all the animals
    Returns
  ----------
    Updated owl population
    '''
    alpha=1
    beta=1
    return (-alpha + beta*d)*o

def fox(c, d, s, o, f):
    '''Lotka-Volterra predator equation.
    Parameters
    ----------
    c, d, s, o, f: current population of all the animals
    Returns
    ----------
    Updated fox population
    '''
    alpha=1
    gamma=1
    beta = 1
    return (-alpha*f + f*gamma*d) + f*beta*s


#==============================================
# Population array filling
#----------------------------------------------
time_list=([])
cactus_list= ([])#([])
dove_list=([])
snake_list=([])
owl_list=([])
fox_list=([])

def integrate_euler(Nmax):
    '''Integrates all of the populations using Eulers method.
    Parameters
    ----------
    Nmax: number of iterations
    Returns
    ----------
    array of time points
    all population arrays
    '''
    h = 0.01
    t = 0
    c = 2
    d = 2
    s = 2
    o = 2
    f = 2

    for i in range(Nmax):
        c = euler_cactus(h, c, d, s, o, f, cactus)
        cactus_list.append([c])
        d = euler_dove(h, c, d, s, o, f, dove)
        dove_list.append([d])
        s = euler_snake(h, c, d, s, o, f, snake)
        snake_list.append([s])
        o = euler_owl(h, c, d, s, o, f, owl)
        owl_list.append([o])
        f = euler_fox(h, c, d, s, o, f, fox)
        fox_list.append([f])
        t += h
        time_list.append([t])

    return np.array(time_list), np.array(cactus_list), np.array(dove_list), np.array(snake_list), np.array(owl_list), np.array(fox_list)


#==============================================
# Plot population functions with respect to time
#----------------------------------------------
def graph_euler(Nmax):
    x=integrate_euler(Nmax)
    plt.plot(x[0], x[1], label = 'cactus')
    plt.plot(x[0], x[2], label = 'dove')
    plt.plot(x[0], x[3], label = 'snake')
    plt.plot(x[0], x[4], label = 'owl')
    plt.plot(x[0], x[5], label = 'fox')
    plt.xlabel('time')
    plt.ylabel('population density')
    plt.legend(loc = 'best')
    plt.show()

#==============================================
# Plot population functions with respect to each other
#can exchange any element of x to look at any
#two populations (x can be 1,2,3,4 or 5)
#not all of the combinations look great
#----------------------------------------------
# x=integrate_euler(1000)
# plt.plot(x[1], x[2], label = 'cactus vs dove')
# plt.xlabel('cactus popultaion')
# plt.ylabel('dove population')
# plt.legend(loc = 'best')
# plt.show()
