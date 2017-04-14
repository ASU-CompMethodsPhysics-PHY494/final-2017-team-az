# Population Dynamics final project - team AZ

import numpy as np
import matplotlib.pyplot as plt

#==============================================
# Ecosystem being modeled: Plants => Insect => Horned lizard => Western Diamondback Rattlesnake => Hawk
#----------------------------------------------

names = np.array(['cactus', 'ants', 'horned lizard', 'rattlesanke', 'hawk']) # For now we can try and model a linear food chain
animals = np.arange(len(names))

#==============================================
# Initial Conditions
#----------------------------------------------

'''
To team AZ,

I've intentionally left this area blank for now. Normally, different
constants are used for each predator prey interaction and I planned to
put all those constants here but seeing as how we currently have none I'll
leave this section unfilled. I did however use a single set of constants
for the Ant/Horned-lizard interaction examined below. These constnats are
numbers I found on the internet an likely do not reflect reality. I directly
put these equations int the Lokta-Volterra eqautions for now but that will
have to change once more constants are determined. My intention was simply
 to get an idea for what the graphs should look like.
'''

#==============================================
# Empty population list for later filling
#----------------------------------------------

cactus         = []
ant            = []
horned_lizard  = []
rattlesnake    = []
hawk           = []

#==============================================
# Integrators
#----------------------------------------------

def euler_prey(x, prey, y, h):
    """Euler integrator.
    Specifically for prey.

    Returns new x at t+h.
    """
    return x + h * prey(y, x)

def euler_pred(y, pred, x, h):
    """Euler integrator.
    Specifically for predators.

    Returns new y at t+h.
    """
    return y + h * pred(x, y)


def rk4_prey(x, prey, y, h):
    """Runge-Kutta RK4
    Specifically for prey

    Returns new x at t+h
    """
    k1 = prey(y, x)
    k2 = prey(y + 0.5*h, x + 0.5*h*k1)
    k3 = prey(y + 0.5*h, x + 0.5*h*k2)
    k4 = prey(y + h, x + h*k3)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)

def rk4_pred(y, pred, x, h):
    """Runge-Kutta RK4
    Specifically for predators

    Returns new y at t+h
    """
    k1 = pred(x, y)
    k2 = pred(x + 0.5*h, y + 0.5*h*k1)
    k3 = pred(x + 0.5*h, y + 0.5*h*k2)
    k4 = pred(x + h, y + h*k3)
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

#==============================================
# Lotka-Volterrs functions without competition
#----------------------------------------------

def prey(y, x):
    '''Lotka-Volterra prey equation.
    Parameters
    ----------
    y: predator population
    x: prey population
    Returns
    ----------
    Updated prey population
    '''
    return 0.1*x - 0.02*x*y

def pred(x, y):
    '''Lotka-Volterra predator equation.
    Parameters
    ----------
    x: prey population
    y: predator population
    Returns
    ----------
    Updated predator population
    '''
    return 0.02*x*y - 0.4*y

#==============================================
# Predator/Prey population array filling
#----------------------------------------------

def integrate_euler(Nmax):
    '''Integrates one pair of pred/prey using Eulers method.
    Parameters
    ----------
    Nmax: number of iterations
    Returns
    ----------
    Prey population array
    Predator population array
    '''
    h = 0.01
    t = 0
    x = 100
    y = 100
    for i in range(Nmax):
        x = euler_prey(x, prey, y, h)
        ant.append([t, x])
        y = euler_pred(y, pred, x, h)
        horned_lizard.append([t, y])
        t += h
    return np.array(ant), np.array(horned_lizard)


def integrate_rk4(Nmax):
    '''Integrates one pair of pred/prey using RK4 method.
    Parameters
    ----------
    Nmax: number of iterations
    Returns
    ----------
    Prey population array
    Predator population array
    '''
    h = 0.01
    t = 0
    x = 100
    y = 100
    for i in range(Nmax):
        x = rk4_prey(x, prey, y, h)
        ant.append([t, x])
        y = rk4_pred(y, pred, x, h)
        horned_lizard.append([t, y])
        t += h
    return np.array(ant), np.array(horned_lizard)

#==============================================
# Population vs Time plotting
#----------------------------------------------

def graph(Nmax, Integrator = rk4):
    '''Plots ant and horned-lizard populations using RK4 or Eulers method
    Parameters
    ----------
    Nmax:        number of integration iterations
    Integrator:  which integrator to use
    Returns
    ----------
    Graph of ant and horned-lizard populations vs time
    '''
    if Integrator == rk4:
        ant, horn = integrate_rk4(Nmax)
        plt.plot(ant[:,0], ant[:,1], label = 'Ant population (prey)')
        plt.plot(horn[:,0], horn[:,1], label = 'Horned-lizard population (predator)')
        plt.title('Predator and Prey Populations')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend(loc = 'best')
        plt.show()
    else:
        ant, horn = integrate_euler(Nmax)
        plt.plot(ant[:,0], ant[:,1], label = 'Ant population (prey)')
        plt.plot(horn[:,0], horn[:,1], label = 'Horned-lizard population (predator)')
        plt.title('Predator and Prey Populations')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend(loc = 'best')
        plt.show()

def graph1(Nmax, prey = True, Integrator = rk4):
    '''Plots predator or prey populations using RK4 or Eulers method
    Parameters
    ----------
    Nmax:        number of integration iterations
    prey:        determines prey or predator
    Integrator:  which integrator to use
    Returns
    ----------
    Graph of predator or prey populations vs time
    '''
    if prey == True:
        if Integrator == rk4:
            x, y = integrate_rk4(Nmax)
            plt.plot(x[:,0], x[:,1], label = 'prey population')
            plt.xlabel('Time')
            plt.ylabel('Population')
            plt.legend(loc='best')
            plt.show()
        else:
            x, y = integrate_euler(Nmax)
            plt.plot(x[:,0], x[:,1], label = 'prey population')
            plt.xlabel('Time')
            plt.ylabel('Population')
            plt.legend(loc='best')
            plt.show()
    else:
        if Integrator == rk4:
            x, y = integrate_rk4(Nmax)
            plt.plot(y[:,0], y[:,1], label = 'predator population')
            plt.xlabel('Time')
            plt.ylabel('Population')
            plt.legend(loc='best')
            plt.show()
        else:
            x, y = integrate_euler(Nmax)
            plt.plot(y[:,0], y[:,1], label = 'predator population')
            plt.xlabel('Time')
            plt.ylabel('Population')
            plt.legend(loc='best')
            plt.show()
