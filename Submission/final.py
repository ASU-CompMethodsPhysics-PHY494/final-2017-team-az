# Population Dynamics final project - team AZ

import numpy as np
import matplotlib.pyplot as plt

#==============================================
# Integrators
#----------------------------------------------

def rk4(y, f, t, h):
    """Runge-Kutta RK4
    Returns new y at t+h
    """
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

def euler(y, f, t, h):
    '''Eulers method for integration
    Returns new y at t+h
    '''
    return y + h*f(t, y)

#==============================================
# Population density array filling
#----------------------------------------------

def integrate(Nmax, h = 0.01, integrator = rk4):
    '''Integrates the Lotka-Volterra equations

    Parameters
    -----------
    Nmax: The number of iterations performed

    Returns
    -----------
    An array with populations of all 7 animals over time
    '''
    # Initial population densities of the five animals
    c0 = 9.57     #initial cactus population density
    d0 = 0.12     #initial dove population density
    s0 = 0.12     #initial rattlesnake population density
    f0 = 0.15     #initial fox population density
    o0 = 0.105     #initial owl population density
    p0 = 0.024     #initial puma population density
    e0 = 0.024     #initial eagle population density

    # Lotka-Volterra constants
    alpha_c    = 1     #natural growth rate of cactus
    beta_c     = 1     #death rate of cactus due to dove predation

    alpha_d    = 1     #predative success of doves on cacti
    beta_d     = 1     #natural death rate of doves in the absence of other predators or prey
    gamma_d    = 1     #death rate of doves due to snake predation
    delta_d    = 1     #death rate of doves due to fox predation
    epsilon_d  = 1     #death rate of doves due to owl predation

    alpha_s    = 1     #predative success of snakes on doves
    beta_s     = 1     #natural death rate of snakes in the absence of other predators or prey
    gamma_s    = 0.99     #death rate of snakes due to puma predation
    delta_s    = 0.99     #death rate of snakes due to eagle predation

    alpha_f    = 1     #predative success of foxes on doves
    beta_f    = 1     #natural death rate of foxes in the absence of other predators or prey
    gamma_f    = 0.97     #death rate of foxes due to puma predation
    delta_f    = 0.97     #death rate of foxes due to eagle predation

    alpha_o    = 1     #predative success of owls on doves
    beta_o     = 1     #natural death rate of owls in the absence of other predators or prey
    gamma_o    = 0.95     #death rate of owls due to puma predation
    delta_o    = 0.95     #death rate of owls due to eagle predation

    alpha_p    = 0.3     #predative success of pumas on snakes
    beta_p     = 0.3     #predative success of pumas on foxes
    gamma_p    = 0.3     #predative success of pumas on owls
    delta_p    = 0.3     #natural death rate of pumas in the absence of prey

    alpha_e    = 0.3     #predative success of eagles on snakes
    beta_e     = 0.3     #predative success of eagles on foxes
    gamma_e    = 0.3     #predative success of eagles on owls
    delta_e    = 0.3    #natural death rate of eagles in the absence of prey


    y = np.array([c0, d0, s0, f0, o0])
    t = 0
    populations = []


    def f(t, y):
        '''
        Creates an array with time derivatives of all five populations

        Parameters
        -------------
        t: time
        y: an array with the initial population values

        Returns
        -------------
        an array with the time derivatives of the population densities
        '''
        c = alpha_c*y[0] - beta_c*y[0]*y[1]
        d = alpha_d*y[0]*y[1] - beta_d*y[1] - gamma_d*y[1]*y[2] - delta_d*y[1]*y[3] - epsilon_d*y[1]*y[4]
        s = alpha_s*y[1]*y[2] - beta_s*y[2] #- gamma_s*y[2]*y[5] - delta_s*y[2]*y[6]
        f = alpha_f*y[1]*y[3] - beta_f*y[3] #- gamma_f*y[3]*y[5] - delta_f*y[3]*y[6]
        o = alpha_o*y[1]*y[4] - beta_o*y[4] #- gamma_o*y[4]*y[5] - delta_o*y[4]*y[6]
        #p = alpha_p*y[2]*y[5] + beta_p*y[3]*y[5] + gamma_p*y[4]*y[5] - delta_p*y[5]
        #e = alpha_e*y[2]*y[6] + beta_e*y[3]*y[6] + gamma_e*y[4]*y[6] - delta_e*y[6]
        return np.array([c, d, s, f, o])

    for i in range(Nmax):
        populations.append([t, y[0], y[1], y[2], y[3], y[4]])
        if integrator == rk4:
            y = rk4(y, f, t, h)
        else:
            y = euler(y, f, t, h)
        t += h

    return np.array(populations)

#========================================
# Graphing functions
#----------------------------------------

def graph(Nmax, h = 0.01, integrator = rk4):
    '''
    Graphs all 5 populations vs time using rk4

    Parameters
    -----------
    Nmax: number of integration interations

    Returns
    -----------
    Graph of population densities vs time
    '''
    x = integrate(Nmax, h, integrator)
    plt.plot(x[:,0], x[:,1], label = 'cactus')
    plt.plot(x[:,0], x[:,2], label = 'dove')
    plt.plot(x[:,0], x[:,3], label = 'snake')
    plt.plot(x[:,0], x[:,4], label = 'fox')
    plt.plot(x[:,0], x[:,5], label = 'owl')
    #plt.plot(x[:,0], x[:,6], label = 'puma')
    #plt.plot(x[:,0], x[:,7], label = 'eagle')
    plt.xlabel('Time')
    plt.ylabel('Population density')
    plt.legend(loc = 'best')
    plt.show()

def graph_phase(Nmax, h = 0.01, integrator = rk4, x = 1, y = 2):
    '''
    Graphs 2 specific populations in the phase plane

    Parameters
    -----------
    Nmax: number of integration interations
    x:    index of the population plotted on the x axis
    y:    index of the population plotted on the y axis

    Returns:
    -----------
    Phase plane graph
    '''
    p = integrate(Nmax, h, integrator)
    plt.plot(p[:,x], p[:,y])
    if x == 1:
        plt.xlabel('Cactus population')
    elif x == 2:
        plt.xlabel('Dove population')
    elif x == 3:
        plt.xlabel('Rattlesnake population')
    elif x == 4:
        plt.xlabel('Fox population')
    elif x == 5:
        plt.xlabel('Owl population')
    elif x == 6:
        plt.xlabel('Puma population')
    elif x == 7:
        plt.xlabel('Eagle population')
    else:
        print("Animal index chosen does not exist")
    if y == 1:
        plt.ylabel('Cactus population')
    elif y == 2:
        plt.ylabel('Dove population')
    elif y == 3:
        plt.ylabel('Rattlesnake population')
    elif y == 4:
        plt.ylabel('Fox population')
    elif y == 5:
        plt.ylabel('Owl population')
    elif y == 6:
        plt.xlabel('Puma population')
    elif y == 7:
        plt.xlabel('Eagle population')
    else:
        print("Animal index chosen does not exist")
    plt.show()

def graph_trophic(Nmax, level = 1, h = 0.01, integrator = rk4):
    x = integrate(Nmax, h, integrator)
    if level == 1:
        plt.plot(x[:,0], x[:,1], label = 'cactus')
        plt.plot(x[:,0], x[:,2], label = 'dove')
    elif level == 2:
        plt.plot(x[:,0], x[:,3], label = 'snake')
        plt.plot(x[:,0], x[:,4], label = 'fox')
        plt.plot(x[:,0], x[:,5], label = 'owl')
    elif level == 3:
        plt.plot(x[:,0], x[:,6], label = 'puma')
        plt.plot(x[:,0], x[:,7], label = 'eagle')
    else:
        print('The trophic level selected is not part of the model')
    plt.xlabel('Time')
    plt.ylabel('Population density')
    plt.legend(loc = 'best')
    plt.show()

def graph_individual(Nmax, index = 1, h = 0.01, integrator = rk4):
    x = integrate(Nmax, h, integrator)
    if index == 1:
        plt.plot(x[:,0], x[:,1], label = 'cactus')
    elif index == 2:
        plt.plot(x[:,0], x[:,2], label = 'dove')
    elif index == 3:
        plt.plot(x[:,0], x[:,3], label = 'snake')
    elif index == 4:
        plt.plot(x[:,0], x[:,4], label = 'fox')
    elif index == 5:
        plt.plot(x[:,0], x[:,5], label = 'owl')
    elif index == 6:
        plt.plot(x[:,0], x[:,6], label = 'puma')
    elif index == 7:
        plt.plot(x[:,0], x[:,7], label = 'eagle')
    else:
        print('The index selected is not part of the model')
    plt.xlabel('Time')
    plt.ylabel('Population density')
    plt.legend(loc = 'best')
    plt.show()
