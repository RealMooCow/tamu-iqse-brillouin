import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from diffrax import diffeqsolve, ODETerm, Dopri5
import jax.numpy as jnp
import math

#various variables
v_0 = 1.0
#specific heats: translational, internal, total
C_i_v = 0.0
C_t_v = 0.0
C_v = 0.0

#first analytical expressions
f_0 = 1.0
f_1 = maxwell_boltzmann(v,m,t)
f = f_0 + n_0*f_1(v,m,t)
J = special.jv(1, f)
field = lambda t, f, args: f(t, args)
term = ODETerm(field)
time_diff = diffeqsolve(term, Dopri5())

def maxwell_boltzmann(v, m, t):
    '''
        This is a function that computes the Maxwell-Boltzmann Distribution for: 
            - `v`: np.array of velocities
            - `m`: mass of the gas of interest
            - `t`: temperature of the system of interest
    '''
    r = 8.31446261815324 # J / (mol * K)
    return(4.*np.pi * ((m / (2 * np.pi * r * t))**1.5) * (v**2) * np.exp(- (m * v**2)/(2 * r * t)))

M_t = n*rho(T_t)*psi(T_i)



# CASE 5

w = 0.0
Z = 0.0
W = 1/(math.pi*v_0^2)^(3/2) * pi * v_0^2 * v_0(integrate.quad((math.e^(-w^2))/(w-Z), np.inf, -np.inf))
u = 0.0

pb_numerator = (1+Z*W)*(1j*u*W-(3/2))-1j*u((Z^2-1/2)*W+Z)
pb_denominator = (1j*y*W-1+2j*x*y(1 + Z*W))*(1j*u*W-(3/2)) + (1/2-x^2)*1j*u((Z^2-1/2)*W+Z)

p_bar = pb_numerator/pb_denominator