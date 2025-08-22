import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
import scipy.constants as const
from diffrax import diffeqsolve, ODETerm, Dopri5
from sympy import symbols, diff
import jax.numpy as jnp
import math
import random

# (p) = placeholder
# (c) = correctly defined

############### 1. FUNCTIONS ###############

#Phi (Φ) (c)
def wavePhase(temp, vel, mass, dens):
    return (mass/(2*np.pi*temp))^(3/2) * np.exp((-mass*(vel-dens)^2)/2*temp)

#Psi (ψ) (c)
def waveFunc(temp, vel, mass, E_array, E):
    numerator = np.exp(-E/temp)
    denominator = 0.0
    for i in E_array:
        denominator += np.exp(-E_array(i)/temp)
    return numerator/denominator

# MAXWELL-BOLTZMANN (p)
def maxwell_boltzmann(v, m, t):
    '''
        This is a function that computes the Maxwell-Boltzmann Distribution for: 
            - `v`: np.array of velocities
            - `m`: mass of the gas of interest
            - `t`: temperature of the system of interest
    '''
    r = 8.31446261815324 # J / (mol * K)
    return(4.*np.pi * ((m / (2 * np.pi * r * t))**1.5) * (v**2) * np.exp(- (m * v**2)/(2 * r * t)))

########### 2. INITIAL VARIABLES ###########

# initial speed, defined as roughly 3 * 10^8 m/s (c)
v_0 = const.speed_of_light

#initial temperature (in K) (p)
T_0 = 273.0

#equilibrium distribution
f_0 = 1.0

#specific heats: translational, internal, total (p)
C_i_v = 1.0
C_tr_v = 1.0
C_v = C_i_v + C_tr_v

# relaxation time estimates (p)
tau_elastic = 3.876 * (10^-7)
tau_inelastic = 2.47 * (10^-8)

#energy_levels is ⟨E⟩, observed_energy is E (p)
energy_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
expected_energy = np.mean(energy_levels)
observed_energy = energy_levels[random.randint(1,10)]

#first analytical expressions
f_1 = maxwell_boltzmann(v,m,t)
f = f_0 + n_0*f_1(v,m,t)
J = special.jv(1, f)
field = lambda t, f, args: f(t, args)
term = ODETerm(field)
time_diff = diffeqsolve(term, Dopri5())

M_t = n*rho(T_t)*psi(T_i)

# MODEL EQUATION
M_elastic = n*phi(temp_elastic)*psi(temp_elastic)
M_inelastic = n*phi(temp_inelastic)*psi(temp_inelastic)

############### CASE 1 ###############

# thermal coefficient (p)
A_0 = 1.0



############### CASE 5 ###############

x = Omega/(2*k_bar*v_0)
w = 0.0
Z = x-yj
W = 1/(math.pi*v_0^2)^(3/2) * pi * v_0^2 * v_0*(integrate.quad((math.e^(-w^2))/(w-Z), np.inf, -np.inf))
u = (y/(1+alpha)) * (1 + alpha_disp/(1+disp))
energy_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
observed_energy = energy_levels[random.randint(1,10)]

pb_numerator = (1+Z*W) * (1j*u*W-(3/2)) - 1j*u * ((Z^2-1/2)*W+Z)
pb_denominator = (1j*y*W-1+2j*x*y(1+Z*W)) * (1j*u*W-(3/2)) + (1/2-x^2) * 1j*u * ((Z^2-1/2)*W+Z)

p_bar = (pb_numerator/pb_denominator) * ((-1j * A_0)/(k * v_0^2))

case_5_matrix = np.array([
    [1j*y*W - 1, 2j*y * (1+Z*W), 1j*u * ((Z^2-1/2)*W+Z), 1j*r * ((Z^2-1/2)*W+Z)],
    [x, -1, 0, 0],
    [-1/2, x, 1j*u*W-3/2, 1j*r*W],
    [0, 0, 1j*V*W, 1j*s*W - 1]
])

case_5_operand = np.array([
    [p_bar],
    [v_bar_local],
    [delta_T_translational/T_0],
    [delta_T_internal/T_0]
])

#program demo
while (true):
    prompt = input("What would you like to know?").lower
    if prompt == "brillouin gain":
        print("gain: " + r) 
    elif prompt == "result matrix":
        print("Result matrix: " + np.dot(case_5_matrix, case_5_operand))
