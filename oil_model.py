""" Script solving the dynamics of sinking particles with 
mixed oil and ballast phase, being degraded by bacteria, and
computing transfer efficiency of the oil flux.
Includes plotting of the results in a figure corresponding to Supplementary Figure 8 of manuscript:
“Microbial metabolic specificity controls pelagic lipid export efficiency” Lars Behrendt, Uria Alcolombri, Jonathan E. Hunter, Steven Smriga, Tracy Mincer, Daniel P. Lowenstein, Yutaka Yawata, François J. Peaudecerf, Vicente I. Fernandez, Helen F. Fredricks, Henrik Almblad, Joe J. Harrison, Roman Stocker, Benjamin A. S. Van Mooy, *BioRxiv* (2024)
DOI of manuscript: 10.1101/2023.12.08.570822
Direct link to manuscript: https://www.biorxiv.org/content/10.1101/2023.12.08.570822v1


Author: Francois Peaudecerf
Creation: 29.08.2020

History of modifications
12.03.2021: - further edits to compute integrated transfer efficiency
            - modified value of rho_bal to 2.71e3 in kg per m3
            - add safety of radius oil not going under zero after total degradation.
25.04.2021: - minor edits
01.05.2021: - merging of the three subpanels in a single figure
28.05.2021: - creation of a v1 figure with modified values for default parameters
26.09.2021: - creation of a v2 of the script and figure, due to update in values of experimental rates of degradation.
              Correction: function T had inconsistency in search of final position, total degradation was considered after t_oil when in general total
              degradation happens after t_delay+t_oil. Fixed at two points of function T (see local comments), and also function phi (~l535) (see local comments)
              This fixes some of the abberation observed at low radii and high delays.
              Addition of creation of r0l array in final figure plotting.
              Curation of the last figure improved.
21.10.2021: - creation of a v3 of the script and figure, due to update of the value of ballast fraction
14.01.2021: - creation of a v4 of the script: following change in experimental degradation rates constants from experiments
              the input data coming from Monte-Carlo simulation has changed and the figure need to be adapted. 
04.03.2024: - editing for upload on Github

"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as matplotlib
import utils
from scipy import optimize
from scipy import integrate

###### Preamble ######

plt.close('all')
 
# plotting using Tex
plt.rc('text', usetex=False) #to get Latex rendering is True : slows down the saving step
 
# width of figure in mm
mm_w = 165
# height of figure in mm
mm_h = 120
 
def mm2inch(value):
    return value/25.4
 
# definition of fontsize
fs = 10
matplotlib.rcParams.update({'font.size': fs})
 
# linewidth
lw = 1.0
 
# pad on the outside of the gridspec box
pad = 0.2
 
# extension for figures
ext = 'pdf'

# Set of color-blind friendly palette. Ref: https://jfly.uni-koeln.de/color/
orange = [230/255.0, 159/255.0, 0.0/255.0]
skyBlue = [86/255.0, 180/255.0, 233/255.0]
bluishGreen = [0/255.0, 158/255.0, 115.0/255.0]
yellow = [240/255.0, 228/255.0, 66.0/255.0]
blue = [0/255.0, 114/255.0, 178.0/255.0]
vermilion = [213/255.0, 94/255.0, 0.0/255.0]
reddishPurple = [204/255.0, 121/255.0, 167.0/255.0]


##### Parameters ######

ka      = 2.0e-8   #  in kg per m2 per second, mass degradation rate per unit surface area
rho_oil = 8.8e2    #  in kg per m3, oil density
rho_sw  = 1.027e3  #  in kg per m3, sea water density
rho_bal = 2.35e3   #  in kg per m3, ballast density

d_rho_bal = rho_bal - rho_sw
d_rho_oil = rho_oil - rho_sw

g = 9.8 # in m per s2, acceleration of gravity
mu = 1.40e-3 # in kg per m per s, dynamic viscosity

phi0 = 0.125 # fraction of ballast

r0_l = np.array([25, 50, 100, 250])*1e-6 # in m, selection of initial radii for plotting

rho_mean = rho_bal*phi0 + (1-phi0)*rho_oil


##### Functions based on analytical solutions of single particle dynamics #####

def roilt(t, roil0, ka, rho_oil):
	'''Returns equivalent radius of oil phase with time, in the case of no delay for degradation.
	INPUTS
	t      : array of time points on which to compute solution, in seconds
	roil0  : initial radius of the oil phase, in m
	ka     : mass degradation rate per unit surface area, in kg per m2 per second
	rho_oil: oil density in kg per m3
	OUTPUT
	roil(t): array with radius of oil phase at times t, in m 
	 '''
	return np.maximum(0, roil0 - ka*t/rho_oil)

def roilt_f(t, roil0, ka, rho_oil, t_delay):
	'''Returns radius of oil phase with time, in the delay case.
	INPUTS
	t      : array of time points on which to compute solution, in seconds
	roil0  : initial radius of the oil phase, in m
	ka     : mass degradation rate per unit surface area, in kg per m2 per second
	rho_oil: oil density in kg per m3
	t_delay: delay time, in seconds
	OUTPUT
	roil(t)   : array with radius of oil phase at times t, in m 
	 '''
	if t_delay ==0:
		roil = roilt(t, roil0, ka, rho_oil)
	else:
		roil1 = 0*t[t<t_delay] + roil0 # no degradation before t_delay
		roil2 = roilt((t[t>=t_delay]-t_delay), roil0, ka, rho_oil)
		roil = np.concatenate((roil1, roil2), axis = -1)
	return roil 

def rt(t, roil0, rbal0, ka, rho_oil):
	'''Returns total radius of particle with time, in the no delay case.
	INPUTS
	t      : array of time points on which to compute solution, in seconds
	roil0  : initial radius of the oil phase, in m
	rbal0  : radius of ballast phase, in m
	ka     : mass degradation rate per unit surface area, in kg per m2 per second
	rho_oil: oil density in kg per m3
	OUTPUT
	r(t)   : array with radius of particle at times t, in m 
	 '''
	return np.power( rbal0**3 + roilt(t, roil0, ka, rho_oil)**3 , 1/3.0)

def z_ref(t, r0, phi0, ka, g, mu, rho_oil, rho_sw, rho_bal):
	'''Returns position z of sinking particle as a function of time without delay
	Initial condition is z(t=0) = 0.
	INPUTS
	t      : array of time points on which to compute solution, in seconds 
	r0     : initial total radius, in meters
	phi0   : initial ballast fraction
	ka     : mass degradation rate per unit surface area, in kg per m2 per second
	g      : acceleration of gravity, in m per s2
	mu     : dynamic viscosity, in kg per m per s = Pa s
	rho_oil: oil density in kg per m3
  rho_sw : seawater density in kg per m3 
  rho_bal: ballast density in kg per m3
  OUTPUT
  z(t)      : depth of particle at times t, in meters
	'''
	d_rho_bal = rho_bal - rho_sw
	d_rho_oil = rho_oil - rho_sw


	# print('initial part radius = ', r0*1e6, ' microns')
	rbal0  = np.power(phi0*r0**3,1/3.0)
	# print('initial ballast radius = ', rbal0*1e6, ' microns')
	roil0  = np.power((1-phi0)*r0**3,1/3.0)
	# print('initial oil radius = ', roil0*1e6, ' microns')

	theta0 = roil0/r0

   # we compute the solutions for the oil radius and total radius
	roilt_l = roilt(t, roil0, ka, rho_oil)
	rt_l    = rt(t, roil0, rbal0, ka, rho_oil)

  # we compute the variable thetat from this solution
	thetat = roilt_l/rt_l
	
	# we compute the depth of the particle with time from its analytical expression
	# in equation (25) of the supplementary material of the manuscript
	z = rho_oil*g*(
                    6*d_rho_oil*(
                                  theta0*r0**3 - thetat*rt_l**3
                    	         )
                    + (3*d_rho_bal - d_rho_oil)*rbal0**3*(
                                                         2*np.sqrt(3)*(np.arctan((1 + 2*theta0)/np.sqrt(3)) - np.arctan((1 + 2*thetat)/np.sqrt(3)))
                                                         - 2*np.log( (1 - theta0)/(1 - thetat) )
                                                         + np.log( (1 + theta0 + theta0**2)/(1 + thetat + thetat**2) )
                    	                                )
    	           )/(81*ka*mu)

	return z

def zf(t, r0, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay): 
	'''Returns position z as a function of time for any delay
	Initial condition is z(t=0) = 0.
	INPUTS
	t      : array of time points on which to compute solution, in seconds 
	r0     : initial total radius, in meters
	phi0   : initial ballast fraction
	ka     : mass degradation rate per unit surface area, in kg per m2 per second
	rho_oil: oil density in kg per m3
	rho_bal: ballast density in kg per m3
	rho_sw : seawater density in kg per m3 
	mu     : dynamic viscosity, in kg per m per s = Pa s
	g      : acceleration of gravity, in m per s2
	t_delay: delay time, in seconds
  OUTPUT
  z(t)      : depth of particle at times t, in meters'''

	d_rho_bal = rho_bal - rho_sw
	d_rho_oil = rho_oil - rho_sw

	# print('initial part radius = ', r0*1e6, ' microns')
	rbal0  = np.power(phi0*r0**3,1/3.0)
	# print('initial ballast radius = ', rbal0*1e6, ' microns')
	roil0  = np.power((1-phi0)*r0**3,1/3.0)
	# print('initial oil radius = ', roil0*1e6, ' microns')

	if t_delay ==0:
		z = z_ref((t-t_delay), r0, phi0, ka, g, mu, rho_oil, rho_sw, rho_bal)
	else:

		# before delay, the particle sinks at a constant speed S0
		S0 = 2*g*(rbal0**3*d_rho_bal + roil0**3*d_rho_oil)/(9*mu*r0)
		z1 = S0*t[t<t_delay]

	  # we add one little step at speed S0 for continuity af the solution
	  # and then switch to the analytical solution for sinking position when degradation kicks in
		z2 = z1[-1] + S0*(t_delay - t[t<t_delay][-1]) + z_ref((t[t>=t_delay]-t_delay), r0, phi0, ka, g, mu, rho_oil, rho_sw, rho_bal)

		z = np.concatenate((z1, z2), axis = -1)

	return z

def zf_float(t, r0, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay):
	'''Returns position at a single float time for a given delay
	Used for root solving, but does the same as zf above.
	Initial condition is z(t=0) = 0.
	INPUTS
	t      : time point on which to compute solution, in seconds 
	r0     : initial total radius, in meters
	phi0   : initial ballast fraction
	ka     : mass degradation rate per unit surface area, in kg per m2 per second
	rho_oil: oil density in kg per m3
	rho_bal: ballast density in kg per m3
	rho_sw : seawater density in kg per m3 
	mu     : dynamic viscosity, in kg per m per s = Pa s
	g      : acceleration of gravity, in m per s2
	t_delay: delay time, in seconds
  OUTPUT
  z(t)   : depth of particle at time t, in meters'''

	d_rho_bal = rho_bal - rho_sw
	d_rho_oil = rho_oil - rho_sw

	# print('initial part radius = ', r0*1e6, ' microns')
	rbal0  = np.power(phi0*r0**3,1/3.0)
	# print('initial ballast radius = ', rbal0*1e6, ' microns')
	roil0  = np.power((1-phi0)*r0**3,1/3.0)
	# print('initial oil radius = ', roil0*1e6, ' microns')

	if t_delay ==0:
		z = z_ref((t-t_delay), r0, phi0, ka, g, mu, rho_oil, rho_sw, rho_bal)
	else:

    # before delay, the particle sinks at a constant speed S0
		S0 = 2*g*(rbal0**3*d_rho_bal + roil0**3*d_rho_oil)/(9*mu*r0)
		if t<t_delay:
			z = S0*t
		else:
			z = S0*t_delay + z_ref((t-t_delay), r0, phi0, ka, g, mu, rho_oil, rho_sw, rho_bal)

	return z

def zopt(t, r0, zT, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay):
	'''Optimisation function whose root gives the time at which a particle reaches
	a target depth zT, as usual with initial condition z(t=0) = 0.
	INPUTS
	t      : time variable, in seconds 
	r0     : initial total radius, in meters
	zT     : target depth, in meters
	phi0   : initial ballast fraction
	ka     : mass degradation rate per unit surface area, in kg per m2 per second
	rho_oil: oil density in kg per m3
	rho_bal: ballast density in kg per m3
	rho_sw : seawater density in kg per m3 
	mu     : dynamic viscosity, in kg per m per s = Pa s
	g      : acceleration of gravity, in m per s2
	t_delay: delay time, in seconds
  OUTPUT
  z - zT : difference between depth z reached at t and target zT, in meters'''
	return zf_float(t, r0, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay) - zT

def toil(r0, phi0, ka, rho_oil):
	'''Returns the time necessary for complete degradation of oil phase
	INPUTS
	r0     : initial total radius, in meters
	phi0   : initial ballast fraction
	ka     : mass degradation rate in kg per m2 per second
	rho_oil: oil density in kg per m3
	OUTPUT
	toil   : time for complete oil degradation, in seconds'''
	roil0  = np.power((1-phi0)*r0**3,1/3.0)
	return roil0*rho_oil/ka


##### Example plot of single particle sinking dynamics, on a few chosen radii #####

# switch
plot_example = 1
if plot_example == 1:

	scaling = 0.75
	fig = plt.figure(figsize = (mm2inch(scaling*mm_w), mm2inch(scaling*mm_h))) 
	gs = gridspec.GridSpec(1,1)
	ax1 = fig.add_subplot(gs[0,0])

	t_oil = toil(r0_l, phi0, ka, rho_oil)
	#print('toil = ', t_oil, 'in seconds')
	#print('toil = ', t_oil/(3600*24), 'in days')

	tmax = 10 # in days
	tmax = tmax*24*60*60 # in seconds
	t = np.linspace(0, tmax, 200)
	t_delay = 12 # in hours
	t_delay = t_delay*60*60 # in seconds

	for i, r0 in enumerate(r0_l):
		z = zf(t, r0, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay)
		label = r'$r_0$ = {:.0f} $\mu$m'.format(r0*1e6)
		ax1.plot(t/(3600*24), z, label = label)

		# To overlay approximated solution for checking consistency, uncomment below
		# rbal0  = np.power(phi0*r0**3,1/3.0)
		# roil0  = np.power((1-phi0)*r0**3,1/3.0)
		# r_t = rt(t, roil0, rbal0, ka, rho_oil)
		# roil_t = roilt(t, roil0, ka, rho_oil)
		# dzdt = 2*g*(d_rho_bal*rbal0**3 + d_rho_oil*roil_t**3)/(r_t*9*mu)
		# z_approx = np.cumsum(dzdt[:-1]*np.diff(t))
		# ax1.plot(t[:-1]/(3600*24),z_approx,'--k')

	ax1.set_xlabel(r'time $t$ (days)')#(\mathrm{m}/\mathrm{day})$')
	ax1.set_ylabel(r'depth $z$ (m)')#' (\mathrm{mm}^3/\mathrm{s})$')
	ax1.invert_yaxis()

	plt.legend()

	gs.tight_layout(fig, rect = [0.0, 0.0, 1.0, 1.0],  h_pad = 0.1, w_pad = 0.0, pad = pad) # pad corresponds to the padding around the subplot domain

	# plt.show()
	# plt.close()
	utils.save('sinking_dynamics_example_oil_model', ext = 'eps')


###### Fonctions and computations for tranfer efficiency #####

# We are first interested in transfer efficiency for disctinct size classes #

def T(zT, r0l, phi0, ka, g, mu, rho_oil, rho_sw, rho_bal, t_delay): # could be renamed Tp...
	'''Function returning the transfer efficiency at depth z for a given size class,
	which corresponds to the ratio of current volume at depth z to initial oil volume.
	See Supplementary  of manuscript for details.
	INPUTS
	zT      : depth where transfer efficiency is computed, in meters
	r0l     : initial total radius, in meters. Can be an array
	phi0    : initial ballast fraction
	ka      : mass degradation rate per unit surface area, in kg per m2 per second
	g       : acceleration of gravity, in m per s2
	mu      : dynamic viscosity, in kg per m per s = Pa s
	rho_oil : oil density in kg per m3
	rho_sw  : seawater density in kg per m3 
	rho_bal : ballast density in kg per m3
	t_delay : delay time, in seconds
  OUTPUT:
  T(z, r0): transfer efficiency at depth z for a given initial size r0 '''

	T = np.zeros_like(r0l)

	# we compute the positions with time for each radius and look for residual oil content at target depth
	for i, r0 in enumerate(r0l):
		# we compute the max time of degradation
		t_oil = toil(r0, phi0, ka, rho_oil)

		roil0  = np.power((1-phi0)*r0**3,1/3.0)

		# we compute the depth reached when the last of the oil is degraded
		zr0 = zf_float(t_delay + t_oil, r0, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay) #<- CORRECTION: the last of oil is degraded at t_oil + t_delay, not at t_oil as initially stated (no degradation before t_delay)

		if zr0 < zT:
			print('depth never reached before total degradation')
			# T takes the default value of 0 given at initialisation
		else:
			# there is a time before t_delay+t_oil when the particle crosses zT
			sol = optimize.root_scalar(zopt, args = (r0, zT, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay), bracket=[0, t_delay+t_oil], method='brentq') # <- CORRECTION: the time range when particle cross horizon of interest is [0, t_delay+t_oil], not [0, t_oil] as earlier version had it.
			tT2 = np.asarray(sol.root)

			# at that time there is still some oil in the particle
			roiltz = roilt_f(tT2, roil0, ka, rho_oil, t_delay)
			roiltz = roiltz.item() 

			# the transfer efficiency is then accessible
			T[i] = roiltz**3/roil0**3

	return T

# We then want to consider transfer efficiencies integrated acrosse size classes #

def P_R0(R_0, R_l, R_g, beta):
	'''Size distribution of particles between the two limit radii R_l and R_g
	INPUTS:
	R_0  : array of initial radii [m]
	R_l  : minimum radius cut off [m]
	R_g  : maximum radius cut-off [m]
	beta : power law exponent of size distribution (positive)
	OUTPUT:
	P(R_0), probability distribution for the concentration of particles of a given radius, in 1/m4'''
	return ( (beta - 1)*(
                         R_l**(beta-1)*R_g**(beta-1)/(R_g**(beta-1)-R_l**(beta-1))
                         )*R_0**(-beta)
        )


def S0(R_0, phi0, rho_oil, rho_bal, rho_sw, mu, g):
	'''Return the initial speed of a particle of a given initial size
	INPUTS
	R_0    : initial total radius, in meters
	phi0   : initial ballast fraction
	rho_oil: oil density in kg per m3
	rho_bal: ballast density in kg per m3
	rho_sw : seawater density in kg per m3 
	mu     : dynamic viscosity, in kg per m per s = Pa s
	g      : acceleration of gravity, in m per s2
  OUTPUT
  S0     : initial sinking speed of particle, in meters per seconds'''

	d_rho_bal = rho_bal - rho_sw
	d_rho_oil = rho_oil - rho_sw

	# print('initial part radius = ', r0*1e6, ' microns')
	rbal0  = np.power(phi0*R_0**3,1/3.0)
	# print('initial ballast radius = ', rbal0*1e6, ' microns')
	roil0  = np.power((1-phi0)*R_0**3,1/3.0)

	return 2*g*(rbal0**3*d_rho_bal + roil0**3*d_rho_oil)/(9*mu*R_0)


def phi(R_0, zT, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay, R_l, R_g, beta):
	'''Function giving the integrand of the oil flux at a given depth zT
	See equation (18) in Supplementary to the manuscript
	INPUTS:
	R_0              : initial radius of particle in meters
	zT               : depth at which speed and volumes are evaluated, in meters
	phi0             : initial ballast fraction
	ka               : mass degradation rate per unit surface area, in kg per m2 per second
	rho_oil          : density of oil, in kg per m3
	rho_bal          : density of ballast, in kg per m3
	rho_sw           : density of sea water, in kg per m3
	mu               : dynamic viscosity of water in kg per m per s
	g                : acceleration of gravity, in m per s2
	t_delay          : time before start of degradation, in seconds
	R_l              : lower size cut-off in meters for particle size distribution
	R_g              : greater size cut-off in meters for particle size distribution
	beta             : exponent of the size distribution (positive)
	OUTPUT:
	phi              : integrand value for calculating vertical flux
	'''

	# compute number of particles in this size class
	N_R0 = P_R0(R_0, R_l, R_g, beta)

	# we compute the max time of degradation
	t_oil = toil(R_0, phi0, ka, rho_oil)
	# initial oil radius
	roil0  = np.power((1-phi0)*R_0**3,1/3.0)


	# we compute the depth reached when the last of the oil is degraded
	z_R0_t = zf_float(t_delay + t_oil, R_0, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay)  #<- CORRECTION: the last of oil is degraded at t_oil + t_delay, not at t_oil as earlier version stated (no degradation before t_delay)

	if z_R0_t < zT:
		print('depth never reached before total degradation')
		Voiltz = 0
	else:
		# there is a time tT before t_delay+t_oil when the particle crosses zT
		sol = optimize.root_scalar(zopt, args = (R_0, zT, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay), bracket=[0, t_delay + t_oil], method='brentq')# <- CORRECTION: the time range when particle cross horizon of interest is [0, t_delay+t_oil], not [0, t_oil] as earlier!
		tT = np.asarray(sol.root)

		# We compute the associated radius of oil associated with this particle
		roiltz = roilt_f(tT, roil0, ka, rho_oil, t_delay) 

		Voiltz = 4*np.pi*roiltz**3/3


	# we also need the initial speed
	S0_R0 = S0(R_0, phi0, rho_oil, rho_bal, rho_sw, mu, g)

	return N_R0*S0_R0*Voiltz

def phi_z0(R_0, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, R_l, R_g, beta):
	'''Function giving the integrand of the oil flux at initial depth
	INPUTS:
	R_0              : radius of particle of interest, in meters
	phi0             : initial ballast fraction
	ka               : mass degradation rate per unit surface area, in kg per m2 per second
	rho_oil          : density of oil, in kg per m3
	rho_bal          : density of ballast, in kg per m3
	rho_sw           : density of sea water, in kg per m3
	mu               : dynamic viscosity of water in kg per m per s, dynamic viscosity
	g                : acceleration of gravity, in m per s2
	R_l              : lower size cut-off in meters
	R_g              : greater size cut-off in meters
	beta             : exponent of the size distribution (positive)
	OUTPUT:
	phi_z0        : integrand value for calculating vertical flux at initial depth, without degradation
                      '''

  # the number of particle of radius R_0 comes from the size distribution
	N_z0 = P_R0(R_0, R_l, R_g, beta)

	# we compute the radius and volume of oil associated with this particle size
	roil_z0  = np.power((1-phi0)*R_0**3,1/3.0)
	Voil_z0 = 4*np.pi*roil_z0**3/3

	# the speed of particles of this radius R_0 comes from the sedimentation law
	S_z0 = S0(R_0, phi0, rho_oil, rho_bal, rho_sw, mu, g)

	return N_z0*S_z0*Voil_z0

def F(zT, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay, R_l, R_g, beta):
  '''Function returning flux of oil at a given depth zT for all particles radii R_0
  comprised between R_l and R_g
  INPUTS:
  zT               : depth at which speed and volumes are evaluated, in meters
	phi0             : initial ballast fraction
	ka               : mass degradation rate per unit surface area, in kg per m2 per second
	rho_oil          : density of oil, in kg per m3
	rho_bal          : density of ballast, in kg per m3
	rho_sw           : density of sea water, in kg per m3
	mu               : dynamic viscosity of water in kg per m per s, dynamic viscosity
	g                : acceleration of gravity, in m per s2
	t_delay          : time before start of degradation, in seconds
	R_l              : lower size cut-off in meters
	R_g              : greater size cut-off in meters
	beta             : exponent of the size distribution (positive)
  OUTPUT:
  F(z), in m per s'''

  result = integrate.quad(phi, R_l, R_g, args = (zT, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay, R_l, R_g, beta))
  return result[0]

def Fz0(phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, R_l, R_g, beta):
  '''Function returning flux of oil at initial depth z0 for all particles radii R_0
  comprised between R_l and R_g
  INPUTS:
	phi0             : initial ballast fraction
	ka               : mass degradation rate per unit surface area, in kg per m2 per second
	rho_oil          : density of oil, in kg per m3
	rho_bal          : density of ballast, in kg per m3
	rho_sw           : density of sea water, in kg per m3
	mu               : dynamic viscosity of water in kg per m per s, dynamic viscosity
	g                : acceleration of gravity, in m per s2
	R_l              : lower size cut-off in meters
	R_g              : greater size cut-off in meters
	beta             : exponent of the size distribution (positive)
  OUTPUT:
  F(z_0), in m per s'''

  result = integrate.quad(phi_z0, R_l, R_g, args = (phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, R_l, R_g, beta))
  return result[0]

# Example #

print("##################################################")
print("Example of usage of the flux computation functions for transfer efficiency estimation")
print("##################################################")


R_l = 25e-6 # minimum size in microns
R_g = 250e-6 # maximum size in microns

z = 100 # in meters

ka = 2.0e-8
t_delay = 0*3600

beta = +2.965

F0 = Fz0(phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, R_l, R_g, beta)
F100 = F(z, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay, R_l, R_g, beta)

print('Flux at z0 =', F0)
print('Flux at z =100m is ', F100)
print('T100 = ', F100/F0)

print("##################################################")


######### Building Supplementary Figure 8 from manuscript ####


# common parameters
ka_l = np.array([2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4])*1e-3 # in kg m-2 s-1 < modified in v4 according to new rates
t_delay_l = np.array([0, 6, 12, 24])*3600 # in seconds
colors_l = ['orange','turquoise','royalblue','orchid']
titles_l = [r'$k_A$ = 2e-6 g m$^{-2}$ s$^{-1}$',r'$k_A$ = 5e-6 g m$^{-2}$ s$^{-1}$',r'$k_A$ = 1e-5 g m$^{-2}$ s$^{-1}$', r'$k_A$ = 2e-5 g m$^{-2}$ s$^{-1}$',
r'$k_A$ = 5e-5 g m$^{-2}$ s$^{-1}$',r'$k_A$ = 1e-4 g m$^{-2}$ s$^{-1}$']

R_l = 25e-6 # minimum size in meters
R_g = 250e-6 # maximum size in meters
N_R = 200
r0l = np.linspace(R_l, R_g, N_R) # in meters

# the size distribution is described by the exponent of its power law
beta = +2.965


scaling = 2.1
fig = plt.figure(figsize = (mm2inch(scaling*mm_w), mm2inch(3*0.4*scaling*mm_h))) 
gs = gridspec.GridSpec(3,6)
# axis for the Monte Carlo results
axA = fig.add_subplot(gs[0,:])
# axes for 100m TE
axB7 = fig.add_subplot(gs[1,:]) # extra for labels
axB1 = fig.add_subplot(gs[1,0])
axB2 = fig.add_subplot(gs[1,1])
axB3 = fig.add_subplot(gs[1,2])
axB4 = fig.add_subplot(gs[1,3])
axB5 = fig.add_subplot(gs[1,4])
axB6 = fig.add_subplot(gs[1,5])
# axes for 900m TE
axC7 = fig.add_subplot(gs[2,:]) # extra for labels
axC1 = fig.add_subplot(gs[2,0])
axC2 = fig.add_subplot(gs[2,1])
axC3 = fig.add_subplot(gs[2,2])
axC4 = fig.add_subplot(gs[2,3])
axC5 = fig.add_subplot(gs[2,4])
axC6 = fig.add_subplot(gs[2,5])

axB7.spines['top'].set_color('none')
axB7.spines['bottom'].set_color('none')
axB7.spines['left'].set_color('none')
axB7.spines['right'].set_color('none')
axB7.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

axB7.set_ylabel(r'Transfer efficiency at 100 m $T_\mathrm{p}(100,r_\mathrm{tot,0})$')

axC7.spines['top'].set_color('none')
axC7.spines['bottom'].set_color('none')
axC7.spines['left'].set_color('none')
axC7.spines['right'].set_color('none')
axC7.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

axC7.set_xlabel(r'Initial diameter $2 r_\mathrm{tot,0}$ ($\mu$m)')
axC7.set_ylabel(r'Transfer efficiency at 900 m $T_\mathrm{p}(900,r_\mathrm{tot,0})$')


###### panel A: Monte Carlo #######
# we import the rates from csv file
rates = np.genfromtxt('monte_carlo_rates.csv',delimiter=',',skip_header=1) #<- modified in v2
rates = rates[:, 1]

ka_min = np.amin(rates)
ka_max = np.amax(rates)
axA.hist(rates, bins=10**np.linspace(np.log10(ka_min), np.log10(ka_max), 85))
axA.set_xscale('log')

axA.set_xlabel(r'Mass degradation constant $k_A$ (g m$^{-2}$ s$^{-1}$) from Monte Carlo simulation')
axA.set_ylabel(r'Counts (over 1000000 simulations)')

#### Panel B: depth 100 m####

zT = 100 # in meters

for i, ka in enumerate(ka_l):

	# first, we plot the transfer efficiency per class size
	for j, t_delay in enumerate(t_delay_l):

		T_test = T(zT, r0l, phi0, ka, g, mu, rho_oil, rho_sw, rho_bal, t_delay)
		label = r"$t_{{delay}}$ = {:.0f} h".format(t_delay/(3600)) #+ str(np.round(t_delay/(3600)))
		fig.axes[i+2].plot(2*r0l*1e6, T_test, color = colors_l[j], label = label)


	fig.axes[i+2].set_title(titles_l[i], fontsize = fs)
	fig.axes[i+2].set_ylim(-0.05, 1.05)
	

	# then, we compute the transfer efficiency integrated with delay 0 and delay 24h
	F0 = Fz0(phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, R_l, R_g, beta)
	F100_0 = F(zT, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay_l[0], R_l, R_g, beta)
	F100_24 = F(zT, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay_l[-1], R_l, R_g, beta)

	T100_0 = F100_0/F0
	T100_24 = F100_24/F0

	xpos = 1.1*1e6*(R_g+R_l)
	ypos1 = 0.25
	ypos2 = 0.05

	if i == 5:
		xpos = xpos*1.18


	fig.axes[i+2].text(xpos, ypos1, r"For $t_\mathrm{{delay}}=0$ h," "\n" r"$T(100) =${:.2f}%".format(T100_0*100), fontsize=10, color='k', 
            ha="center", va="center", bbox=dict(boxstyle="square", ec='None', fc='None',alpha = 0.7))
	fig.axes[i+2].text(xpos, ypos2, r"For $t_\mathrm{{delay}}=24$ h," "\n" r"$T(100) =${:.2f}%".format(T100_24*100), fontsize=10, color='k', 
            ha="center", va="center", bbox=dict(boxstyle="square", ec='None', fc='None',alpha = 0.7))

for label in axB2.get_yticklabels()[:]:
	label.set_visible(False)
for label in axB3.get_yticklabels()[:]:
	label.set_visible(False)
for label in axB4.get_yticklabels()[:]:
	label.set_visible(False)
for label in axB5.get_yticklabels()[:]:
	label.set_visible(False)
for label in axB6.get_yticklabels()[:]:
	label.set_visible(False)


fig.axes[2].legend(loc="upper left", bbox_to_anchor = (0.2,0.8))


###### Panel C: T900 #####
zT = 900 # in meters

for i, ka in enumerate(ka_l):

	# first, we plot the transfer efficiency per class size
	for j, t_delay in enumerate(t_delay_l):

		T_test = T(zT, r0l, phi0, ka, g, mu, rho_oil, rho_sw, rho_bal, t_delay)
		label = r"$t_{{delay}}$ = {:.0f} h".format(t_delay/(3600)) #+ str(np.round(t_delay/(3600)))
		fig.axes[i+7+2].plot(2*r0l*1e6, T_test, color = colors_l[j], label = label)

	fig.axes[i+7+2].set_ylim(-0.05, 1.05)
  # we create shared axes
	fig.axes[i+2].get_shared_x_axes().join(fig.axes[i+2], fig.axes[i+7+2])
	fig.axes[i+2].set_xticklabels([])

	# then, we compute the transfer efficiency integrated with delay 0 and delay 24h
	F0 = Fz0(phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, R_l, R_g, beta)
	F100_0 = F(zT, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay_l[0], R_l, R_g, beta)
	F100_24 = F(zT, phi0, ka, rho_oil, rho_bal, rho_sw, mu, g, t_delay_l[-1], R_l, R_g, beta)

	T100_0 = F100_0/F0
	T100_24 = F100_24/F0

	xpos = 1.1*1e6*(R_g+R_l)
	ypos1 = 0.25
	ypos2 = 0.05
	if i == 2:
		xpos = xpos*1.18
		print('i=11')
	if i == 3:
		xpos = xpos*1.18
		#xpos = 200
		#ypos1 = 0.96
		#ypos2 = 0.76
	if i == 4:
		xpos = xpos*1.18
		#xpos = 200
		#ypos1 = 0.96
		#ypos2 = 0.76
	if i == 5:
		xpos = 200
		ypos1 = 0.96
		ypos2 = 0.76
	fig.axes[i+7+2].text(xpos, ypos1, r"For $t_\mathrm{{delay}}=0$ h," "\n" r"$T(900) =${:.2f}%".format(T100_0*100), fontsize=10, color='k', 
            ha="center", va="center", bbox=dict(boxstyle="square", ec='None', fc='None',alpha = 0.7))
	fig.axes[i+7+2].text(xpos, ypos2, r"For $t_\mathrm{{delay}}=24$ h," "\n" r"$T(900) =${:.2f}%".format(T100_24*100), fontsize=10, color='k', 
            ha="center", va="center", bbox=dict(boxstyle="square", ec='None', fc='None',alpha = 0.7))

for label in axC2.get_yticklabels()[:]:
	label.set_visible(False)
for label in axC3.get_yticklabels()[:]:
	label.set_visible(False)
for label in axC4.get_yticklabels()[:]:
	label.set_visible(False)
for label in axC5.get_yticklabels()[:]:
	label.set_visible(False)
for label in axC6.get_yticklabels()[:]:
	label.set_visible(False)

gs.tight_layout(fig, rect = [0.0, 0.0, 1.0, 1.0],  h_pad = 0.0, w_pad = 0.0, pad = pad) # pad corresponds to the padding around the subplot domain

axC7.get_yaxis().labelpad = 17
axC7.get_xaxis().labelpad = 17

axB7.get_yaxis().labelpad = 17

utils.save('FigSF_full', ext = 'eps')


