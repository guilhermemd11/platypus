""" 
The code performs a simulation of a nitrous oxide - paraffin hybrid
rocket motor. Some inputs from the user are needed, as seen below. For
results, use Jupyter notebook, preferably, and import it as

from platypusAlpha import *

And create a Motor class. Example:

marimbondo = Motor()

This will get the motor design. Consider changing the default inputs for different motors.

For the proper simulation, type:

marimbondo.ignite()

After that, the code will already start carrying on the simulation.

Author: Guilherme Tavares

ASSUMPTIONS

- Isentropic flow along the nozzle

- Combustion chamber is adiabatic

- Oxidiser tank empties adiabatically

- Available volume of combustion chamber does not change

- Combustion products form a mixture which behaves like an ideal gas

- Ideal burning (no erosion along the burning port)

- After total consumption of liquid nitrous oxide, quality remains 1 (although, as the temperature insists on decreasing, liquid nitrous may be formed again).

- O/F ratio change along the burn, and thus the combustion chamber temperature, the molar mass of the gaseous products and the ratio of specific heats also change. But these changes are only function of the O/F ratio change. Neither the chamber pressure nor the nitrous oxide temperature affect these latter variables (in real life they do, although not in a strong manner). The variables are relative to chamber pressure of 35 bar and nitrous oxide temperature of 298 K.

KNOW BEFORE RUNNING

- Some inputs should be provided by the user on the section below.

- Scipy and CoolProp needed.

- The ullage is computed as being 10% of the length of the tank occupied by the liquid nitrous. This percentage can be changed inside the code. The tank's diameter is an input.

- The vent of the oxidiser tank is considered to be on the top of the tank.

- For plotting different variables, proceed to the bottom of the code and follow the examples.

- New versions of the code will be gradually uploaded to tackle the above-mentioned assumptions, and to improve user experience.

REFERENCES

For general internal ballistic's equations:
SUTTON, G.P.; BIBLARZ, Oscar. Rocket Propulsion Elements. 8th ed. John Wiley & Sons, Inc., Hoboken, New Jersey, 2010.

For self-pressurizing N20 Model:
BORGNAKKE, Claus; SONNTAG, Richard E. Fundamentos da termodinâmica. Editora Blucher, 2016.
Pages 158 - 161

For N20-paraffin's regression law:
LESTRADE, J.-Y.; ANTHOINE, J.; LAVERGNE, G. Liquefying fuel regression rate modeling in hybrid propulsion. The French Aerospace Lab, Mauzac, France; The French Aerospace Lab, Toulouse, France, 2010.

For N20 properties:
CoolProp (available documentation in http://www.coolprop.org/index.html)

"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simps
from scipy.integrate import solve_ivp
import time
import sys
try:
    import CoolProp.CoolProp
    from CoolProp.CoolProp import PropsSI
except ImportError:
    print('Unable to load CoolProp. CoolProp files will not be imported.')
    
    
class Motor:
    
    """ Initializes the hybrid engine simulation  
    """
    
    def __init__(self,
                 ambient_temperature = 300,
                 ambient_pressure = 1.01325,
                 g = 9.81,
                 initial_thrust = 2700,
                 total_impulse = 7900,
                 chamber_pressure = 35,
                 chamber_temperature = 3347,
                 molar_mass = 26.912,
                 k = 1.1475,
                 OFratio = 8,
                 a_gox = 1.28*10**-5,
                 n_gox = 0.94,
                 combustion_eff = 0.885,
                 expansion_eff = 0.95,
                 nozzle_conv_angle = 30,
                 nozzle_div_angle = 15,
                 rho_paraffin = 900,
                 d_grain = 0.05,
                 d_chamber = 0.098,
                 d_tank = 0.11446,
                 d_vent = 0.001,
                 Cd = 0.65,
                 Cd_Vent = 0.65,
                 oxidiser_mass_multiplier = 1.2,
                 OFeffect = False,
                 time_vector = 0):

        """   
        INPUTS
        
        ambient_temperature - Ambient temperature (K)
        DEFAULT: 300 K
        
        ambient_pressure - Ambient Pressure (Bar)
        DEFAULT = 1.01325 Bar (1 atm)
        
        g - Gravitational Acceleration (m/s^2)
        DEFAULT = 9.81 m/s^2
        
        initial_thrust - Initial Thrust (N)
        DEFAULT = 2700 N
        
        total_impulse - Total Impulse (Ns)
        DEFAULT = 7900 Ns
        
        chamber_pressure - Nominal Chamber Pressure (Bar)
        DEFAULT: 35 Bar
        
        chamber_temperature - Nominal Chamber Temperature (K)
        DEFAULT: 3347 K
        
        molar_mass - Molar mass of the products of combustion (kg/kmol)
        DEFAULT: 26.912 kg/kmol
        
        k - Heat capacity ratio of the products of combustion (Adimensional)
        DEFAULT: 1.1475
        
        OFratio - Oxidiser/Fuel Mass ratio (Adimensional)
        DEFAULT: 8
        
        a_gox - Factor on fuel regression law, given by: r_dot = a_gox*Gox^n_gox, where Gox is the oxidiser flux
        DEFAULT: 1.28*10**-5, from (LESTRADE,2010)
        
        n_gox - Factor on fuel regression law, given by: r_dot = a_gox*Gox^n_gox, where Gox is the oxidiser flux
        DEFAULT: 0.94, from (LESTRADE,2010)
        
        combustion_eff - Combustion Efficiency (Adimensional, < 1)
        DEFAULT: 0.885
        
        expansion_eff - Nozzle Expansion Efficiency (Adimensional, < 1)
        DEFAULT: 0.95
        
        nozzle_conv_angle - Nozzle convergence half-angle (degrees)
        DEFAULT = 30 degrees
        
        nozze_div_angle - Nozzle divergence half-angle (degrees)
        DEFAULT = 15 degrees
        
        rho_paraffin - Paraffin's density (kg/m^3)
        DEFAULT = 900 kg/m^3
        
        d_grain - Internal diameter of the paraffin grain (m)
        DEFAULT = 0.05 m
        
        d_chamber - Internal combustion chamber diamenter (m)
        DEFAULT = 0.10226
        
        d_tank - Internal oxidiser tank diameter (m)
        DEFAULT = 0.11446
        
        d_vent - Vent diameter (m)
        DEFAULT = 0.001
        
        Cd - Injector discharge coefficient
        DEFAULT = 0.65
        
        Cd_Vent - Vent discharge coefficient
        DEFAULT = 0.65
        
        oxidiser_mass_multiplier - Multiplication factor on oxidiser mass, to account for decreasing thrust
        DEFAULT = 1.2
        
        OFeffect    - Boolean: if true, considers the change in chamber temperature, molar mass of
                      the products of combustion and ratio of specific heats due to the inherent
                      change in OFratio inside an hybrid rocket engine.
        DEFAULT = False
                      
        time_vector - Time vector for output variables calculation
        DEFAULT = 0 (time vector will be the one yielded by the integration method)
        
        NOTE: The input of pressure must be in Bar, since it is a 
        common measure of pressure inside combustion chambers of rockets. Nevertheless,
        the calculation will be made in Pascal, for readiness.
        
        NOTE 2: If any of Nominal Chamber Pressure, Nominal Chamber Temperature, 
        Molar mass of the products of combustion, Heat Capacity ratio of the products of combustion
        or Oxidiser/Mass ratio should be changed, all the others must change as well.
        For getting the correct values, the author recommends the CEA NASA software.
        
        COMPUTED PARAMETERS:
        
        Cf - Thrust coefficient (Adimensional)
        
        R - Gas constant for products of combustion (J/kgK)
        
        cstar - Characteristic velocity (m/s)
        
        Isp - Specific Impulse (s)
        
        At - Nozzle throat area (m^2)
        
        throatRadius - Nozzle throat radius (m)
        
        Ae - Nozzle exit area (m^2)
        
        exitRadius - Nozzle exit radius (m)
        
        fuel_flow - Nominal fuel mass flow rate (kg/s)
        
        oxidiser_flow - Nominal oxidiser mass flow rate (kg/s)
        
        fuel_mass - Total fuel mass (kg)
        
        oxidiser_mass - Total oxidiser mass (kg)
        
        Lgrain - Paraffin grain length (m)
        
        IA - Injector area (m^2)
        
        nozzle_length - Nozzle length (m)
        
        chamber_free_volume - Free volume inside combustion chamber (m^3) 
        
        tank_length_until_vent - Length of the oxidiser tank until vent (m)
        
        tank_length - Total length of oxidiser tank (m)
        
        tank_volume - Volume of oxidiser tank (m^3)
        
        VA - Vent area (m^2)
    
        """
        self.ambient_temperature = ambient_temperature
        
        self.ambient_pressure = ambient_pressure*(10**5)
        
        self.g = g
        
        self.initial_thrust = initial_thrust
        
        self.total_impulse = total_impulse
        
        self.chamber_pressure = chamber_pressure*(10**5)
        
        self.chamber_temperature = chamber_temperature
        
        self.molar_mass = molar_mass
    
        self.k = k
        
        self.OFratio = OFratio
        
        self.a_gox = a_gox
        
        self.n_gox = n_gox
        
        self.combustion_eff = combustion_eff
        
        self.expansion_eff = expansion_eff
        
        self.nozzle_conv_angle = nozzle_conv_angle
        
        self.nozzle_div_angle = nozzle_div_angle
                 
        self.rho_paraffin = rho_paraffin
        
        self.d_grain = d_grain
        
        self.d_chamber = d_chamber
        
        self.d_tank = d_tank
        
        self.d_vent = d_vent
        
        self.Cd = Cd
        
        self.Cd_Vent = Cd_Vent
        
        self.oxidiser_mass_multiplier = oxidiser_mass_multiplier
        
        self.OFeffect = OFeffect
        
        self.time_vector = time_vector
        
        self.Cf = math.sqrt((2*self.k**2/(self.k-1))*(2/(self.k+1))**((self.k+1)/(self.k-1))*(1-(self.ambient_pressure/self.chamber_pressure)**((self.k-1)/self.k)))
        
        self.R = 8314.51/self.molar_mass
        
        self.cstar = math.sqrt(self.k*self.R*self.chamber_temperature)/(self.k*math.sqrt((2/(self.k+1))**((self.k+1)/(self.k-1))))
        
        self.Isp = self.cstar*self.combustion_eff*self.Cf*self.expansion_eff/self.g
        
        self.fuel_flow = self.initial_thrust*(1/(self.OFratio+1))/(self.Isp*self.g) #Sutton,nominal fuel mass flow (kg/s)
        
        self.oxidiser_flow = self.initial_thrust*(self.OFratio/(self.OFratio+1))/(self.Isp*self.g) #Sutton, nominal oxidiser mass flow (kg/s)
        
        self.At = self.initial_thrust/(self.expansion_eff*self.Cf*self.chamber_pressure)
        
        self.throatRadius = math.sqrt(4*self.At/np.pi)/2
        if self.throatRadius > self.d_chamber/2:
            print("WARNING: Throat diameter bigger than chamber diameter")
            exit()
        
        self.Ae = self.At/(((self.k+1)/2)**(1/(self.k-1))*(self.ambient_pressure/self.chamber_pressure)**(1/self.k)*math.sqrt(((self.k+1)/(self.k-1))*(1-(self.ambient_pressure/self.chamber_pressure)**((self.k-1)/self.k))))
        
        self.exitRadius = math.sqrt(4*self.Ae/np.pi)/2
        if self.exitRadius > self.d_chamber/2:
            print("WARNING: Exit diameter bigger than chamber diameter")
        
        self.fuel_mass = self.fuel_flow*self.total_impulse/self.initial_thrust
    
        self.oxidiser_mass = oxidiser_mass_multiplier*self.oxidiser_flow*self.total_impulse/self.initial_thrust
        
        self.Lgrain = np.around(self.fuel_flow/(np.pi*rho_paraffin*self.d_grain*1.28*(10**-5)*(4*self.oxidiser_flow/(np.pi*self.d_grain**2))**0.94),2) #Length of the paraffin grain (estimated from the nominal oxidiser and fuel mass flows and from regression law found on LESTRADE,2014).
        
        self.nox = CoolProp.AbstractState("TTSE&HEOS", "NITROUSOXIDE")
        self.noxHEOS = CoolProp.AbstractState("HEOS", "NITROUSOXIDE")
        
        self.nox.update(CoolProp.QT_INPUTS, 0, self.ambient_temperature)
    
        self.IA = self.oxidiser_flow/(self.Cd*math.sqrt(2*self.nox.rhomass()*(self.nox.p()-self.chamber_pressure)))
        
        self.nozzle_length = (self.d_chamber/2-self.throatRadius)/np.tan(np.pi*self.nozzle_conv_angle/180) + (self.exitRadius -self.throatRadius)/np.tan(np.pi*self.nozzle_div_angle/180)
        "NOTE: The author is aware that an entrance radius for the convergent part should be considered, but it will be neglected for the sake of simplicity in the design computation"
        
        self.chamber_free_volume = 2*d_chamber*(np.pi*d_chamber**2/4) + self.Lgrain*(np.pi*d_grain**2/4) # A distance of one chamber diameter between grain and injector and between grain and nozzle is given for proper atomization and combustion, respectively.
        
        self.tank_length_until_vent = 4*self.oxidiser_mass/(np.pi*(self.d_tank**2)*self.nox.rhomass())
        
        self.tank_length = 1.1*self.tank_length_until_vent
        
        self.nox.update(CoolProp.QT_INPUTS, 1, self.ambient_temperature)
        
        self.oxidiser_mass = self.oxidiser_mass + self.nox.rhomass()*(np.pi*d_tank**2/4)*(self.tank_length - self.tank_length_until_vent)
        
        self.tank_volume = self.tank_length*(np.pi*d_tank**2/4)
        
        self.VA = np.pi*(self.d_vent**2)/4
        
        if self.OFeffect == True:
        
            self.OF_list = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16]
        
            self.temperature_list = [0,1066.89,1243.39,1409.27,1615.65,1822.91,2000.79,2296.05,2594.58,2833.88,3020.84,3158.77,3251.37,3306.76,3335.60,3347.07,3347.50,3340.88,3329.70,3315.53,3299.36,3281.87,3263.51,3244.61,3225.37,3205.98,3186.55,3167.17,3147.91,3128.83,3109.96,3091.33,3072.97]
        
            self.molar_mass_list = [24,24.389,21.197,20.066,19.882,20.032,20.220,20.994,22.125,23.118,23.981,24.718,25.335,25.843,26.263,26.614,26.912,27.168,27.389,27.583,27.754,27.905,28.040,28.160,28.267,28.364,28.450,28.528,28.599,28.663,28.721,28.773,28.821]
        
            self.gamma_list = [1.12,1.1276,1.1607,1.2113,1.2534,1.2656,1.2643,1.2851,1.2613,1.2382,1.2156,1.1938,1.1755,1.1627,1.1547,1.1501,1.1475,1.1462,1.1456,1.1456,1.1460,1.1466,1.1474,1.1484,1.1495,1.1508,1.1522,1.1536,1.1551,1.1567,1.1583,1.1600,1.1616]
        
            self.spline_temperature = interpolate.interp1d(self.OF_list, self.temperature_list, kind='linear')
        
            self.spline_molar_mass = interpolate.interp1d(self.OF_list, self.molar_mass_list, kind='linear')
        
            self.spline_gamma = interpolate.interp1d(self.OF_list, self.gamma_list, kind='linear')
        
        return None
    
    def ignite(self):
        
        """
        Simulation of the nitrous oxide-paraffin hybrid motor
        """
        self.noxHEOS.update(CoolProp.QT_INPUTS, 1, self.ambient_temperature)
        self.initial_vapour_density = self.noxHEOS.rhomass()
        self.noxHEOS.update(CoolProp.QT_INPUTS, 0, self.ambient_temperature)
        self.initial_liquid_density = self.noxHEOS.rhomass()
        
        self.X = (self.initial_vapour_density*self.initial_liquid_density*self.tank_volume-self.initial_vapour_density*self.oxidiser_mass)/(self.oxidiser_mass*(self.initial_liquid_density-self.initial_vapour_density)) # Initial quality of nitrous oxide inside the tank
        
        if self.X < 0:
            print("Warning: X is less than zero! Increase tank volume or decrease initial mass")
            exit()

        "INITIALIZE MONITOR SOLUTIONS"
        
        self.mass_nitrous = self.oxidiser_mass
        self.initial_mass_nitrous = self.oxidiser_mass
        self.mass_liquid_nitrous = (1-self.X)*self.oxidiser_mass
        self.mass_vapour_nitrous = self.X*self.oxidiser_mass
        self.temperature_tank = self.ambient_temperature
        self.old_mass_nitrous = self.oxidiser_mass
        self.old_out_liquid_nitrous = 0
        self.old_out_vapour_nitrous = 0
        self.quality = self.X
        self.noxHEOS.update(CoolProp.QT_INPUTS, self.quality, self.temperature_tank)
        self.old_specific_internal_energy = self.noxHEOS.umass()
        
        # The following vectors are needed in order to retrieve, after the simulation, the temperature at the tank
        # at each time step. An interpolation between an output variable (radius of the paraffin grain) and a variable
        # calculated only inside the function (temperature_tank) will allow the reconstruction of the tank temperature
        # time history.
        self.temperature_tank_vector = []
        self.radius_grain_vector = []
        self.quality_vector = []
        
        # Initial solution
        self.y0 = [self.ambient_pressure,self.d_grain/2,0,0]
        
        tic = time.time()
        
        self.sol = solve_ivp(self.burnout, [0, 10*self.oxidiser_mass/self.oxidiser_flow], self.y0, method='LSODA', events=[self.event_pressure])
        
        toc = time.time()
        
        print("Elapsed time: ",round(toc-tic,1)," seconds")
        
        self.spline_chamber_pressure = interpolate.interp1d(self.sol.t, self.sol.y[0,:], kind='linear')
        
        self.spline_radius_grain = interpolate.interp1d(self.sol.t, self.sol.y[1,:], kind='linear')
        
        self.spline_consumed_liquid = interpolate.interp1d(self.sol.t, self.sol.y[2,:], kind='linear')
        
        self.spline_consumed_vapour = interpolate.interp1d(self.sol.t, self.sol.y[3,:], kind='linear')
        
        self.radius_grain_vector.sort()
        
        self.temperature_tank_vector.sort(reverse=True)
        
        self.quality_vector.sort()
        
        self.spline_temperature_tank = interpolate.interp1d(self.radius_grain_vector,self.temperature_tank_vector, kind='linear')
        
        self.spline_quality = interpolate.interp1d(self.radius_grain_vector,self.quality_vector, kind='linear')
        
        self.retrieve()
   
        return None
    
    ################
    # FUNCTIONS
    ################
    
    def event_pressure(self,t,x):
        """
        Provides the terminal event for the hybrid rocket simulation
        It imposes that, if the chamber pressure falls below ambient pressure,
        the code shall be terminated.
        
        INPUTS:
        
        t - time step of the simulation
        
        x - vector with integration variables
        """
        return x[0] - 0.9999999999999*self.ambient_pressure 
        
    event_pressure.terminal = True
    
    def burnout(self,t,x,retrieve = False):
        """
        This function provides the derivatives for integration through solve.ivp
        
        INPUTS:
        
        "x[0] - chamber pressure Po, Pascal"
        "x[1] - radius of Paraffin Grain, m"
        "x[2] - used liquid oxidiser mass, kg"
        "x[3] - used vapour oxidiser mass, kg"
        """
        
        if retrieve == True:
            
            if x[0] < self.ambient_pressure:
                x[0] = self.ambient_pressure
            if x[1] >= self.sol.y[1][len(self.sol.y[1,:])-1]:
                x[1] = self.sol.y[1][len(self.sol.y[1,:])-1] - 10**-6
                
            self.chamber_pressure_out.append(x[0])
            self.grain_radius_out.append(x[1])
            mass_nitrous = self.initial_mass_nitrous-x[2]-x[3]
            self.mass_nitrous_out.append(mass_nitrous)
            overall_density = round(mass_nitrous/self.tank_volume,3)
            #temperature_tank = self.spline_temperature_tank(round(x[1],8))
            temperature_tank = self.spline_temperature_tank(x[1])
            self.tank_temperature_out.append(temperature_tank)
            self.nox.update(CoolProp.DmassT_INPUTS, overall_density, temperature_tank)
            tank_pressure = self.nox.p()
            self.tank_pressure_out.append(tank_pressure)
            
            quality = self.spline_quality(x[1])
            
            #if only_vapour == False:
            #    phase = CoolProp.CoolProp.PhaseSI('D',overall_density,'T',temperature_tank,'N2O')
            #else:
            #    phase = 'gas'

            #if phase == 'gas' or phase == 'supercritical_gas':
            if quality > 0.99:
                vapour_nitrous_density = overall_density

                self.mass_liquid_nitrous_out.append(0)
                self.liquid_nitrous_density_out.append(self.liquid_nitrous_density_out[len(self.liquid_nitrous_density_out)-1])
                self.mass_vapour_nitrous_out.append(mass_nitrous)
                self.vapour_nitrous_density_out.append(vapour_nitrous_density)
                self.quality_out.append(1)

                if self.OFeffect == True:
            
                    self.nox.update(CoolProp.QT_INPUTS, 1, self.temperature_tank)
                    cp_vapour = self.nox.cpmass()
                    cv_vapour = self.nox.cvmass()
                    gamma_vapour = cp_vapour/cv_vapour
                    
                    if tank_pressure > x[0]*(1+(gamma_vapour-1)/2)**(gamma_vapour/(gamma_vapour-1)):
                        # Choked Flow
                        oxidiser_flow_rate = self.Cd*self.IA*(gamma_vapour*tank_pressure*vapour_nitrous_density*(2/(gamma_vapour+1))**( (gamma_vapour+1)/(gamma_vapour-1) ))**0.5
                    else:
                        # Subsonic Flow
                        oxidiser_flow_rate = self.Cd*self.IA*(2*gamma_vapour/(gamma_vapour-1)*tank_pressure*vapour_nitrous_density*((x[0]/tank_pressure)**(2/gamma_vapour)- (x[0]/tank_pressure)**((gamma_vapour+1)/gamma_vapour)) )**0.5
                        
                    fuel_flow_rate = self.Lgrain*2*np.pi*x[1]*self.rho_paraffin*self.a_gox*(oxidiser_flow_rate/(np.pi*(x[1]**2)))**self.n_gox
                    self.k_out.append(self.spline_gamma(oxidiser_flow_rate/fuel_flow_rate))
                    
                else:
                    self.k_out.append(self.k)
                    
            #elif phase == 'liquid' or phase == 'supercritical_liquid' or phase == 'twophase':
            else:
                #quality = self.nox.Q()
                self.quality_out.append(quality)
                
                self.mass_liquid_nitrous_out.append((1-quality)*mass_nitrous)
                self.mass_vapour_nitrous_out.append(quality*mass_nitrous)
                
                self.nox.update(CoolProp.QT_INPUTS, 0, temperature_tank)
                liquid_nitrous_density = self.nox.rhomass()
                self.liquid_nitrous_density_out.append(liquid_nitrous_density)

                self.nox.update(CoolProp.QT_INPUTS, 1, temperature_tank)
                vapour_nitrous_density = self.nox.rhomass()
                self.vapour_nitrous_density_out.append(vapour_nitrous_density)
                
                if self.OFeffect == True:
                    cp_vapour = self.nox.cpmass()
                    cv_vapour = self.nox.cvmass()      
                    gamma_vapour = cp_vapour/cv_vapour
                    
                    oxidiser_flow_rate = self.Cd*self.IA*(2*liquid_nitrous_density*(tank_pressure-x[0]))**0.5
                    fuel_flow_rate = self.Lgrain*2*np.pi*x[1]*self.rho_paraffin*self.a_gox*(oxidiser_flow_rate/(np.pi*(x[1]**2)))**self.n_gox
                    self.k_out.append(self.spline_gamma(oxidiser_flow_rate/fuel_flow_rate))
                    
                else:
                    self.k_out.append(self.k)
            
            return None
        
        #Two-phase simulation
        
        if self.quality < 0.99:
            
            out_liquid_nitrous = x[2] - self.old_out_liquid_nitrous
            out_vapour_nitrous = x[3] - self.old_out_vapour_nitrous
            self.mass_nitrous = self.initial_mass_nitrous-x[2]-x[3]
            
            self.nox.update(CoolProp.QT_INPUTS, 0, self.temperature_tank)  
            specific_enthalpy_liquid_nitrous = self.nox.hmass()
            
            self.nox.update(CoolProp.QT_INPUTS, 1, self.temperature_tank)
            specific_enthalpy_vapour_nitrous = self.nox.hmass()
            
            specific_internal_energy = (self.old_mass_nitrous*self.old_specific_internal_energy-out_liquid_nitrous*specific_enthalpy_liquid_nitrous-out_vapour_nitrous*specific_enthalpy_vapour_nitrous)/self.mass_nitrous
            overall_density = round(self.mass_nitrous/self.tank_volume,3)
            
            self.temperature_tank_vector.append(self.temperature_tank)
            self.radius_grain_vector.append(x[1])
            self.quality_vector.append(self.quality)
            
            self.noxHEOS.update(CoolProp.DmassUmass_INPUTS, overall_density, specific_internal_energy)
            self.temperature_tank = self.noxHEOS.T()
            
            self.nox.update(CoolProp.DmassT_INPUTS, overall_density, self.temperature_tank)
            self.quality = self.nox.Q()
            
            # Avoids quality = -1000
            if self.quality < 0:
                self.quality = 1
            
            #self.nox.update(CoolProp.QT_INPUTS, self.quality, self.temperature_tank)
            tank_pressure = self.nox.p()

            self.mass_liquid_nitrous = (1-self.quality)*self.mass_nitrous
            self.mass_vapour_nitrous = self.quality*self.mass_nitrous
            self.old_mass_nitrous = self.mass_nitrous
            self.old_specific_internal_energy = specific_internal_energy
            self.old_out_liquid_nitrous = x[2]
            self.old_out_vapour_nitrous = x[3]  
            
            self.nox.update(CoolProp.QT_INPUTS, 0, self.temperature_tank)
            liquid_nitrous_density = self.nox.rhomass()
            
            self.nox.update(CoolProp.QT_INPUTS, 1, self.temperature_tank)
            vapour_nitrous_density = self.nox.rhomass()
            cp_vapour = self.nox.cpmass()
            cv_vapour = self.nox.cvmass()      

            gamma_vapour = cp_vapour/cv_vapour
            oxidiser_flow_rate = self.Cd*self.IA*(2*liquid_nitrous_density*(tank_pressure-x[0]))**0.5
            fuel_flow_rate = self.Lgrain*2*np.pi*x[1]*self.rho_paraffin*self.a_gox*(oxidiser_flow_rate/(np.pi*(x[1]**2)))**self.n_gox
            if self.OFeffect == True:
                self.chamber_temperature = self.spline_temperature(oxidiser_flow_rate/fuel_flow_rate)
                self.molar_mass = self.spline_molar_mass(oxidiser_flow_rate/fuel_flow_rate)
                self.k = self.spline_gamma(oxidiser_flow_rate/fuel_flow_rate)
            self.R = 8314.51/self.molar_mass
            rho_gas = x[0]*self.molar_mass/(8314*self.chamber_temperature);
            dx0 = (self.R*self.chamber_temperature/self.chamber_free_volume)*(oxidiser_flow_rate+self.Lgrain*2*np.pi*x[1]*(self.rho_paraffin-rho_gas)*self.a_gox*((oxidiser_flow_rate/(np.pi*(x[1]**2)))**self.n_gox)-x[0]/self.combustion_eff*(np.pi*((2*self.throatRadius)**2)/4)*((self.k/(self.R*self.chamber_temperature))*(2/(self.k+1))**((self.k+1)/(self.k-1)))**0.5)
            dx1 = self.a_gox*(oxidiser_flow_rate/(np.pi*(x[1]**2)))**self.n_gox
            dx2 = oxidiser_flow_rate
            if tank_pressure > self.ambient_pressure*(1+(gamma_vapour-1)/2)**(gamma_vapour/(gamma_vapour-1)):
                dx3 = self.VA*self.Cd_Vent*(gamma_vapour*tank_pressure*vapour_nitrous_density*(2/(gamma_vapour+1))**( (gamma_vapour+1)/(gamma_vapour-1) ))**0.5
            else:
                dx3 = self.VA*self.Cd_Vent*(2*gamma_vapour/(gamma_vapour-1)*tank_pressure*vapour_nitrous_density*((self.ambient_pressure/tank_pressure)**(2/gamma_vapour)- (self.ambient_pressure/tank_pressure)**((gamma_vapour+1)/gamma_vapour)) )**0.5

            dx = [dx0, dx1, dx2, dx3]
            
        #Vapour-phase only
            
        elif self.quality >= 0.99 and x[0] > 101325:
            
            self.mass_liquid_nitrous = 0
            
            self.time_no_liquid = t
            
            self.quality = 1
            out_vapour_nitrous = x[3] - self.old_out_vapour_nitrous
            self.mass_nitrous = self.initial_mass_nitrous-x[2]-x[3]
            
            self.nox.update(CoolProp.QT_INPUTS, 1, self.temperature_tank)
            specific_enthalpy_vapour_nitrous = self.nox.hmass()
            
            specific_internal_energy = (self.old_mass_nitrous*self.old_specific_internal_energy-out_vapour_nitrous*specific_enthalpy_vapour_nitrous)/self.mass_nitrous
            overall_density = round(self.mass_nitrous/self.tank_volume,3)
            
            self.temperature_tank_vector.append(self.temperature_tank)
            self.radius_grain_vector.append(x[1]) 
            self.quality_vector.append(self.quality)
            
            self.noxHEOS.update(CoolProp.DmassUmass_INPUTS, overall_density, specific_internal_energy)
            self.temperature_tank = self.noxHEOS.T()
            
            self.nox.specify_phase(CoolProp.iphase_gas)
            self.nox.update(CoolProp.DmassT_INPUTS, overall_density, self.temperature_tank)
            tank_pressure = self.nox.p()

            self.mass_vapour_nitrous = self.mass_nitrous
            self.old_mass_nitrous = self.mass_nitrous
            self.old_specific_internal_energy = specific_internal_energy
            self.old_out_vapour_nitrous = x[3]

            self.nox.update(CoolProp.PT_INPUTS, tank_pressure, self.temperature_tank)
            vapour_nitrous_density = self.nox.rhomass()
            
            self.nox.specify_phase(CoolProp.iphase_not_imposed)
            
            self.nox.update(CoolProp.QT_INPUTS, 1, self.temperature_tank)
            cp_vapour = self.nox.cpmass()
            cv_vapour = self.nox.cvmass()
            
            gamma_vapour = cp_vapour/cv_vapour
            if tank_pressure > x[0]*(1+(gamma_vapour-1)/2)**(gamma_vapour/(gamma_vapour-1)):
                # Choked Flow
                oxidiser_flow_rate = self.Cd*self.IA*(gamma_vapour*tank_pressure*vapour_nitrous_density*(2/(gamma_vapour+1))**( (gamma_vapour+1)/(gamma_vapour-1) ))**0.5
            else:
                # Subsonic Flow
                oxidiser_flow_rate = self.Cd*self.IA*(2*gamma_vapour/(gamma_vapour-1)*tank_pressure*vapour_nitrous_density*((x[0]/tank_pressure)**(2/gamma_vapour)- (x[0]/tank_pressure)**((gamma_vapour+1)/gamma_vapour)) )**0.5
                
            fuel_flow_rate = self.Lgrain*2*np.pi*x[1]*self.rho_paraffin*self.a_gox*(oxidiser_flow_rate/(np.pi*(x[1]**2)))**self.n_gox
            if self.OFeffect == True:
                self.chamber_temperature = self.spline_temperature(oxidiser_flow_rate/fuel_flow_rate)
                self.molar_mass = self.spline_molar_mass(oxidiser_flow_rate/fuel_flow_rate)
                self.k = self.spline_gamma(oxidiser_flow_rate/fuel_flow_rate)
            self.R = 8314.51/self.molar_mass
            rho_gas = x[0]*self.molar_mass/(8314*self.chamber_temperature);
            dx0 = (self.R*self.chamber_temperature/self.chamber_free_volume)*(oxidiser_flow_rate+self.Lgrain*2*np.pi*x[1]*(self.rho_paraffin-rho_gas)*self.a_gox*((oxidiser_flow_rate/(np.pi*(x[1]**2)))**self.n_gox)-x[0]/self.combustion_eff*(np.pi*((2*self.throatRadius)**2)/4)*((self.k/(self.R*self.chamber_temperature))*(2/(self.k+1))**((self.k+1)/(self.k-1)))**0.5)
            dx1 = self.a_gox*(oxidiser_flow_rate/(np.pi*(x[1]**2)))**self.n_gox
            dx2 = 0
            if tank_pressure > self.ambient_pressure*(1+(gamma_vapour-1)/2)**(gamma_vapour/(gamma_vapour-1)):
                dx3 = oxidiser_flow_rate + self.VA*self.Cd_Vent*(gamma_vapour*tank_pressure*vapour_nitrous_density*(2/(gamma_vapour+1))**( (gamma_vapour+1)/(gamma_vapour-1) ))**0.5
            else:
                dx3 = oxidiser_flow_rate + self.VA*self.Cd_Vent*(2*gamma_vapour/(gamma_vapour-1)*tank_pressure*vapour_nitrous_density*((self.ambient_pressure/tank_pressure)**(2/gamma_vapour)- (self.ambient_pressure/tank_pressure)**((gamma_vapour+1)/gamma_vapour)) )**0.5
            dx = [dx0, dx1, dx2, dx3]
        
        #Tail-off phase
        
        else:
            
            dx = [-10**9, 0, 0, 0]
            
        return dx
    
    def retrieve(self):
        """
        Retrieves data from burnout function for outputs, at time instants given by time vector.
        It creates class vectors for relevant variables, which can be latter used by output functions.
        
        INPUTS:
        
        time - 0: default time vector, retrieved from the integration
                  supplied time vector: interpolation is carried out
                  
        OUTPUTS: none
                 but creates vectors self.k_out, self.mass_nitrous_out, self.mass_liquid_nitrous_out,
                                     self.liquid_nitrous_density_out, self.mass_vapour_nitrous_out,
                                     self.vapour_nitrous_density_out, self.grain_radius_out, 
                                     self.chamber_pressure_out, self.tank_pressure_out,
                                     self.tank_temperature_out, self.quality_out
        """
        # out == output
        self.k_out = []
        self.mass_nitrous_out = []
        self.mass_liquid_nitrous_out = []
        self.liquid_nitrous_density_out = []
        self.mass_vapour_nitrous_out = []
        self.vapour_nitrous_density_out = []
        self.grain_radius_out = []
        self.chamber_pressure_out = []
        self.tank_pressure_out = []
        self.tank_temperature_out = []
        self.quality_out = []
        
        if np.all(self.time_vector == 0):
            for i in range(len(self.sol.t)):
                self.burnout(self.sol.t[i],[self.sol.y[0][i],self.sol.y[1][i],self.sol.y[2][i],self.sol.y[3][i]],retrieve=True)
        else:
            if isinstance(self.time_vector,list) == False:
                self.time_vector = self.time_vector.tolist()
            if self.time_vector[0] < 0:
                print("Instant of time out of range!")
                return None
            if self.time_vector[len(self.time_vector)-1] > self.sol.t[len(self.sol.t)-1]:
                for i in range(len(self.time_vector)):
                    if self.time_vector[i] > self.sol.t[len(self.sol.t)-1]:
                        self.time_vector = self.time_vector[0:i-1]
                        break
                if self.time_vector[len(self.time_vector)-1] < self.sol.t[len(self.sol.t)-1]:
                    self.time_vector.append(self.sol.t[len(self.sol.t)-1])
            x0 = self.spline_chamber_pressure(self.time_vector)
            x1 = self.spline_radius_grain(self.time_vector)
            x2 = self.spline_consumed_liquid(self.time_vector)
            x3 = self.spline_consumed_vapour(self.time_vector)
            for i in range(len(self.time_vector)):
                self.burnout(self.time_vector[i],[x0[i],x1[i],x2[i],x3[i]],retrieve=True)
        return None
    
    ################
    # OUTPUTS
    ################    
    
    def thrust(self):
        """
        Returns the thrust vector (N) for the given time vector t (seconds)
        
        """
        
        self.thrust_vector = []
        
        if np.all(self.time_vector == 0):
            len_time = len(self.sol.t)
        else:
            len_time = len(self.time_vector)
            
        for i in range(len_time):
            cf = (((2*self.k_out[i]**2)/(self.k_out[i]-1))*(2/(self.k_out[i]+1))**((self.k_out[i]+1)/(self.k_out[i]-1))*(1-(self.ambient_pressure/self.chamber_pressure_out[i])**((self.k_out[i]-1)/self.k_out[i])))**0.5 
            self.thrust_vector.append(self.expansion_eff*cf*(np.pi*((2*self.throatRadius)**2)/4)*self.chamber_pressure_out[i])
                
        return self.thrust_vector    
        
    def mass(self,mass_empty_motor = 0):
        """
        Returns the motor mass vector (kg) for the given time vector t (seconds)
        
        If mass_empty_motor = 0, it returns only the propellant mass
        """
        self.motor_mass = []
        
        if np.all(self.time_vector == 0):
            len_time = len(self.sol.t)
        else:
            len_time = len(self.time_vector)

        for i in range(len_time):
            self.motor_mass.append(self.mass_nitrous_out[i] + (self.d_chamber**2-(2*self.grain_radius_out[i])**2)*np.pi/4*self.Lgrain*self.rho_paraffin + mass_empty_motor)
                                       
        return self.motor_mass
        
    def center_of_mass(self,time,mass_empty_motor = 0,cm_empty_motor = 0,distance_grain = 0, distance_nitrous = 0):
        """
        Returns the motor's center of mass vector (m) for the given time vector t (seconds)
        
        The center of mass is computed with respect to a reference frame at the center of the nozzle's exit diameter
        
        If mass_empty_motor == 0 or cm_empty_motor == 0, it returns only the propellant's center of mass
        
        distance_grain refers to the distance the bottom part of the paraffin grain is from the reference frame. If distance_grain = 0, it will approximately compute this distance according to prior calculations.
        
        distance_nitrous refers to the distance the bottom part of the nitrous oxide 'cylinder' is from the reference frame. If distance_nitrous = 0, it will approximately compute this distance according to prior calculations.
        """
        self.motor_center_of_mass = []
                                   
        if np.all(self.time_vector == 0):
            len_time = len(self.sol.t)
        else:
            len_time = len(self.time_vector)

        if distance_grain == 0:
            distance_grain = self.d_chamber + self.nozzle_length # A distance of one chamber diameter between grain and injector and between grain and nozzle is given for proper atomization and combustion, respectively.
        if distance_nitrous == 0:
            distance_nitrous = self.nozzle_length + 2*self.d_chamber + self.Lgrain # A distance of one chamber diameter between grain and injector and between grain and nozzle is given for proper atomization and combustion, respectively.
        d_cm_grain = distance_grain+self.Lgrain/2
                                       
        for i in range(len_time):                               
            d_cm_liquid_nitrous = distance_nitrous + self.mass_liquid_nitrous_out[i]/(self.liquid_nitrous_density_out[i]*np.pi*(self.d_tank**2)/4)/2
            d_cm_vapour_nitrous = distance_nitrous + self.mass_liquid_nitrous_out[i]/(self.liquid_nitrous_density_out[i]*np.pi*(self.d_tank**2)/4) + self.mass_vapour_nitrous_out[i]/(self.vapour_nitrous_density_out[i]*np.pi*(self.d_tank**2)/4)/2
            mass_grain = self.rho_paraffin*(np.pi*(self.d_chamber**2-4*self.grain_radius_out[i]**2)/4)*self.Lgrain
            if mass_empty_motor == 0 or cm_empty_motor == 0:
                self.motor_center_of_mass.append((mass_grain*d_cm_grain + self.mass_liquid_nitrous_out[i]*d_cm_liquid_nitrous + self.mass_vapour_nitrous_out[i]*d_cm_vapour_nitrous)/(mass_grain+self.mass_liquid_nitrous_out[i]+self.mass_vapour_nitrous_out[i]))
            else:
                self.motor_center_of_mass.append((mass_grain*d_cm_grain + self.mass_liquid_nitrous_out[i]*d_cm_liquid_nitrous + self.mass_vapour_nitrous_out[i]*d_cm_vapour_nitrous + mass_empty_motor*cm_empty_motor)/(mass_grain+self.mass_liquid_nitrous_out[i]+self.mass_vapour_nitrous_out[i]+mass_empty_motor)) 
        
        return self.motor_center_of_mass
        
    def Izz(self,Izz_empty_motor = 0):
        """
        Returns the motor's axial moment of inertia vector (kg m^2) for a given time vector t (seconds)
        
        The moment of inertia is computed relative to a reference frame at the center of mass of the assembly.
        
        If Izz_empty_motor == 0, it returns only the axial moment of inertia of the propellant
        """
        
        self.motor_Izz = []
                                   
        if np.all(self.time_vector == 0):
            len_time = len(self.sol.t)
        else:
            len_time = len(self.time_vector)
        
        for i in range(len_time):
            mass_grain = self.rho_paraffin*(np.pi*(self.d_chamber**2-4*self.grain_radius_out[i]**2)/4)*self.Lgrain
            
            Izz_grain = 0.5*mass_grain*(self.grain_radius_out[i]**2+(self.d_chamber/2)**2)

            Izz_liquid_nitrous = 0.5*self.mass_liquid_nitrous_out[i]*(self.d_tank/2)**2

            Izz_vapour_nitrous = 0.5*self.mass_vapour_nitrous_out[i]*(self.d_tank/2)**2

            self.motor_Izz.append(Izz_grain + Izz_liquid_nitrous + Izz_vapour_nitrous + Izz_empty_motor)
    
        return self.motor_Izz
        
    def Ixx(self,Ixx_empty_motor = 0,mass_empty_motor = 0,cm_empty_motor = 0,distance_grain = 0, distance_nitrous = 0):
        """
        Returns the motor's transversal moment of inertia vector (kg m^2) for a given time vector t (seconds)
        
        The moment of inertia is computed relative to a reference frame at the center of mass of the assembly.
        
        This function depends on center_of_mass function, so some of the inputs are actually required for the latter. Please reffer to it.
        
        If Ixx_empty_motor = 0, it returns only the axial moment of inertia of the propellant
        """
        self.motor_Ixx = []
                                   
        if np.all(self.time_vector == 0):
            len_time = len(self.sol.t)
        else:
            len_time = len(self.time_vector)
        
        cm = self.center_of_mass(time,mass_empty_motor,cm_empty_motor,distance_grain,distance_nitrous)
        
        for i in range(len_time):
            if distance_grain == 0:
                distance_grain = self.d_chamber + self.nozzle_length # A distance of one chamber diameter between grain and injector and between grain and nozzle is given for proper atomization and combustion, respectively.
            if distance_nitrous == 0:
                distance_nitrous = self.nozzle_length + 2*self.d_chamber + self.Lgrain
            
            d_cm_grain = distance_grain+self.Lgrain/2
            d_cm_liquid_nitrous = distance_nitrous + self.mass_liquid_nitrous_out[i]/(self.liquid_nitrous_density_out[i]*np.pi*(self.d_tank**2)/4)/2
            d_cm_vapour_nitrous = distance_nitrous + self.mass_liquid_nitrous_out[i]/(self.liquid_nitrous_density_out[i]*np.pi*(self.d_tank**2)/4) + self.mass_vapour_nitrous_out[i]/(self.vapour_nitrous_density_out[i]*np.pi*(self.d_tank**2)/4)/2
            
            liquid_nitrous_height = self.mass_liquid_nitrous_out[i]/(self.liquid_nitrous_density_out[i]*np.pi*(self.d_tank**2)/4)
            
            vapour_nitrous_height = self.mass_vapour_nitrous_out[i]/(self.vapour_nitrous_density_out[i]*np.pi*(self.d_tank**2)/4)
            
            mass_grain = self.rho_paraffin*(np.pi*(self.d_chamber**2-4*self.grain_radius_out[i]**2)/4)*self.Lgrain
            Ixx_grain = (1/12)*mass_grain*(3*(self.grain_radius_out[i]**2+(self.d_chamber/2)**2)+self.Lgrain**2) + mass_grain*(cm[i]-d_cm_grain)**2
            Ixx_liquid_nitrous = (1/12)*self.mass_liquid_nitrous_out[i]*(3*(self.d_tank/2)**2+liquid_nitrous_height**2) + self.mass_liquid_nitrous_out[i]*(cm[i]-d_cm_liquid_nitrous)**2
            Ixx_vapour_nitrous = (1/12)*self.mass_vapour_nitrous_out[i]*(3*(self.d_tank/2)**2+vapour_nitrous_height**2) + self.mass_vapour_nitrous_out[i]*(cm[i]-d_cm_vapour_nitrous)**2
            
            self.motor_Ixx.append(Ixx_grain + Ixx_liquid_nitrous + Ixx_vapour_nitrous + Ixx_empty_motor)
            
        return self.motor_Ixx 
    
    def plot_thrust(self):
        if 'self.thrust_vector' in globals():
            pass
        else:
            self.thrust_vector = self.thrust()
        if np.all(self.time_vector == 0):
            fig = plt.figure()
            fig.suptitle('Thrust Curve', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Thrust (N)', fontsize=18)
            plt.plot(self.sol.t,self.thrust_vector)
            plt.show()
        else:
            fig = plt.figure()
            fig.suptitle('Thrust Curve', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Thrust (N)', fontsize=18)
            plt.plot(self.time_vector,self.thrust_vector)
            plt.show()
        return None
    
    def plot_chamber_pressure(self):
        if np.all(self.time_vector == 0):
            fig = plt.figure()
            fig.suptitle('Chamber Pressure', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Pressure (Pa)', fontsize=18)
            plt.plot(self.sol.t,self.chamber_pressure_out)
            plt.show()
        else:
            fig = plt.figure()
            fig.suptitle('Chamber Pressure', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Pressure (Pa)', fontsize=18)
            plt.plot(self.time_vector,self.chamber_pressure_out)
            plt.show()
        return None

    def plot_tank_pressure(self):
        if np.all(self.time_vector == 0):
            fig = plt.figure()
            fig.suptitle('Tank Pressure', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Pressure (Pa)', fontsize=18)
            plt.plot(self.sol.t,self.tank_pressure_out)
            plt.show()
        else:
            fig = plt.figure()
            fig.suptitle('Tank Pressure', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Pressure (Pa)', fontsize=18)
            plt.plot(self.time_vector,self.tank_pressure_out)
            plt.show()
        return None
    
    def plot_tank_temperature(self):
        if np.all(self.time_vector == 0):
            fig = plt.figure()
            fig.suptitle('Tank Temperature', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Temperature (K)', fontsize=18)
            plt.plot(self.sol.t,self.tank_temperature_out)
            plt.show()
        else:
            fig = plt.figure()
            fig.suptitle('Tank Temperature', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Temperature (K)', fontsize=18)
            plt.plot(self.time_vector,self.tank_temperature_out)
            plt.show()
        return None
    
    def plot_grain_radius(self):
        if np.all(self.time_vector == 0):
            fig = plt.figure()
            fig.suptitle('Grain Radius', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Meters (m)', fontsize=18)
            plt.plot(self.sol.t,self.grain_radius_out)
            plt.show()
        else:
            fig = plt.figure()
            fig.suptitle('Grain Radius', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Meters (m)', fontsize=18)
            plt.plot(self.time_vector,self.grain_radius_out)
            plt.show()
        return None
    
    def plot_quality(self):
        if np.all(self.time_vector == 0):
            fig = plt.figure()
            fig.suptitle('Quality inside oxidiser tank', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Quality', fontsize=18)
            plt.plot(self.sol.t,self.quality_out)
            plt.show()
        else:
            fig = plt.figure()
            fig.suptitle('Quality inside oxidiser tank', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Quality', fontsize=18)
            plt.plot(self.time_vector,self.quality_out)
            plt.show()
        return None
    
    def plot_center_of_mass(self,mass_empty_motor = 0,cm_empty_motor = 0,distance_grain = 0,distance_nitrous = 0):
        if 'self.motor_center_of_mass' in globals():
            pass
        else:
            self.motor_center_of_mass = self.center_of_mass(mass_empty_motor,cm_empty_motor,distance_grain, distance_nitrous)
        if np.all(self.time_vector == 0):
            fig = plt.figure()
            fig.suptitle('Center of mass (relative to nozzle exit diameter)', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Meters (m)', fontsize=18)
            plt.plot(self.sol.t,self.motor_center_of_mass)
            plt.show()
        else:
            fig = plt.figure()
            fig.suptitle('Center of mass (relative to nozzle exit diameter)', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('Meters (m)', fontsize=18)
            plt.plot(self.time_vector,self.motor_center_of_mass)
            plt.show()                       
        return None
    
    def plot_Izz(self,Izz_empty_motor = 0):
        if 'self.motor_Izz' in globals():
            pass
        else:
            self.motor_Izz = self.Izz(Izz_empty_motor)
        if np.all(self.time_vector == 0):
            fig = plt.figure()
            fig.suptitle('Izz (axial moment of inertia)', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('kg m^2', fontsize=18)
            plt.plot(self.sol.t,self.motor_Izz)
            plt.show()
        else:
            fig = plt.figure()
            fig.suptitle('Izz (axial moment of inertia)', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('kg m^2', fontsize=18)
            plt.plot(self.time_vector,self.motor_Izz)
            plt.show()     
        return None
    
    def plot_Ixx(self,Ixx_empty_motor = 0,mass_empty_motor = 0,cm_empty_motor = 0,distance_grain = 0,distance_nitrous = 0):
        if 'self.motor_Ixx' in globals():
            pass
        else:
            self.motor_Ixx = self.Ixx(Ixx_empty_motor,mass_empty_motor,cm_empty_motor,distance_grain,distance_nitrous)
        if np.all(self.time_vector == 0):
            fig = plt.figure()
            fig.suptitle('Ixx (transversal moment of inertia)', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('kg m^2', fontsize=18)
            plt.plot(self.sol.t,self.motor_Ixx)
            plt.show()
        else:
            fig = plt.figure()
            fig.suptitle('Ixx (transversal moment of inertia)', fontsize=20)
            plt.xlabel('Seconds', fontsize=18)
            plt.ylabel('kg m^2', fontsize=18)
            plt.plot(self.time_vector,self.motor_Ixx)
            plt.show()
        return None