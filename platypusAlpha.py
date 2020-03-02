        #########################################
        # VAPOUR-ONLY PHASE (INSIDE OXIDISER TANK)
        #########################################
        
        ### Problems at vapour-phase only: I must impose the properties to be relative to the gas phase, because actually the quality is decreasing as the tank empties, which means there should be liquid being formed. However, I dont know if this should happen.
        
        ### Problems at tail-off: temperature decreasing below freezing point. I have put a condition to abort the simulation after it reaches this temperature


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

- Temperature inside the combustion chamber does not change along the
  burn

- Combustion chamber is adiabatic

- Oxidiser tank empties adiabatically

- Available volume of combustion chamber does not change

- Combustion products form a mixture which behaves like an ideal gas

- Ideal burning (no erosion along the burning port)

- O/F ratio change along the burn, and thus the combustion chamber temperature, the molar mass of the gaseous products and the ratio of specific heats also change. But these changes are only function of the O/F ratio change. Neither the chamber pressure nor the nitrous oxide temperature affect these latter variables (in real life they do, although not in a strong manner). The variables are relative to chamber pressure of 35 bar and nitrous oxide temperature of 298 K.

KNOW BEFORE RUNNING

- Some inputs should be provided by the user on the section below.

- Scipy and CoolProp needed.

- The ullage is computed as being 10% of the length of the tank occupied by the liquid nitrous. This percentage can be changed inside the code. The tank's diameter is an input.

- The vent of the oxidiser tank is considered to be on the top of the tank.

- For plotting different variables, proceed to the bottom of the code and follow the examples.

- Step can be reduced for further precision, but must be accompanied by a reduction on "while" conditions.

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
from scipy import integrate
from scipy.integrate import odeint
from scipy.integrate import simps
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
                 oxidiser_mass_multiplier = 1.2):

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
            print("CAUTION: Exit diameter bigger than chamber diameter")
        
        self.fuel_mass = self.fuel_flow*self.total_impulse/self.initial_thrust
    
        self.oxidiser_mass = oxidiser_mass_multiplier*self.oxidiser_flow*self.total_impulse/self.initial_thrust
        
        self.Lgrain = np.around(self.fuel_flow/(np.pi*rho_paraffin*self.d_grain*1.28*(10**-5)*(4*self.oxidiser_flow/(np.pi*self.d_grain**2))**0.94),2) #Length of the paraffin grain (estimated from the nominal oxidiser and fuel mass flows and from regression law found on LESTRADE,2014).
    
        self.IA = self.oxidiser_flow/(self.Cd*math.sqrt(2*PropsSI('D','T',self.ambient_temperature,'Q',0,'NITROUSOXIDE')*(PropsSI('P','T',self.ambient_temperature,'Q',1,'NITROUSOXIDE')-self.chamber_pressure)))
        
        self.nozzle_length = (self.d_chamber/2-self.throatRadius)/np.tan(np.pi*self.nozzle_conv_angle/180) + (self.exitRadius -self.throatRadius)/np.tan(np.pi*self.nozzle_div_angle/180)
        "NOTE: The author is aware that an entrance radius for the convergent part should be considered, but it will be neglected for the sake of simplicity in the design computation"
        
        self.chamber_free_volume = 2*d_chamber*(np.pi*d_chamber**2/4) + self.Lgrain*(np.pi*d_grain**2/4) # A distance of one chamber diameter between grain and injector and between grain and nozzle is given for proper atomization and combustion, respectively.
        
        self.tank_length_until_vent = 4*self.oxidiser_mass/(np.pi*(self.d_tank**2)*PropsSI('D','T',self.ambient_temperature,'Q',0,'NITROUSOXIDE'))
        
        self.tank_length = 1.1*self.tank_length_until_vent
        
        self.oxidiser_mass = self.oxidiser_mass + PropsSI('D','T',self.ambient_temperature,'Q',1,'NITROUSOXIDE')*(np.pi*d_tank**2/4)*(self.tank_length - self.tank_length_until_vent)
        
        self.tank_volume = self.tank_length*(np.pi*d_tank**2/4)
        
        self.VA = np.pi*(self.d_vent**2)/4
        
        return None
    
    def ignite(self):
    
        "SIMULATION"

        self.X = (PropsSI('D','T',self.ambient_temperature,'Q',1,'NITROUSOXIDE')*PropsSI('D','T',self.ambient_temperature,'Q',0,'NITROUSOXIDE')*self.tank_volume-PropsSI('D','T',self.ambient_temperature,'Q',1,'NITROUSOXIDE')*self.oxidiser_mass)/(self.oxidiser_mass*(PropsSI('D','T',self.ambient_temperature,'Q',0,'NITROUSOXIDE')-PropsSI('D','T',self.ambient_temperature,'Q',1,'NITROUSOXIDE'))) # Initial quality of nitrous oxide inside the tank
        if self.X < 0:
            print("Warning: X is less than zero! Increase tank volume or decrease initial mass")
            exit()

        "INITIALIZE MONITOR SOLUTIONS"

        self.mass_nitrous = [self.oxidiser_mass] # Available nitrous mass at each time instant, kg
        self.mass_liquid_nitrous = [(1-self.X)*self.oxidiser_mass] # Available liquid nitrous mass at each time instant, kg
        self.mass_vapour_nitrous = [self.X*self.oxidiser_mass] # Available vapour nitrous mass at each time instant, kg
        self.temperature_tank = [self.ambient_temperature] # Tank temperature at each time instant, Kelvin
        self.out_liquid_nitrous = [0] # Mass of liquid nitrous oxide that escapes through the injector, kg
        self.out_vapour_nitrous = [0] # Mass of vapour nitrous oxide that escapes through the injector, kg
        self.consumed_liquid = [0] # Total consumed liquid nitrous at each time instant, kg
        self.consumed_vapour = [0] # Total consumed vapour nitrous at each time instant, kg
        self.chamber_pressure = [self.ambient_pressure] # Combustion chamber pressure, Pa
        self.grain_radius = [self.d_grain/2] # Paraffin grain radius, m
        self.tank_pressure = [PropsSI('P','T',self.ambient_temperature,'Q',self.X,'NITROUSOXIDE')] # Oxidiser tank pressure, Pa
        
        self.specific_internal_energy = [PropsSI('U','T',self.ambient_temperature,'Q',self.X,'NITROUSOXIDE')] # Oxidiser tank internal specific energy, J/kg
        
        self.specific_enthalpy_liquid_nitrous = [PropsSI('H','T',self.ambient_temperature,'Q',0,'NITROUSOXIDE')] # Specific enthalpy of liquid nitrous, J/kg
        
        self.specific_enthalpy_vapour_nitrous = [PropsSI('H','T',self.ambient_temperature,'Q',1,'NITROUSOXIDE')] # Specific enthalpy of vapour nitrous, J/kg   
        
        self.overall_density = [self.oxidiser_mass/self.tank_volume] # Density of nitrous oxide inside the tank (considering both phases), kg/m3
        self.liquid_nitrous_density = [PropsSI('D','T',self.ambient_temperature,'Q',0,'NITROUSOXIDE')] # Density of liquid nitrous oxide inside the tank, kg/m3
        self.vapour_nitrous_density = [PropsSI('D','T',self.ambient_temperature,'Q',1,'NITROUSOXIDE')] # Density of vapour nitrous oxide inside the tank, kg/m3
        self.quality = [self.X] # Quality of the satured nitrous oxide
        self.thrust_vector = [0] # Thrust generated by the motor, N
        self.cf = [0] # Thrust coefficient

        ########################################
        # TWO-PHASE FLOW (INSIDE OXIDISER TANK)
        ########################################

        self.step = 0.01
        self.t = []

        self.x0 = [self.chamber_pressure[0],self.grain_radius[0],0,0]

        self.ysol = []

        self.i = 0
        
        while self.mass_liquid_nitrous[self.i] > 0.01:
            self.ysol.append(self.x0)
            self.ts = [self.step*self.i,self.step*(self.i+1)]
            self.y = odeint(self.two_phase,self.x0,self.ts)
            self.x0 = self.y[1,:].tolist()
            self.t.append(self.i*self.step)
            self.i = self.i + 1
            self.out_liquid_nitrous.append(self.x0[2]-self.consumed_liquid[self.i-1])
            self.out_vapour_nitrous.append(self.x0[3]-self.consumed_vapour[self.i-1])
            
            self.mass_nitrous.append(self.mass_nitrous[self.i-1]-self.out_liquid_nitrous[self.i]-self.out_vapour_nitrous[self.i])
            
            self.specific_internal_energy.append( (self.mass_nitrous[self.i-1]*self.specific_internal_energy[self.i-1]-self.out_liquid_nitrous[self.i]*self.specific_enthalpy_liquid_nitrous[self.i-1] - self.out_vapour_nitrous[self.i]*self.specific_enthalpy_vapour_nitrous[self.i-1])/self.mass_nitrous[self.i])
            
            self.overall_density.append(self.mass_nitrous[self.i]/self.tank_volume)
            self.temperature_tank.append(PropsSI('T','U', self.specific_internal_energy[self.i],'D',self.overall_density[self.i],'NITROUSOXIDE'))
            self.quality.append(PropsSI('Q','D',self.overall_density[self.i],'T',self.temperature_tank[self.i],'NITROUSOXIDE'))
            
            self.specific_enthalpy_liquid_nitrous.append(PropsSI('H','T',self.temperature_tank[self.i],'Q',0,'NITROUSOXIDE'))
        
            self.specific_enthalpy_vapour_nitrous.append(PropsSI('H','T',self.temperature_tank[self.i],'Q',1,'NITROUSOXIDE'))  
            
            self.mass_liquid_nitrous.append((1-self.quality[self.i])*self.mass_nitrous[self.i])
            self.mass_vapour_nitrous.append(self.quality[self.i]*self.mass_nitrous[self.i])
            self.consumed_liquid.append(self.x0[2])
            self.consumed_vapour.append(self.x0[3])
            self.chamber_pressure.append(self.x0[0])
            self.grain_radius.append(self.x0[1])
            self.tank_pressure.append(PropsSI('P','T',self.temperature_tank[self.i],"Q",self.quality[self.i],'NITROUSOXIDE'))
            self.liquid_nitrous_density.append(PropsSI('D','T|liquid',self.temperature_tank[self.i],"P",self.tank_pressure[self.i],'NITROUSOXIDE'))
            self.vapour_nitrous_density.append(PropsSI('D','T|gas',self.temperature_tank[self.i],"P",self.tank_pressure[self.i],'NITROUSOXIDE'))
            self.cf.append((((2*self.k**2)/(self.k-1))*(2/(self.k+1))**((self.k+1)/(self.k-1))*(1-(self.ambient_pressure/self.chamber_pressure[self.i])**((self.k-1)/self.k)))**0.5)
            self.thrust_vector.append(self.expansion_eff*self.cf[self.i]*(np.pi*((2*self.throatRadius)**2)/4)*self.chamber_pressure[self.i])

        #########################################
        # VAPOUR-ONLY PHASE (INSIDE OXIDISER TANK)
        #########################################
        
        self.quality_assess = []
        
        #self.mass_liquid_nitrous[self.i] = 0
        #self.mass_vapour_nitrous[self.i] = self.mass_nitrous[self.i]
        #self.quality[self.i] = 1
        #self.overall_density[self.i] = PropsSI('D','T', self.temperature_tank[self.i],'Q',1,'NITROUSOXIDE')
        #self.tank_pressure[self.i] = (PropsSI('P','T', self.temperature_tank[self.i],'Q',1,'NITROUSOXIDE'))
        
        print(self.i)

        self.x0 = [self.chamber_pressure[self.i],self.grain_radius[self.i],self.consumed_vapour[self.i]]
        self.ysol = []

        while self.tank_pressure[self.i]>self.ambient_pressure and self.chamber_pressure[self.i] > self.ambient_pressure:
            self.ysol.append(self.x0)
            self.ts = [self.step*self.i,self.step*(self.i+1)]
            self.y = odeint(self.vapour_phase,self.x0,self.ts)
            self.x0 = self.y[1,:].tolist()
            self.t.append(self.i*self.step)
            self.i = self.i + 1
            self.out_liquid_nitrous.append(0)
            self.out_vapour_nitrous.append(self.x0[2]-self.consumed_vapour[self.i-1])
            
            self.mass_nitrous.append(self.mass_nitrous[self.i-1]-self.out_vapour_nitrous[self.i])
            
            self.specific_internal_energy.append( (self.mass_nitrous[self.i-1]*self.specific_internal_energy[self.i-1]-self.out_liquid_nitrous[self.i]*self.specific_enthalpy_liquid_nitrous[self.i-1] - self.out_vapour_nitrous[self.i]*self.specific_enthalpy_vapour_nitrous[self.i-1])/self.mass_nitrous[self.i])

            self.overall_density.append(self.mass_nitrous[self.i]/self.tank_volume)
            self.temperature_tank.append(PropsSI('T','U', self.specific_internal_energy[self.i],'D',self.overall_density[self.i],'NITROUSOXIDE'))         
            self.quality.append(1)
            self.quality_assess.append(PropsSI('Q','D',self.overall_density[self.i],'T',self.temperature_tank[self.i],'NITROUSOXIDE'))
            self.mass_liquid_nitrous.append(0)
            self.mass_vapour_nitrous.append(self.mass_nitrous[self.i])
            self.consumed_liquid.append(self.consumed_liquid[self.i-1])
            self.consumed_vapour.append(self.x0[2])
            #self.chamber_pressure.append(self.x0[0])
            self.grain_radius.append(self.x0[1])
            self.tank_pressure.append(PropsSI('P','T|gas',self.temperature_tank[self.i],'D',self.overall_density[self.i],'NITROUSOXIDE')) # MUDEI AQUI 28/01 23h55 TROQUEI QUALIDADE POR TEMPERATURA - NAO E MAIS VAPOR SATURADO, PODE SER VAPOR SUPERAQUECIDO - Q = 1 NAO VALE MAIS!!!
            
            self.specific_enthalpy_liquid_nitrous.append(PropsSI('H','T|liquid',self.temperature_tank[self.i],'P',self.tank_pressure[self.i],'NITROUSOXIDE'))
        
            self.specific_enthalpy_vapour_nitrous.append(PropsSI('H','T|gas',self.temperature_tank[self.i],'P',self.tank_pressure[self.i],'NITROUSOXIDE'))
            self.liquid_nitrous_density.append(PropsSI('D','T|liquid',self.temperature_tank[self.i],"P",self.tank_pressure[self.i],'NITROUSOXIDE'))
            self.vapour_nitrous_density.append(PropsSI('D','T|gas',self.temperature_tank[self.i],"P",self.tank_pressure[self.i],'NITROUSOXIDE'))
            
            if self.ambient_pressure < self.x0[0]:
                self.chamber_pressure.append(self.x0[0])
                self.cf.append((((2*self.k**2)/(self.k-1))*(2/(self.k+1))**((self.k+1)/(self.k-1))*(1-(self.ambient_pressure/self.chamber_pressure[self.i])**((self.k-1)/self.k)))**0.5)
                self.thrust_vector.append(self.expansion_eff*self.cf[self.i]*(np.pi*((2*self.throatRadius)**2)/4)*self.chamber_pressure[self.i])
                
            else:
                self.chamber_pressure.append(self.ambient_pressure)
                self.cf.append(0)
                self.thrust_vector.append(0)

        ################
        # TAIL-OFF PHASE
        ################      
        
        self.ysol = []
        if self.chamber_pressure[self.i] > self.ambient_pressure:
            self.x0 = self.chamber_pressure[self.i]
        if self.mass_vapour_nitrous[self.i] > 0.01:
            self.x0 = self.consumed_vapour[self.i]
       
        while (self.chamber_pressure[self.i] > self.ambient_pressure or self.tank_pressure[self.i] > self.ambient_pressure) and (self.temperature_tank[self.i] > 183):
            self.ysol.append(self.x0)
            self.ts = [self.step*self.i,self.step*(self.i+1)]
            self.y = odeint(self.tail_off,self.x0,self.ts)
            self.t.append(self.i*self.step)
            self.x0 = self.y[1].tolist()
            self.x0 = self.x0[0]
            self.i = self.i + 1
            if self.chamber_pressure[self.i-1] > self.ambient_pressure:
                self.chamber_pressure.append(self.x0)
                self.consumed_vapour.append(self.consumed_vapour[self.i-1])
                self.mass_vapour_nitrous.append(self.mass_vapour_nitrous[self.i-1])
                self.mass_nitrous.append(self.mass_nitrous[self.i-1])
                self.tank_pressure.append(self.tank_pressure[self.i-1])
                self.specific_internal_energy.append(self.specific_internal_energy[self.i-1])
                self.overall_density.append(self.overall_density[self.i-1])
                self.temperature_tank.append(self.temperature_tank[self.i-1])
                self.specific_enthalpy_liquid_nitrous.append(self.specific_enthalpy_liquid_nitrous[self.i-1])
                self.specific_enthalpy_vapour_nitrous.append(self.specific_enthalpy_vapour_nitrous[self.i-1])
                self.liquid_nitrous_density.append(self.liquid_nitrous_density[self.i-1])
                self.vapour_nitrous_density.append(self.vapour_nitrous_density[self.i-1])
            if self.mass_vapour_nitrous[self.i-1] > 0.01:
                self.chamber_pressure.append(self.chamber_pressure[self.i-1])
                self.consumed_vapour.append(self.x0)
                self.out_vapour_nitrous.append(self.x0-self.consumed_vapour[self.i-1])
                self.out_liquid_nitrous.append(0)
                self.mass_nitrous.append(self.mass_nitrous[self.i-1]-self.out_vapour_nitrous[self.i])
                self.mass_vapour_nitrous.append(self.mass_nitrous[self.i])
                
                self.specific_internal_energy.append( (self.mass_nitrous[self.i-1]*self.specific_internal_energy[self.i-1]-self.out_liquid_nitrous[self.i]*self.specific_enthalpy_liquid_nitrous[self.i-1] - self.out_vapour_nitrous[self.i]*self.specific_enthalpy_vapour_nitrous[self.i-1])/self.mass_nitrous[self.i])
                
                self.overall_density.append(self.mass_nitrous[self.i]/self.tank_volume)
                self.temperature_tank.append(PropsSI('T','U|gas',self.specific_internal_energy[self.i],'D',self.overall_density[self.i],'NITROUSOXIDE'))
                self.tank_pressure.append(PropsSI('P','T|gas',self.temperature_tank[self.i],'D',self.overall_density[self.i],'NITROUSOXIDE'))
                
                self.specific_enthalpy_liquid_nitrous.append(PropsSI('H','T|liquid',self.temperature_tank[self.i],'P',self.tank_pressure[self.i],'NITROUSOXIDE'))
        
            self.specific_enthalpy_vapour_nitrous.append(PropsSI('H','T|gas',self.temperature_tank[self.i],'P',self.tank_pressure[self.i],'NITROUSOXIDE'))
            self.liquid_nitrous_density.append(PropsSI('D','T|liquid',self.temperature_tank[self.i],"P",self.tank_pressure[self.i],'NITROUSOXIDE'))

            self.vapour_nitrous_density.append(PropsSI('D','T|gas',self.temperature_tank[self.i],"P",self.tank_pressure[self.i],'NITROUSOXIDE'))
            self.quality.append(1)
            self.mass_liquid_nitrous.append(self.mass_liquid_nitrous[self.i-1])
            self.consumed_liquid.append(self.consumed_liquid[self.i-1])
            self.grain_radius.append(self.grain_radius[self.i-1])
            if self.ambient_pressure < self.chamber_pressure[self.i]:
                self.cf.append((((2*self.k**2)/(self.k-1))*(2/(self.k+1))**((self.k+1)/(self.k-1))*(1-(self.ambient_pressure/self.chamber_pressure[self.i])**((self.k-1)/self.k)))**0.5)
                self.thrust_vector.append(self.expansion_eff*self.cf[self.i]*(np.pi*((2*self.throatRadius)**2)/4)*self.chamber_pressure[self.i])
            else:
                self.cf.append(0)
                self.thrust_vector.append(0)
                        
        self.burnTime = self.t[len(self.t)-1]
        
        self.totalI = simps(self.thrust_vector, dx=self.step)
    
        return None
    
    ################
    # FUNCTIONS
    ################
                
    def two_phase(self,x,t):
        "x[0] is the chamber pressure Po, Pascal"
        "x[1] is the radius of Paraffin Grain, m"
        "x[2] is the used liquid oxidiser mass, kg"
        "x[3] is the used vapour oxidiser mass, kg"
        "T is the temperature inside the tank, in Kelvin"
        gamma_vapour =  PropsSI('C','T',self.temperature_tank[self.i],'Q',1,'NITROUSOXIDE')/PropsSI('CVMASS','T',self.temperature_tank[self.i],'Q',1,'NITROUSOXIDE')
        oxidiser_flow_rate = self.Cd*self.IA*(2*self.liquid_nitrous_density[self.i]*(self.tank_pressure[self.i]-x[0]))**0.5
        fuel_flow_rate = self.Lgrain*2*np.pi*x[1]*self.rho_paraffin*1.28*10**(-5)*(oxidiser_flow_rate/(np.pi*(x[1]**2)))**0.94
        self.chamber_temperature = self.gas_combustion(oxidiser_flow_rate/fuel_flow_rate,1)
        self.molar_mass = self.gas_combustion(oxidiser_flow_rate/fuel_flow_rate,2)
        self.k = self.gas_combustion(oxidiser_flow_rate/fuel_flow_rate,3)
        self.R = 8314.51/self.molar_mass
        rho_gas = x[0]*self.molar_mass/(8314*self.chamber_temperature);
        dx0 = (self.R*self.chamber_temperature/self.chamber_free_volume)*(oxidiser_flow_rate+self.Lgrain*2*np.pi*x[1]*(self.rho_paraffin-rho_gas)*1.28*10**(-5)*((oxidiser_flow_rate/(np.pi*(x[1]**2)))**0.94)-x[0]/self.combustion_eff*(np.pi*((2*self.throatRadius)**2)/4)*((self.k/(self.R*self.chamber_temperature))*(2/(self.k+1))**((self.k+1)/(self.k-1)))**0.5)
        dx1 = 1.28*10**(-5)*(oxidiser_flow_rate/(np.pi*(x[1]**2)))**0.94
        dx2 = oxidiser_flow_rate
        if self.tank_pressure[self.i] > self.ambient_pressure*(1+(gamma_vapour-1)/2)**(gamma_vapour/(gamma_vapour-1)):
            dx3 = self.VA*self.Cd_Vent*(gamma_vapour*self.tank_pressure[self.i]*self.vapour_nitrous_density[self.i]*(2/(gamma_vapour+1))**( (gamma_vapour+1)/(gamma_vapour-1) ))**0.5
        else:
            dx3 = self.VA*self.Cd_Vent*(2*gamma_vapour/(gamma_vapour-1)*self.tank_pressure[self.i]*self.vapour_nitrous_density[self.i]*((self.ambient_pressure/self.tank_pressure[self.i])**(2/gamma_vapour)- (self.ambient_pressure/self.tank_pressure[self.i])**((gamma_vapour+1)/gamma_vapour)) )**0.5
        
        dx = [dx0, dx1, dx2, dx3]
        return dx            
                
    def vapour_phase(self,x,t):
        "x[0] is the chamber pressure Po, Pascal"
        "x[1] is the radius of Paraffin Grain, m"
        "x[2] is the consumed vapour oxidiser mass, kg"
        "P is the pressure inside the tank, in Pascal"
        gamma_vapour = PropsSI('C','T',self.temperature_tank[self.i],'D',self.vapour_nitrous_density[self.i],'NITROUSOXIDE')/PropsSI('CVMASS','T',self.temperature_tank[self.i],'D',self.vapour_nitrous_density[self.i],'NITROUSOXIDE')
        if self.tank_pressure[self.i] > self.chamber_pressure[self.i]*(1+(gamma_vapour-1)/2)**(gamma_vapour/(gamma_vapour-1)):
            # Choked Flow
            oxidiser_flow_rate = self.Cd*self.IA*(gamma_vapour*self.tank_pressure[self.i]*self.vapour_nitrous_density[self.i]*(2/(gamma_vapour+1))**( (gamma_vapour+1)/(gamma_vapour-1) ))**0.5
        else:
            # Subsonic Flow
            oxidiser_flow_rate = self.Cd*self.IA*(2*gamma_vapour/(gamma_vapour-1)*self.tank_pressure[self.i]*self.vapour_nitrous_density[self.i]*((self.chamber_pressure[self.i]/self.tank_pressure[self.i])**(2/gamma_vapour)- (self.chamber_pressure[self.i]/self.tank_pressure[self.i])**((gamma_vapour+1)/gamma_vapour)) )**0.5
        
        fuel_flow_rate = self.Lgrain*2*np.pi*x[1]*self.rho_paraffin*1.28*10**(-5)*(oxidiser_flow_rate/(np.pi*(x[1]**2)))**0.94
        self.chamber_temperature = self.gas_combustion(oxidiser_flow_rate/fuel_flow_rate,1)
        self.molar_mass = self.gas_combustion(oxidiser_flow_rate/fuel_flow_rate,2)
        self.k = self.gas_combustion(oxidiser_flow_rate/fuel_flow_rate,3)
        self.R = 8314.51/self.molar_mass
        rho_gas = x[0]*self.molar_mass/(8314*self.chamber_temperature);
        dx0 = (self.R*self.chamber_temperature/self.chamber_free_volume)*(oxidiser_flow_rate+self.Lgrain*2*np.pi*x[1]*(self.rho_paraffin-rho_gas)*1.28*10**(-5)*((oxidiser_flow_rate/(np.pi*(x[1]**2)))**0.94)-x[0]/self.combustion_eff*(np.pi*((2*self.throatRadius)**2)/4)*((self.k/(self.R*self.chamber_temperature))*(2/(self.k+1))**((self.k+1)/(self.k-1)))**0.5)
        dx1 = 1.28*10**(-5)*(oxidiser_flow_rate/(np.pi*(x[1]**2)))**0.94
        if self.tank_pressure[self.i] > self.ambient_pressure*(1+(gamma_vapour-1)/2)**(gamma_vapour/(gamma_vapour-1)):
            dx2 = oxidiser_flow_rate + self.VA*self.Cd_Vent*(gamma_vapour*self.tank_pressure[self.i]*self.vapour_nitrous_density[self.i]*(2/(gamma_vapour+1))**( (gamma_vapour+1)/(gamma_vapour-1) ))**0.5
        else:
            dx2 = oxidiser_flow_rate + self.VA*self.Cd_Vent*(2*gamma_vapour/(gamma_vapour-1)*self.tank_pressure[self.i]*self.vapour_nitrous_density[self.i]*((self.ambient_pressure/self.tank_pressure[self.i])**(2/gamma_vapour)- (self.ambient_pressure/self.tank_pressure[self.i])**((gamma_vapour+1)/gamma_vapour)) )**0.5
        dx = [dx0, dx1, dx2]
        return dx
            
    def tail_off(self,x,t):
        if self.chamber_pressure[self.i] > self.ambient_pressure:
            "x is the chamber pressure Po, Pascal"
            rho_gas = x*self.molar_mass/(8314*self.chamber_temperature);
            dx0 = (self.R*self.chamber_temperature/self.chamber_free_volume)*(-x/self.combustion_eff*(np.pi*((2*self.throatRadius)**2)/4)*((self.k/(self.R*self.chamber_temperature))*(2/(self.k+1))**((self.k+1)/(self.k-1)))**0.5)
            return dx0
        if self.mass_vapour_nitrous[self.i] > 0.01:
            "x is the consumed vapour nitrous, kg"
            gamma_vapour =  PropsSI('C','T|gas',self.temperature_tank[self.i],'P',self.tank_pressure[self.i],'NITROUSOXIDE')/PropsSI('CVMASS','T|gas',self.temperature_tank[self.i],'P',self.tank_pressure[self.i],'NITROUSOXIDE')
            if self.tank_pressure[self.i] > self.chamber_pressure[self.i]*(1+(gamma_vapour-1)/2)**(gamma_vapour/(gamma_vapour-1)):
            # Choked Flow
                oxidiser_flow_rate = self.Cd*self.IA*(gamma_vapour*self.tank_pressure[self.i]*self.vapour_nitrous_density[self.i]*(2/(gamma_vapour+1))**( (gamma_vapour+1)/(gamma_vapour-1) ))**0.5
            else:
            # Subsonic Flow
                oxidiser_flow_rate = self.Cd*self.IA*(2*gamma_vapour/(gamma_vapour-1)*self.tank_pressure[self.i]*self.vapour_nitrous_density[self.i]*((self.chamber_pressure[self.i]/self.tank_pressure[self.i])**(2/gamma_vapour)- (self.chamber_pressure[self.i]/self.tank_pressure[self.i])**((gamma_vapour+1)/gamma_vapour)) )**0.5
            if self.tank_pressure[self.i] > self.ambient_pressure*(1+(gamma_vapour-1)/2)**(gamma_vapour/(gamma_vapour-1)):
                dx0 = oxidiser_flow_rate + self.VA*self.Cd_Vent*(gamma_vapour*self.tank_pressure[self.i]*self.vapour_nitrous_density[self.i]*(2/(gamma_vapour+1))**( (gamma_vapour+1)/(gamma_vapour-1) ))**0.5
            else:
                dx0 = oxidiser_flow_rate + self.VA*self.Cd_Vent*(2*gamma_vapour/(gamma_vapour-1)*self.tank_pressure[self.i]*self.vapour_nitrous_density[self.i]*((self.ambient_pressure/self.tank_pressure[self.i])**(2/gamma_vapour)- (self.ambient_pressure/self.tank_pressure[self.i])**((gamma_vapour+1)/gamma_vapour)) )**0.5    
            return dx0
        
    def gas_combustion(self,OFratio,index):
        """
        Given an OFratio (nitrous oxide flow rate/paraffin flow rate), the function returns 
        properties from the combustion chamber gas, namely temperature (K), molar mass (kg/kmol) and ratio of specific heats
        Data retrieved NASA CEA
        """
        
        OF_list = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16]
        
        temperature_list = [0,1066.89,1243.39,1409.27,1615.65,1822.91,2000.79,2296.05,2594.58,2833.88,3020.84,3158.77,3251.37,3306.76,3335.60,3347.07,3347.50,3340.88,3329.70,3315.53,3299.36,3281.87,3263.51,3244.61,3225.37,3205.98,3186.55,3167.17,3147.91,3128.83,3109.96,3091.33,3072.97]
        
        molar_mass_list = [24,24.389,21.197,20.066,19.882,20.032,20.220,20.994,22.125,23.118,23.981,24.718,25.335,25.843,26.263,26.614,26.912,27.168,27.389,27.583,27.754,27.905,28.040,28.160,28.267,28.364,28.450,28.528,28.599,28.663,28.721,28.773,28.821]
        
        gamma_list = [1.12,1.1276,1.1607,1.2113,1.2534,1.2656,1.2643,1.2851,1.2613,1.2382,1.2156,1.1938,1.1755,1.1627,1.1547,1.1501,1.1475,1.1462,1.1456,1.1456,1.1460,1.1466,1.1474,1.1484,1.1495,1.1508,1.1522,1.1536,1.1551,1.1567,1.1583,1.1600,1.1616]
        
        temperature = np.interp(OFratio, OF_list, temperature_list)
        
        molar_mass = np.interp(OFratio, OF_list, molar_mass_list)
        
        gamma = np.interp(OFratio, OF_list, gamma_list)
        
        if index == 1:
            return temperature
        if index == 2:
            return molar_mass
        if index == 3:
            return gamma
            
                
    ################
    # OUTPUTS
    ################    
    
    def thrust(self,time):
        """
        Returns the thrust value (N) at a given instant of time t (seconds)
        """
        if 0 <= time <= self.t[len(self.t)-1]:
            index = int(time/self.step)
            return self.thrust_vector[index]
        else:
            print("Instant of time out of range!")
            return None
        
    def mass(self,time,mass_empty_motor = 0):
        """
        Returns the motor mass (kg) at a given instant of time t (seconds)
        
        If mass_empty_motor = 0, it returns only the propellant mass
        """
        if 0 <= time <= self.t[len(self.t)-1]:
            index = int(time/self.step)
            if mass_empty_motor == 0:
                return self.mass_nitrous[index] + (self.d_chamber**2-(2*self.grain_radius[index])**2)*np.pi/4*self.Lgrain*self.rho_paraffin
            else:
                return self.mass_nitrous[index] + (self.d_chamber**2-(2*self.grain_radius[index])**2)*np.pi/4*self.Lgrain*self.rho_paraffin + mass_empty_motor
        else:
            print("Instant of time out of range!")
            return None
        
    def center_of_mass(self,time,mass_empty_motor = 0,cm_empty_motor = 0,distance_grain = 0, distance_nitrous = 0):
        """
        Returns the motor's center of mass (m) at a given instant of time t (seconds)
        
        The center of mass is computed with respect to a reference frame at the center of the nozzle's exit diameter
        
        If mass_empty_motor == 0 or cm_empty_motor == 0, it returns only the propellant's center of mass
        
        distance_grain refers to the distance the bottom part of the paraffin grain is from the reference frame. If distance_grain = 0, it will approximately compute this distance according to prior calculations.
        
        distance_nitrous refers to the distance the bottom part of the nitrous oxide 'cylinder' is from the reference frame. If distance_nitrous = 0, it will approximately compute this distance according to prior calculations.
        """
        if 0 <= time <= self.t[len(self.t)-1]:
            index = int(time/self.step)
            if distance_grain == 0:
                distance_grain = self.d_chamber + self.nozzle_length # A distance of one chamber diameter between grain and injector and between grain and nozzle is given for proper atomization and combustion, respectively.
            if distance_nitrous == 0:
                distance_nitrous = self.nozzle_length + 2*self.d_chamber + self.Lgrain
            
            d_cm_grain = distance_grain+self.Lgrain/2
            d_cm_liquid_nitrous = distance_nitrous + self.mass_liquid_nitrous[index]/(self.liquid_nitrous_density[index]*np.pi*(self.d_tank**2)/4)/2
            d_cm_vapour_nitrous = distance_nitrous + self.mass_liquid_nitrous[index]/(self.liquid_nitrous_density[index]*np.pi*(self.d_tank**2)/4) + self.mass_vapour_nitrous[index]/(self.vapour_nitrous_density[index]*np.pi*(self.d_tank**2)/4)/2
            
            mass_grain = self.rho_paraffin*(np.pi*(self.d_chamber**2-4*self.grain_radius[index]**2)/4)*self.Lgrain
            if mass_empty_motor == 0 or cm_empty_motor == 0:
                return (mass_grain*d_cm_grain + self.mass_liquid_nitrous[index]*d_cm_liquid_nitrous + self.mass_vapour_nitrous[index]*d_cm_vapour_nitrous)/(mass_grain+self.mass_liquid_nitrous[index]+self.mass_vapour_nitrous[index])
            else:
                return (mass_grain*d_cm_grain + self.mass_liquid_nitrous[index]*d_cm_liquid_nitrous + self.mass_vapour_nitrous[index]*d_cm_vapour_nitrous + mass_empty_motor*cm_empty_motor)/(mass_grain+self.mass_liquid_nitrous[index]+self.mass_vapour_nitrous[index]+mass_empty_motor)
            
        else:
            print("Instant of time out of range!")
            return None
        
    def Izz(self,time,Izz_empty_motor = 0):
        """
        Returns the axial moment of inertia of the motor (kg m^2) at a given instant of time t (seconds)
        
        The moment of inertia is computed relative to a reference frame at the center of mass of the assembly.
        
        If Izz_empty_motor == 0, it returns only the axial moment of inertia of the propellant
        """
        if 0 <= time <= self.t[len(self.t)-1]:
            index = int(time/self.step)
            
            mass_grain = self.rho_paraffin*(np.pi*(self.d_chamber**2-4*self.grain_radius[index]**2)/4)*self.Lgrain
            Izz_grain = 0.5*mass_grain*(self.grain_radius[index]**2+(self.d_chamber/2)**2)
            
            Izz_liquid_nitrous = 0.5*self.mass_liquid_nitrous[index]*(self.d_tank/2)**2
            
            Izz_vapour_nitrous = 0.5*self.mass_vapour_nitrous[index]*(self.d_tank/2)**2
            
            return Izz_grain + Izz_liquid_nitrous + Izz_vapour_nitrous + Izz_empty_motor
        else:
            print("Instant of time out of range!")
            return None
        
    def Ixx(self,time,Ixx_empty_motor = 0,mass_empty_motor = 0,cm_empty_motor = 0,distance_grain = 0, distance_nitrous = 0):
        """
        Returns the transversal moment of inertia of the motor (kg m^2) at a given instant of time t
        
        The moment of inertia is computed relative to a reference frame at the center of mass of the assembly.
        
        This function depends on center_of_mass function, so some of the inputs are actually required for the latter. Please reffer to it.
        
        If Ixx_empty_motor = 0, it returns only the axial moment of inertia of the propellant
        """
        if 0 <= time <= self.t[len(self.t)-1]:
            index = int(time/self.step)
            cm = self.center_of_mass(time,mass_empty_motor,cm_empty_motor,distance_grain,distance_nitrous)
            
            if distance_grain == 0:
                distance_grain = self.d_chamber + self.nozzle_length # A distance of one chamber diameter between grain and injector and between grain and nozzle is given for proper atomization and combustion, respectively.
            if distance_nitrous == 0:
                distance_nitrous = self.nozzle_length + 2*self.d_chamber + self.Lgrain
            
            d_cm_grain = distance_grain+self.Lgrain/2
            d_cm_liquid_nitrous = distance_nitrous + self.mass_liquid_nitrous[index]/(self.liquid_nitrous_density[index]*np.pi*(self.d_tank**2)/4)/2
            d_cm_vapour_nitrous = distance_nitrous + self.mass_liquid_nitrous[index]/(self.liquid_nitrous_density[index]*np.pi*(self.d_tank**2)/4) + self.mass_vapour_nitrous[index]/(self.vapour_nitrous_density[index]*np.pi*(self.d_tank**2)/4)/2
            
            liquid_nitrous_height = self.mass_liquid_nitrous[index]/(self.liquid_nitrous_density[index]*np.pi*(self.d_tank**2)/4)
            
            vapour_nitrous_height = self.mass_vapour_nitrous[index]/(self.vapour_nitrous_density[index]*np.pi*(self.d_tank**2)/4)
            
            mass_grain = self.rho_paraffin*(np.pi*(self.d_chamber**2-4*self.grain_radius[index]**2)/4)*self.Lgrain
            Ixx_grain = (1/12)*mass_grain*(3*(self.grain_radius[index]**2+(self.d_chamber/2)**2)+self.Lgrain**2) + mass_grain*(cm-d_cm_grain)**2
            Ixx_liquid_nitrous = (1/12)*self.mass_liquid_nitrous[index]*(3*(self.d_tank/2)**2+liquid_nitrous_height**2) + self.mass_liquid_nitrous[index]*(cm-d_cm_liquid_nitrous)**2
            Ixx_vapour_nitrous = (1/12)*self.mass_vapour_nitrous[index]*(3*(self.d_tank/2)**2+vapour_nitrous_height**2) + self.mass_vapour_nitrous[index]*(cm-d_cm_vapour_nitrous)**2
            
            return Ixx_grain + Ixx_liquid_nitrous + Ixx_vapour_nitrous + Ixx_empty_motor
        else:
            print("Instant of time out of range!")
            return None 
    
    def plot_thrust(self):
        fig = plt.figure()
        fig.suptitle('Thrust Curve', fontsize=20)
        plt.xlabel('Seconds', fontsize=18)
        plt.ylabel('Thrust (N)', fontsize=18)
        plt.plot(self.t,self.thrust_vector[0:len(self.thrust_vector)-1])
        plt.show()
        return None
    
    def plot_chamber_pressure(self):
        fig = plt.figure()
        fig.suptitle('Thrust Curve', fontsize=20)
        plt.xlabel('Seconds', fontsize=18)
        plt.ylabel('Thrust (N)', fontsize=18)
        plt.plot(self.t,self.chamber_pressure[0:len(self.chamber_pressure)-1])
        plt.show()
        return None

    def plot_tank_pressure(self):
        fig2 = plt.figure()
        fig2.suptitle('Tank Pressure', fontsize=20)
        plt.xlabel('Seconds', fontsize=18)
        plt.ylabel('Pressure (Pa)', fontsize=18)
        plt.plot(self.t,self.tank_pressure[0:len(self.tank_pressure)-1])
        plt.show()
        return None
    
    def plot_tank_temperature(self):
        fig2 = plt.figure()
        fig2.suptitle('Tank Temperature', fontsize=20)
        plt.xlabel('Seconds', fontsize=18)
        plt.ylabel('Kelvin (K)', fontsize=18)
        plt.plot(self.t,self.temperature_tank[0:len(self.temperature_tank)-1])
        plt.show()
        return None
    
    def plot_grain_radius(self):
        fig2 = plt.figure()
        fig2.suptitle('Grain Radius', fontsize=20)
        plt.xlabel('Seconds', fontsize=18)
        plt.ylabel('Meters (m)', fontsize=18)
        plt.plot(self.t,self.grain_radius[0:len(self.grain_radius)-1])
        plt.show()
        return None
    
    def plot_quality(self):
        fig2 = plt.figure()
        fig2.suptitle('Quality inside oxidiser tank', fontsize=20)
        plt.xlabel('Seconds', fontsize=18)
        plt.plot(self.t,self.quality[0:len(self.quality)-1])
        plt.show()
        return None
    
    def plot_center_of_mass(self,mass_empty_motor = 0,cm_empty_motor = 0,distance_grain = 0,distance_nitrous = 0):
        i = 0
        length = len(self.t)
        self.cm_vector = []
        while i < length:
            time = i*self.step
            self.cm_vector.append(self.center_of_mass(time,mass_empty_motor,cm_empty_motor,distance_grain, distance_nitrous))
            i = i + 1
        fig = plt.figure()
        fig.suptitle('Center of mass (relative to nozzle exit diameter)', fontsize=20)
        plt.xlabel('Seconds', fontsize=18)
        plt.ylabel('Meters (m)', fontsize=18)
        plt.plot(self.t,self.cm_vector)
        plt.show()
        return None
    
    def plot_Izz(self,Izz_empty_motor = 0):
        i = 0
        length = len(self.t)
        self.Izz_vector = []
        while i < length:
            time = i*self.step
            self.Izz_vector.append(self.Izz(time,Izz_empty_motor))
            i = i + 1
        fig = plt.figure()
        fig.suptitle('Izz (axial moment of inertia)', fontsize=20)
        plt.xlabel('Seconds', fontsize=18)
        plt.ylabel('kg m^2', fontsize=18)
        plt.plot(self.t,self.Izz_vector)
        plt.show()
        return None
    
    def plot_Ixx(self,Ixx_empty_motor = 0,mass_empty_motor = 0,cm_empty_motor = 0,distance_grain = 0, distance_nitrous = 0):
        i = 0
        length = len(self.t)
        self.Ixx_vector = []
        while i < length:
            time = i*self.step
            self.Ixx_vector.append(self.Ixx(time,Ixx_empty_motor,mass_empty_motor,cm_empty_motor,distance_grain, distance_nitrous))
            i = i + 1
        fig = plt.figure()
        fig.suptitle('Ixx (transversal moment of inertia)', fontsize=20)
        plt.xlabel('Seconds', fontsize=18)
        plt.ylabel('kg m^2', fontsize=18)
        plt.plot(self.t,self.Ixx_vector)
        plt.show()
        return None