# platypus

Hybrid rocket motor simulation with self-pressurizing oxidiser.

## Getting started

platypus is a Python code for simulating a hybrid rocket launch system. More precisely, it reffers to a nitrous oxide/paraffin hybrid, accounting for the self-pressurization properties of N20.

In order to run the simulation, the user should have installed in his/her computer some packages: Numpy, Scipy, Matplotlib and CoolProp.

If the user is already familiar with the Anaconda package, there is no need to install the first three packages, since they already come with the distribution. Otherwise, they should be installed through the  Windows Command Prompt as follows (provided that you already have Python installed):

```
$ pip install "numpy>=1.0"
$ pip install "scipy>=1.0"
$ pip install "matplotlib>=3.0"
$ pip install "netCDF4>=1.4"
```

For the CoolProp package, open the Windows Command Prompt and type:

```
$ pip install CoolProp
```

## Running

Prior to running platypus, the user is encouraged to open the code and read, on the first lines, the assumptions considered for the model. The code performs a simulation of a nitrous oxide - paraffin hybrid rocket motor. Some inputs from the user are needed, as seen below. For results, use Jupyter notebook, preferably, and import it as

```
from platypusAlpha import *
```

And create a Motor class. Example:

```
marimbondo = Motor(ambient_temperature = 300,
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
                   oxidiser_mass_multiplier = 1.2)
```

This will get the motor design. Consider changing the default inputs for different motors.

For the proper simulation, type:

```
marimbondo.ignite()
```

After that, the code will already start carrying on the simulation.

The code computes, for each instant of the burning, the following parameters of interest:

- Thrust
- Combustion chamber pressure
- Oxidiser tank pressure
- Oxidiser tank temperature
- Radius of the paraffin grain
- Motor's center of mass
- Motor's axial moment of inertia
- Motor's transversal moment of inertia

For plotting those variables, the user must type:

```
marimbondo.plot_VARIABLE^()
```

Be sure to substitute VARIABLE with the variable of interest, and to check each one of the plotting functions for their required inputs.


## Future versions

platypus will be gradually updated with new functionalities in order to provide not just a simulation that is as trustworthy as possible, but also a good user experience.

## Authors

Guilherme Castrignano Tavares

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/guilhermemd11/platypus/blob/master/LICENSE) file for details
