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

Prior to running platypus, the user is encouraged to open the code and read, on the first lines, the assumptions considered for the model. 

The user can run the code by starting a Jupyter Notebbok and typing:

```
from platypusAlpha import *
```
And create a Motor class. Example:

```
marimbondo = Motor()
```

This will get the motor design. Consider changing the default inputs for different motors. These inputs are:

`ambient_temperature`   - Ambient temperature (K)

`ambient_pressure`      - Ambient Pressure (Bar)

`g`                     - Gravitational Acceleration (m/s^2)

`initial_thrust`        - Initial Thrust (N)

`total_impulse`         - Total Impulse (Ns)

`chamber_pressure`      - Nominal Chamber Pressure (Bar)

`chamber_temperature`   - Nominal Chamber Temperature (K)

`molar_mass`            - Molar mass of the products of combustion (kg/kmol)

`k`                     - Heat capacity ratio of the products of combustion (Adimensional)

`OFratio`               - Oxidiser/Fuel Mass ratio (Adimensional)

`a_gox`                 - Factor on fuel regression law

`n_gox`                 - Factor on fuel regression law

`combustion_eff`        - Combustion Efficiency (Adimensional, < 1)

`expansion_eff`         - Nozzle Expansion Efficiency (Adimensional, < 1)

`nozzle_conv_angle`     - Nozzle convergence half-angle (degrees)

`nozze_div_angle`       - Nozzle divergence half-angle (degrees)

`rho_paraffin`          - Paraffin's density (kg/m^3)

`d_grain`               - Internal diameter of the paraffin grain (m)

`d_chamber`             - Internal combustion chamber diamenter (m)

`d_tank`                - Internal oxidiser tank diameter (m)

`d_vent`                - Vent diameter (m)

`Cd`                    - Injector discharge coefficient

`Cd_Vent`               - Vent discharge coefficient

`oxidiser_mass_multiplier` - Multiplication factor on oxidiser mass, to account for decreasing thrust

`OFeffect`              - Boolean: if true, considers the inherent change in OFratio inside an hybrid rocket engine.

`time_vector`           - Time vector for output variables calculation

The default inputs can be verified inside the code.

The lengths of both the oxidiser tank and combustion chamber will be computed by taking into account some
of the inputs, in order to yield the axial and transversal moment of inertia of the motor as an output.
If desired, these lengths can be changed before running the simulation. Other dimensions are also computed.
Please refer to the explanations available in the code if needed.

For the proper simulation, type:

```
marimbondo.ignite()
```

After that, the code will already start carrying on the simulation.

## Outputs

The outputs can be retrieved in vector form or by plotting the desired variable. For instance:

```
thrust_output = marimbondo.thrust()
```

Will return the thrust values for each time instant, in the shape of a list. The time instants are
the ones provided by the input time_vector. If time_vector = 0 (as default), the time instants will 
be defined by the integration routine. 

On the other hand:

```
marimbondo.plot_thrust()
```

Will plot the thrust curve. For the other output variables, please refer to the code.

## Future versions

platypus will be gradually updated with new functionalities in order to provide not just a simulation that is as trustworthy as possible, but also a good user experience.

## Authors

Guilherme Castrignano Tavares

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/guilhermemd11/platypus/blob/master/LICENSE) file for details
