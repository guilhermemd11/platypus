# platypus

Hybrid rocket motor simulation with self-pressurizing oxidiser

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

Also, the user must choose some parameters related to the motor of interest by changing the values in the code itself.

After defining these parameters, the user can run the code by starting a Jupyter Notebbok and typing:

```
%matplotlib inline
import platypusAlpha
```

The code computes, for each instant of the burning, the following parameters of interest:

- Thrust
- Combustion chamber pressure
- Oxidiser tank pressure
- Oxidiser tank temperature
- Radius of the paraffin grain

By default the thrust is chosen to be plotted, but the user can define which variables will be plotted by changing the last lines of the code.

## Future versions

platypus will be gradually updated with new functionalities in order to provide not just a simulation that is as trustworthy as possible, but also a good user experience.

## Authors

Guilherme Castrignano Tavares

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/guilhermemd11/platypus/blob/master/LICENSE) file for details
