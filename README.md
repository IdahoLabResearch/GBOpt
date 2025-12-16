# GBOpt
GBOpt is a Python package for creating, manipulating, and optimizing bicrystal grain boundary structures through configurable global optimization workflows. It uses a modular architecture, with separate modules for structure creation, manipulation, and optimization. In the initial release:
- GB creation is facilitated through a single parameterization of the macroscopic degrees of freedom (misorientation and inclination).
- GBs can be manipulated using grain translation, atom insertion, atom removal, and displacement along soft phonon modes.
- GB optimization is performed using either a Monte Carlo or an evolutionary algorithm optimizer, with energy evaluations currently using LAMMPS as the atomistic calculator.
The modular nature of the software is intended to allow easy extensibility to additional grain boundary parameterizations, structural manipulations, optimization engines, and external calculators.

To install, create a new conda environment
```
conda create --name GBOpt python
conda activate GBOpt
```

Make sure you are in the GBOpt directory:
```
cd ~/projects/GBOpt
```


Then install the dependencies
```
pip install .
```

