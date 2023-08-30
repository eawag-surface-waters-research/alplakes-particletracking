# Alplakes Particle Tracking

[![License: MIT][mit-by-shield]][mit-by] ![Python2][python2-by-shield] ![Python][python-by-shield]

The Alplakes project uses [ctracker](https://c4science.ch/tag/c-tracker/) a modified version of 
[TRACMASS](https://www.tracmass.org/) to run particle tracking (PT) simulations based on the velocity fields of the 3D 
hydrodynamic models. 

## Getting Started

### Clone repository

```commandline
git clone git@github.com:eawag-surface-waters-research/alplakes-particletracking.git
cd alplakes-particletracking
```

### Install python environments

Two python environments are required to run PT simulations using the Alplakes framework:

1) Python 3 - Preprocessing and postprocessing (see requirements.txt)

```commandline
conda env create -f environment.yml
```

2) Python 2.7 - Running ctracker (see ctracker/requirements.txt)

```commandline
conda env create -f ctracker/environment.yml
conda activate ctracker
cd ctracker
python setup.py build_ext --inplace
```

### Running particle tracking

Particle tracking can be run interactively using `notebooks/processing.ipynb` instructions are contained within the 
notebook.

Particle tracking can be run independently of the notebook see `src/tests.py` for an example of how to implement this.

[mit-by]: https://opensource.org/licenses/MIT
[mit-by-shield]: https://img.shields.io/badge/License-MIT-g.svg
[python-by-shield]: https://img.shields.io/badge/Python-3.9-g
[python2-by-shield]: https://img.shields.io/badge/Python-2.7-g
