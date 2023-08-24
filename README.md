# Alplakes Particle Tracking

The Alplakes project uses [ctracker](https://c4science.ch/tag/c-tracker/) a modified version of 
[TRACMASS](https://www.tracmass.org/) to run particle tracking (PT) simulations based on the velocity fields of the 3D 
hydrodynamic models. 

## Getting Started

### Install python environments

Two python environments are required to run PT simulations using the Alplakes framework:

1) Python 3 - Preprocessing and postprocessing (see requirements.txt)

`conda create --name particletracking --file requirements.txt python=3.9`

2) Python 2.7 - Running ctracker (see ctracker/requirements.txt)

`conda create --name ctracker --file ctracker/requirements.txt python=2.7`

It is suggested to use Anaconda to manage the environments.

### Running particle tracking

Particle tracking can be run interactively using `notebooks/processing.ipynb` instructions are contained within the 
notebook.

Particle tracking can be run independently of the notebook see `src/tests.py` for an example of how to implement this.
