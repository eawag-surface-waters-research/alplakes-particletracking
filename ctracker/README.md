# C-tracker

A Lagrangian particle tracker for hydrodynamic numerical simulations written in python and cython.
The numerical part reproduces the results of tracmass code (https://github.com/TRACMASS/tracmass) by Döös et al., while introducing some minor changes in the numerical algorithm.
It is developed for use with MITgcm results, but any C- (or B-) grid model could be used as a source for the hydrodynamic fields.

*The code, while tested for the correctness of the results, should be considered experimental.*

For questions, comments, issues, please contact Andrea Cimatoribus at epfl
https://people.epfl.ch/andrea.cimatoribus

### Installation

To install:
   python setup.py build_ext --inplace

For using non-default compilers, use e.g:
   LDSHARED="icc -shared" CC=icc python setup.py build_ext --inplace

For intel C compiler, a good set of options is:
   LDSHARED="icc -shared" CC=icc CFLAGS="-O3 -xHOST -ipo -static -fp-model source" python setup.py build_ext --inplace

It seems that the results are very stable to further, more aggressive optimization.

packages-list.txt file lists all installed packages in my miniconda python installation.

### Note

A different version of the code is under more active development: https://c4science.ch/source/ctracker3/