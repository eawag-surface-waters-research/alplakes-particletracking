###########################################################################
#                                                                         #
#    Copyright 2017 Andrea Cimatoribus                                    #
#    EPFL ENAC IIE ECOL                                                   #
#    GR A1 435 (Batiment GR)                                              #
#    Station 2                                                            #
#    CH-1015 Lausanne                                                     #
#    Andrea.Cimatoribus@epfl.ch                                           #
#                                                                         #
#    This file is part of ctracker                                        #
#                                                                         #
#    ctracker is free software: you can redistribute it and/or modify it  #
#    under the terms of the GNU General Public License as published by    #
#    the Free Software Foundation, either version 3 of the License, or    #
#    (at your option) any later version.                                  #
#                                                                         #
#    ctracker is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty          #
#    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.              #
#    See the GNU General Public License for more details.                 #
#                                                                         #
#    You should have received a copy of the GNU General Public License    #
#    along with ctracker.  If not, see <http://www.gnu.org/licenses/>.    #
#                                                                         #
###########################################################################

#
# MITgcm results description
#

# Directory containing the hydrodynamics fields
gcm_directory = "$gcm_directory"

# u-velocity filename root
u_root = "UVEL"
# v-velocity filename root
v_root = "VVEL"
# w-velocity filename root
w_root = "WVEL"

# output data type (">f8" for double precision, ">f4" for single)
out_prec = ">f4"

# grid geometry used in the MITgcm simulation
gcm_geometry = "curvilinear"

# reference date of MITgcm simulation
gcm_start = "2013-11-12 12:00"

# MITgcm time step in seconds
gcm_dt = 20.0

# MITgcm output time step in seconds
out_dt = 1800.0


#
# Simulation configuration
#

# file name for output (netcdf)
outfile = "results/tracks.nc"

# compression level in the output file,
# between 0 (larger file size, faster writing)
# and 9 (smaller file size, slower writing)
# Usually a small value, 2 or 3, is already
# enough to reduce substantially the file size
# without slowing down the code too much
complevel = 3

# HDF5 internals configuration
# useful for speeding up IO
# chunking size along particle id direction
chunk_id = 200
# chunking size along time direction
chunk_time = 20

# what output to produce:
# "gcmstep": output at the time step of the GCM,
#            this should be the obvious choice.
# "always": output at all iterations
#           (not to be used for large particle ensembles)
#           it uses a lot of memory, makes the code slower
# "cross": output at cell crossing
outfreq = "gcmstep"

# do we want to run a 2D simulation?
is2D = False

# if outfreq == "gcmstep", one can decide to output
# only at multiple integers of the GCM step
# 1: every timestep, 2: every other, ...
out_gcmstep = 2

# start date of particle tracking simulation
start = "2015-12-12 12:30"

# end date of particle tracking simulation
end = "2017-01-01 00:00"

# list of seed points coordinates
# we release a Gaussian off the Rhone
import numpy as np

n_points = 33
# center of distribution
p0 = (554500., 139300.) # km CH03
# radius (std of gaussian, in m)
s0 = 100.0

np.random.seed(1983)
xy = np.random.multivariate_normal(p0,
                                   [[s0*s0, 0.0], [0.0, s0*s0]],
                                   n_points)
xs = xy[:, 0]
ys = xy[:, 1]

zs = np.array([ 5.0, 10.0, 15.0,
               20.0, 25.0, 30.0,
               50.0, 55.0, 60.0])

X, Z = np.meshgrid(xs, zs)
Y, Z = np.meshgrid(ys, zs)
x_seed = X.ravel()
y_seed = Y.ravel()
z_seed = Z.ravel()

# if inds_seed is True, x_seed, y_seed and z_seed are
# given in terms of (fractional) indices, otherwise they
# are interpreted as being in the same units as the MITgcm grid
inds_seed = False

# Note that seeding only takes place at the GCM output time
# If you pass the wrong time, no seeding will take place.
# seeding start time
seed_start = "2015-12-12 12:30"

# seeding end time
seed_end = "2017-01-01 00:00"

# seeding interval in seconds
seed_interval = 14400

# number of substeps between GCM output
subiters = 1000

# forward (1) or backwards (-1) integration
ff = 1

# define corners of area where particles will be killed
# more than one is possible
# given in (fractional) grid indices of the GCM
iende = [10.0]
iendw = [0.0]
jendn = [100.0]
jends = [0.0]


#
# Configure parallelism
# Note that the current implementation is such that
# parallelism makes sense only for large numbers of particles
#

# number of threads to use. If 1, run serially
n_procs = 32
# minimum number of particles a thread is started with
n_min_part_per_thread = 500
