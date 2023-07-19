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

# Variables stored in one file or in seperate files?
# (If multivariable, must be of format 'UVEL','VVEL','WVEL','THETA')
multivariable_gcm_out = True

# mitgcm output filename root
gcm_out_root = "3Dsnaps"
# u-velocity filename root
u_root = "UVEL"
# v-velocity filename root
v_root = "VVEL"
# w-velocity filename root
w_root = "WVEL"

# output data type (">f8" for double precision, ">f4" for single)
out_prec = ">f8"

# grid geometry used in the MITgcm simulation
gcm_geometry = "cartesian"

# reference date of MITgcm simulation
gcm_start = "2008-03-01 00:00"

# MITgcm time step in seconds
gcm_dt = $gcm_dt

# MITgcm output time step in seconds
out_dt = $out_dt

# Simulation configuration

# file name for output (netcdf)
outfile = "$outfile"

# compression level in the output file,
# between 0 (larger file size, faster writing)
# and 9 (smaller file size, slower writing)
# Usually a small value, 2 or 3, is already
# enough to reduce substantially the file size
# without slowing down the code too much
complevel = 2

# HDF5 internals configuration
# useful for speeding up IO
# chunking size along particle id direction
chunk_id = 1 #200
# chunking size along time direction
chunk_time = 1 #20

# what output to produce:
# "gcmstep": output at the time step of the GCM,
#            this should be the obvious choice.
# "always": output at all iterations
#           (not to be used for large particle ensembles)
#           it uses a lot of memory, makes the code slower
# "cross": output at cell crossing
outfreq = "gcmstep"

# if outfreq == "gcmstep", one can decide to output
# only at multiple integers of the GCM step
# 1: every timestep, 2: every other, ...
out_gcmstep = 1

# start date of particle tracking simulation
start = "$start"

# end date of particle tracking simulation
end = "$end"

# list of seed points coordinates
import numpy as np
# we seed along a transect off the Rhone outflow
f = np.load("$particles")
for k, v in f.items():
        exec("{:s}=v.astype('float64')".format(k))

x_seed = f[f.files[0]]
y_seed = f[f.files[1]]
z_seed = f[f.files[2]]
# if inds_seed is True, x_seed, y_seed and z_seed are
# given in terms of (fractional) indices, otherwise they
# are interpreted as being in the same units as the MITgcm grid
inds_seed = False

# Note that seeding only takes place at the GCM output time
# If you pass the wrong time, no seeding will take place.
# seeding start time
seed_start = "$start"

# seeding end time
seed_end = "$end"

# seeding interval in seconds
seed_interval = 3600

# number of substeps between GCM output
subiters = 10

# forward (1) or backwards (-1) integration
ff = 1

# define corners of area where particles will be killed
# more than one is possible
# given in (fractional) grid indices of the GCM
iende = [0.0]
iendw = [0.0]
jendn = [0.0]
jends = [0.0]

# do we want to run a 2D simulation?
is2D = False

gcm_endian = '<'

n_min_part_per_thread = 30000

#
# Configure parallelism
# Note that the current implementation is such that
# parallelism makes sense only for large numbers of particles
#

# number of threads to use. If 1, run serially
n_procs = 1
