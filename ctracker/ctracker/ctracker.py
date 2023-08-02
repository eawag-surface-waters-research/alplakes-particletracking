#!/usr/bin/env python
from __future__ import division, print_function
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

import numpy as np
from modules.writetools import loc_time_fast


class ctracker(object):

    def __init__(self, config_file):
        from os.path import join
        from xmitgcm import open_mdsdataset as mitgcmds
        from modules import transform, transform_cartesian
        import os

        print("\nInitializing GCM information from %s" % config_file)
        if config_file.endswith(".py"):
            config_file = config_file[:-3]

        # load configuration
        exec("import %s as d" % config_file)

        print("\nConfiguration loaded.")

        #
        # initialize stuff
        #
        if d.gcm_geometry in ("curvilinear", "cartesian"):
            self.gcm_geometry = d.gcm_geometry
        else:
            raise ValueError("Unrecognised MITgcm geometry")

        self.multivariable_gcm_out = d.multivariable_gcm_out
        if self.multivariable_gcm_out:
            if not d.gcm_out_root.endswith('.'):
                d.gcm_out_root += '.'
            self.gcm_out_root = join(d.gcm_directory, d.gcm_out_root)
        else:
            if not d.u_root.endswith('.'):
                d.u_root += '.'
            if not d.v_root.endswith('.'):
                d.v_root += '.'
            if not d.w_root.endswith('.'):
                d.w_root += '.'
            self.u_root = join(d.gcm_directory, d.u_root)
            self.v_root = join(d.gcm_directory, d.v_root)
            self.w_root = join(d.gcm_directory, d.w_root)

        self.out_prec = d.out_prec

        # store some info
        self.gcm_dt = d.gcm_dt
        self.out_dt = d.out_dt
        self.gcm_endian = d.gcm_endian

        d.outfile = d.outfile.rstrip(".nc")
        self.inoutfile = d.outfile + "_inout.nc"
        self.runfile = d.outfile + "_run.nc"

        if d.outfreq in ("gcmstep", "always", "cross"):
            if d.outfreq == "gcmstep":
                self.outfreq = 1
                self.out_gcmstep = d.out_gcmstep
            elif d.outfreq == "always":
                self.outfreq = 2
                self.out_gcmstep = 0
            elif d.outfreq == "cross":
                self.outfreq = 3
                self.out_gcmstep = 0
        else:
            raise ValueError("Outfreq not recognised")

        self.subiters = int(d.subiters)
        self.dstep = 1.0 / self.subiters
        self.dtmax = self.out_dt * self.dstep
        if float(d.ff) not in [1.0, -1.0]:
            raise ValueError("ff controls the time direction, "
                             "it can only be +1 or -1")
        self.ff = float(d.ff)
        if self.ff == -1.0:
            print("\nUsing backward integration.")
        else:
            print("\nUsing forward integration.")

        self.gcm_start = np.datetime64(d.gcm_start)
        print("\nReference date of GCM:", self.gcm_start)
        self.start = np.datetime64(d.start)
        print("\nBegin particle tracking simulation at:", self.start)
        self.end = np.datetime64(d.end)
        print("\nEnd particle tracking simulation at:", self.end)
        self.seed_start = np.datetime64(d.seed_start)
        print("\nStart seeding particles at:", self.seed_start)
        self.seed_end = np.datetime64(d.seed_end)
        print("\nStop seeding particles at:", self.seed_end)

        # Check whether the directory contains the expected mitgcm outfiles
        if self.multivariable_gcm_out:
            file_list = os.listdir(d.gcm_directory)
            if not ([fn for fn in file_list if d.gcm_out_root in fn]):
                raise ValueError("Error loading mitgcm data. Check ctracker configuration file.")
        else:
            file_list = os.listdir(d.gcm_directory)
            if not ([fn for fn in file_list if d.u_root in fn]):
                raise ValueError("Error loading mitgcm data. Check ctracker configuration file.")

        self.is2D = d.is2D;
        print("\nSimulation in 2D : ", self.is2D)

        # open data directory to load grid data
        if self.multivariable_gcm_out:
            self.grid = mitgcmds(d.gcm_directory, read_grid=True,
                                 iters=[], prefix=[d.gcm_out_root],
                                 swap_dims=False, geometry=self.gcm_geometry,
                                 ref_date=self.gcm_start, delta_t=self.gcm_dt, endian=self.gcm_endian)
        else:
            self.grid = mitgcmds(d.gcm_directory, read_grid=True,
                                 iters=[], prefix=[d.u_root, d.v_root, d.w_root],
                                 swap_dims=False, geometry=self.gcm_geometry,
                                 ref_date=self.gcm_start, delta_t=self.gcm_dt)

        # load metrics to compute the conversion from fractional index
        # to physical coordinate
        self.xG = self.grid.XG.to_masked_array().filled(0).astype("float32")
        self.yG = self.grid.YG.to_masked_array().filled(0).astype("float32")
        self.dX = self.grid.dxG.to_masked_array().filled(0).astype("float32")
        self.dY = self.grid.dyG.to_masked_array().filled(0).astype("float32")
        self.dxdy = self.grid.rA.to_masked_array().filled(0).astype("float32")
        # we invert to have at the first position the bottom
        self.dzt = np.ascontiguousarray((self.grid.drF * self.grid.hFacC)
                                        .to_masked_array()
                                        .filled(0)[::-1, :, :])

        self.grid_shape = self.dzt.shape
        if self.multivariable_gcm_out:
            self.gcm_data_shape = (3,) + self.grid_shape  # changed and then unchanged by Abolfazl Irani Rahaghi

        self.kmax, self.jmax, self.imax = self.grid_shape

        zG = np.zeros((self.kmax + 1, self.jmax, self.imax))
        zG[1:, ...] = np.cumsum(self.dzt[::-1], axis=0)
        # tracmass has opposite Z order
        zG = zG[::-1, ...]
        # print('Abolfazl test2',np.sum((self.grid.hFacC).to_masked_array().filled(0).astype("float32")==0))
        self.Z = np.ascontiguousarray(zG).astype("float32")
        self.dxyz = self.dxdy * self.dzt
        if np.any(self.dxyz < 0.0):
            raise ValueError("Cells with negative volume.")
        if np.any((self.dxyz == 0.0) & (self.grid.hFacC[::-1, :, :] != 0.0)):
            raise ValueError("Zero cell volumes.")
        print(np.any(self.dxyz[-2, :, :] == 0.0))
        self.dsmax = self.dtmax / self.dxyz
        # print(self.dsmax[:,200,200])
        self.dzu = np.ascontiguousarray((self.grid.drF *
                                         self.grid.hFacW * self.grid.dyG)
                                        .to_masked_array()
                                        .filled(0)[::-1, :, :])
        self.dzv = np.ascontiguousarray((self.grid.drF *
                                         self.grid.hFacS * self.grid.dxG)
                                        .to_masked_array()
                                        .filled(0)[::-1, :, :])
        self.kmtb = (self.grid.hFacC.sum(dim="k")).to_masked_array().filled(0)
        self.kmt = np.ceil(self.kmtb).astype("int8")

        self.CS = 1
        self.SN = 0
        if self.gcm_geometry in ("curvilinear",):
            self.CS = self.grid.CS \
                .to_masked_array().filled(0).astype("float32")
            self.SN = self.grid.SN \
                .to_masked_array().filled(0).astype("float32")

        self.xGp1, self.yGp1 = self._get_geometry()

        self.grid.attrs["MITgcm_dir"] = d.gcm_directory
        # we also store the full mesh with edges coordinates
        self.grid.coords["i_p1"] = ("i_p1", np.arange(self.xGp1.shape[1]))
        self.grid.coords["j_p1"] = ("j_p1", np.arange(self.xGp1.shape[0]))
        self.grid.coords["XG_p1"] = (("j_p1", "i_p1"), self.xGp1)
        self.grid.coords["YG_p1"] = (("j_p1", "i_p1"), self.yGp1)

        #
        # prepare the simulation
        #
        print("\nIdentify seeding points")

        if d.inds_seed:
            if np.any(d.x_seed > self.imax) or \
                    np.any(d.x_seed < 0) or \
                    np.any(d.y_seed > self.jmax) or \
                    np.any(d.y_seed < 0) or \
                    np.any(d.z_seed > self.kmax) or \
                    np.any(d.z_seed < 0):
                raise ValueError("Seeding with indexes beyond grid limits")
        self.ijk_indseed, self.ijk_seed = \
            self._ijkseed(d.x_seed, d.y_seed, d.z_seed, d.inds_seed)
        self.xyz_seed = np.zeros(self.ijk_seed.shape)
        if self.gcm_geometry in ("curvilinear",):
            transform(self.ijk_seed[:, 0],
                      self.ijk_seed[:, 1],
                      self.ijk_seed[:, 2],
                      self.xG, self.yG, self.dX, self.dY,
                      self.CS, self.SN, self.Z,
                      self.xyz_seed)
        else:
            transform_cartesian(self.ijk_seed[:, 0],
                                self.ijk_seed[:, 1],
                                self.ijk_seed[:, 2],
                                self.xG, self.yG, self.dX, self.dY,
                                self.Z, self.xyz_seed)
        self.n_seed = self.ijk_seed.shape[0]

        print("\nNumber of seeding points: %d" % self.n_seed)

        d.iende = np.asarray(d.iende)
        d.iendw = np.asarray(d.iendw)
        d.jendn = np.asarray(d.jendn)
        d.jends = np.asarray(d.jends)
        self.nend = d.iende.size
        if (self.nend != d.iendw.size) or \
                (self.nend != d.jendn.size) or \
                (self.nend != d.jends.size):
            raise ValueError("Wrong kill area definition")
        self.iende = d.iende
        self.jendn = d.jendn
        self.iendw = d.iendw
        self.jends = d.jends

        if (d.out_dt % self.gcm_dt) != 0:
            raise ValueError("The GCM output interval must be "
                             "a multiple of the model time step.")

        # first GCM iteration
        iter_i = np.floor(
            (self.start - self.gcm_start)
            .astype("timedelta64[s]").astype(float) /
            self.gcm_dt).astype("int32")
        # last GCM iteration
        iter_e = np.ceil(
            (self.end - self.gcm_start)
            .astype("timedelta64[s]").astype(float) /
            self.gcm_dt).astype("int32")

        # forward integration
        self.iters = np.arange(iter_i, iter_e + 1, int(self.out_dt / self.gcm_dt))
        # backward integration
        if self.ff == -1.0:
            self.iters = self.iters[::-1]
        if self.iters.size < 2:
            raise ValueError("To start the computation, the simulation "
                             "must span in time at least two GCM outputs")

        if (d.seed_interval % self.out_dt) != 0:
            raise ValueError("The seeding interval must be "
                             "a multiple of the GCM step.")

        self.seed_interval = d.seed_interval
        print("\nSeeding interval: %ds" % self.seed_interval)

        # seeding steps
        seed_i = np.floor(
            (self.seed_start - self.gcm_start)
            .astype("timedelta64[s]").astype(float) /
            self.gcm_dt).astype("int32")
        seed_e = np.ceil(
            (self.seed_end - self.gcm_start)
            .astype("timedelta64[s]").astype(float) /
            self.gcm_dt).astype("int32")
        iter_seed = np.arange(seed_i, seed_e + 1,
                              int(self.seed_interval / self.gcm_dt))

        self.iter_seed = np.intersect1d(self.iters, iter_seed)

        print("\nNumber of seeding time steps: %d" % self.iter_seed.size)

        self.ntrackmax = self.iter_seed.size * self.n_seed
        print("\nTotal number of particles to be released: %d" %
              self.ntrackmax)

        # configure parallelism
        if (d.n_procs % 1.0) != 0.0:
            raise ValueError("The number of processes must be an integer.")
        self.n_procs = int(d.n_procs)
        if self.n_procs <= 1:
            self.n_procs = 1
            print("\nRunning numerics single threaded")
        else:
            print("\nRunning numerics on %d threads max" % self.n_procs)
        self.n_min_part_per_thread = d.n_min_part_per_thread
        print("\nRunning with %d particles per thread min" % self.n_min_part_per_thread)

        print("\nInput running on a separate thread.")
        print("\nOutput running on a separate thread.")

        # save to netcdf, this will be the output file of the simulation
        # we require NETCDF4
        self.grid.to_netcdf(self.runfile,
                            mode="w", format="NETCDF4", engine="netcdf4")
        self.complevel = int(max(0, min(9, d.complevel)))
        self.chunk_id = min(int(d.chunk_id), self.ntrackmax)
        self.chunk_time = int(d.chunk_time)

    # end init

    def _ijkseed(self, ii, jj, kk, inds):
        """
        Define the ii, jj, kk indices from which particles will be
        released.
        ii, jj, kk: list, array, or scalar with indices (i, j, k) from which
                    particles will be released (must be integers).
        inds:       if True (default), ii, jj and kk are interpreted
                    as indices, otherwise they are interpreted as lists of
                    exact release positions
        """

        from itertools import product
        from matplotlib.path import Path

        ii = np.atleast_1d(np.squeeze(ii))
        jj = np.atleast_1d(np.squeeze(jj))
        kk = np.atleast_1d(np.squeeze(kk))

        if (ii.size != jj.size) or (ii.size != kk.size):
            raise ValueError("ii, jj and kk must have all the same dimension.")

        if inds:
            return np.atleast_2d(np.asarray(zip(
                np.int16(ii),
                np.int16(jj),
                np.int16(kk)))), \
                np.atleast_2d(np.asarray(zip(ii, jj, kk)))
        else:
            # if MITgcm coordinates are passed, we have to load the model grid
            # and then translate into the "normalised" index coordinates of
            # tracmass
            def xy2grid(x, y, dX, dY, C, S):
                nx = (C * x + S * y) / dX
                ny = (-S * x + C * y) / dY
                return nx, ny

            xG = self.xGp1
            yG = self.yGp1

            dX = self.dX
            dY = self.dY
            if self.gcm_geometry in ("curvilinear",):
                cs = self.CS
                sn = self.SN
            dZ = self.dzt
            zG = self.Z

            iout = np.zeros(ii.size) * np.nan
            jout = np.zeros(ii.size) * np.nan
            kout = np.zeros(ii.size) * np.nan
            iindout = np.zeros(ii.size, dtype="int16")
            jindout = np.zeros(ii.size, dtype="int16")
            kindout = np.zeros(ii.size, dtype="int16")
            for nn, (xx, yy, zz) in enumerate(zip(ii, jj, kk)):
                jseed, iseed = np.unravel_index(
                    np.argmin((xx - xG) ** 2 + (yy - yG) ** 2), xG.shape)
                if iseed >= self.imax:
                    iseed -= 1
                if jseed >= self.jmax:
                    jseed -= 1
                p = Path([[xG[jseed, iseed], yG[jseed, iseed]],
                          [xG[jseed + 1, iseed], yG[jseed + 1, iseed]],
                          [xG[jseed + 1, iseed + 1], yG[jseed + 1, iseed + 1]],
                          [xG[jseed, iseed + 1], yG[jseed, iseed + 1]]])

                # is the point inside this polygon?
                if not p.contains_point((xx, yy)):
                    for ijtry in product(range(-1, 2), range(-1, 2)):
                        inew = iseed + ijtry[0]
                        jnew = jseed + ijtry[1]
                        if (inew >= self.imax) or (jnew >= self.jmax) or \
                                (inew < 0) or (jnew < 0):
                            continue
                        p = Path([[xG[jnew, inew], yG[jnew, inew]],
                                  [xG[jnew + 1, inew], yG[jnew + 1, inew]],
                                  [xG[jnew + 1, inew + 1], yG[jnew + 1, inew + 1]],
                                  [xG[jnew, inew + 1], yG[jnew, inew + 1]]])

                        if not p.contains_point((xx, yy)):
                            continue
                        else:
                            iseed = inew
                            jseed = jnew
                            break
                    else:
                        raise ValueError("Could not find the point "
                                         "(x=%f, y=%f)" % (xx, yy))

                if self.gcm_geometry in ("curvilinear",):
                    nx, ny = xy2grid(xx - xG[jseed, iseed],
                                     yy - yG[jseed, iseed],
                                     dX[jseed, iseed], dY[jseed, iseed],
                                     cs[jseed, iseed], sn[jseed, iseed])
                else:
                    nx = (xx - xG[jseed, iseed]) / dX[jseed, iseed]
                    ny = (yy - yG[jseed, iseed]) / dY[jseed, iseed]
                # in case there is some edge issue
                if nx < 0 or nx >= 1 or ny < 0 or ny >= 1:
                    print("\nInvalid point at "
                          "x,y,z=%.2f,%.2f,%.2f" % (xx, yy, zz))
                    print("i,j=%d,%d" % (iseed, jseed))
                    print("nx,ny=%.3e,%.3e" % (nx, ny))
                    continue

                z_here = zG[:, jseed, iseed]
                if (zz >= z_here.max()) or (zz <= z_here.min()):
                    print("\nPoint outside vertical bounds at "
                          "x,y,z=%.2f,%.2f,%.2f" % (xx, yy, zz))
                    continue
                kk = np.where(z_here > zz)[0][-1]
                nz = (zz - z_here[kk]) / (z_here[kk + 1] - z_here[kk])
                if self.dxyz[kk, jseed, iseed] == 0:
                    print("\nPoint on land "
                          "x,y,z=%.2f,%.2f,%.2f" % (xx, yy, zz))
                    continue
                iindout[nn] = iseed
                jindout[nn] = jseed
                kindout[nn] = kk
                iout[nn] = iseed + nx
                jout[nn] = jseed + ny
                kout[nn] = kk + nz
            return np.atleast_2d(np.asarray(zip(
                iindout[np.isfinite(iout)],
                jindout[np.isfinite(jout)],
                kindout[np.isfinite(kout)]))), \
                np.atleast_2d(np.asarray(zip(
                    iout[np.isfinite(iout)],
                    jout[np.isfinite(jout)],
                    kout[np.isfinite(kout)])))

    # end _ijkseed

    def _get_geometry(self):
        """
        Convenience function returning XG and YG of mitgcm
        """

        xG = np.zeros((self.jmax + 1, self.imax + 1))
        yG = np.zeros((self.jmax + 1, self.imax + 1))
        xG[:-1, :-1] = self.xG
        yG[:-1, :-1] = self.yG
        dxG = self.dX
        dyG = self.dY

        if self.gcm_geometry == "curvilinear":
            cs = self.CS
            sn = self.SN

            # Fill (approximate) end points of the grid
            xG[:-1, -1] = xG[:-1, -2] + dxG[:, -1] * cs[:, -1]
            xG[-1, :-1] = xG[-2, :-1] - dyG[-1, :] * sn[-1, :]
            # we lack the last metric at the NE corner, so we use the
            # nearby metric
            xG[-1, -1] = xG[-1, -2] + dxG[-1, -1] * cs[-1, -1]

            yG[-1, :-1] = yG[-2, :-1] + dyG[-1, :] * cs[-1, :]
            yG[:-1, -1] = yG[:-1, -2] + dxG[:, -1] * sn[:, -1]
            yG[-1, -1] = yG[-2, -1] + dyG[-1, -1] * cs[-1, -1]

        elif self.gcm_geometry == "cartesian":
            # Fill (approximate) end points of the grid
            xG[:-1, -1] = xG[:-1, -2] + dxG[:, -1]
            xG[-1, :] = xG[-2, :]

            yG[-1, :-1] = yG[-2, :-1] + dyG[-1, :]
            yG[:, -1] = yG[:, -2]

        else:
            raise ValueError("Grid geometry not recognised.")
        return xG, yG

    # end _get_geometry

    def _nc_init(self, nc_f, kind):
        if kind == "inout":
            # store info on kill areas
            nc_f.setncattr("i_endw", self.iendw)
            nc_f.setncattr("i_ende", self.iende)
            nc_f.setncattr("j_endn", self.jendn)
            nc_f.setncattr("j_ends", self.jends)
            # id
            nc_f.createDimension("pid", size=self.ntrackmax)
            nc_f.createVariable("pid", "int32", ("pid",))
            ncvar = nc_f.variables["pid"]
            ncvar[:] = np.arange(1, self.ntrackmax + 1)
            ncvar.setncattr("long_name", "particle ID")
            # exit code
            nc_f.createVariable("outc", "i2", ("pid",))
            ncvar = nc_f.variables["outc"]
            ncvar.setncattr("code -1", "not released")
            ncvar.setncattr("code 0", "still active")
            ncvar.setncattr("code 1", "out of the horizontal domain")
            ncvar.setncattr("code 2", "particle reached the surface")
            ncvar.setncattr("code 10+N", "reached N-th kill region")
            ncvar[:] = -1
            # time of seeding
            nc_f.createVariable("t_ini", "float64", ("pid",), fill_value=-1.0)
            ncvar = nc_f.variables["t_ini"]
            ncvar.setncattr("units", ("seconds since %s" %
                                      self.gcm_start.astype("datetime64[s]"))
                            .replace("T", " "))
            # end time
            nc_f.createVariable("t_end", "float64", ("pid",), fill_value=-1.0)
            ncvar = nc_f.variables["t_end"]
            ncvar.setncattr("units", ("seconds since %s" %
                                      self.gcm_start.astype("datetime64[s]"))
                            .replace("T", " "))
            # i-position at seeding
            nc_f.createVariable("i_ini", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["i_ini"]
            ncvar.setncattr("units", "fractional cell_index")
            # j-position at seeding
            nc_f.createVariable("j_ini", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["j_ini"]
            ncvar.setncattr("units", "fractional cell_index")
            # k-position at seeding
            nc_f.createVariable("k_ini", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["k_ini"]
            ncvar.setncattr("units", "fractional cell_index")
            # x-position at seeding
            nc_f.createVariable("x_ini", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["x_ini"]
            ncvar.setncattr("units", "GCM grid units")
            # y-position at seeding
            nc_f.createVariable("y_ini", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["y_ini"]
            ncvar.setncattr("units", "GCM grid units")
            # z-position at seeding
            nc_f.createVariable("z_ini", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["z_ini"]
            ncvar.setncattr("units", "GCM grid units")
            # i-position at exit
            nc_f.createVariable("i_end", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["i_end"]
            ncvar.setncattr("units", "fractional cell_index")
            # j-position at exit
            nc_f.createVariable("j_end", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["j_end"]
            ncvar.setncattr("units", "fractional cell_index")
            # k-position at exit
            nc_f.createVariable("k_end", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["k_end"]
            ncvar.setncattr("units", "fractional cell_index")
            # x-position at exit
            nc_f.createVariable("x_end", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["x_end"]
            ncvar.setncattr("units", "GCM grid units")
            # y-position at exit
            nc_f.createVariable("y_end", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["y_end"]
            ncvar.setncattr("units", "GCM grid units")
            # z-position at exit
            nc_f.createVariable("z_end", "float32", ("pid",),
                                zlib=True, complevel=self.complevel,
                                shuffle=True)
            ncvar = nc_f.variables["z_end"]
            ncvar.setncattr("units", "GCM grid units")
        elif kind == "run":
            # id
            nc_f.createDimension("pid", size=self.ntrackmax)
            nc_f.createVariable("pid", "int32", ("pid",))
            ncvar = nc_f.variables["pid"]
            ncvar[:] = np.arange(1, self.ntrackmax + 1)
            ncvar.setncattr("long_name", "particle ID")
            # time
            nc_f.createDimension("time", size=None)
            nc_f.createVariable("time", "float64", ("time",))
            ncvar = nc_f.variables["time"]
            ncvar.setncattr("units", ("seconds since %s" %
                                      self.gcm_start.astype("datetime64[s]"))
                            .replace("T", " "))
            # write times are highly dependent on the chunking in use
            chunks = (self.chunk_id, self.chunk_time)
            # itrack
            nc_f.createVariable("itrack", "float32", ("pid", "time"),
                                zlib=True, complevel=self.complevel,
                                shuffle=True, chunksizes=chunks)
            ncvar = nc_f.variables["itrack"]
            ncvar.setncattr("units", "fractional cell_index")
            # jtrack
            nc_f.createVariable("jtrack", "float32", ("pid", "time"),
                                zlib=True, complevel=self.complevel,
                                shuffle=True, chunksizes=chunks)
            ncvar = nc_f.variables["jtrack"]
            ncvar.setncattr("units", "fractional cell_index")
            # ktrack
            nc_f.createVariable("ktrack", "float32", ("pid", "time"),
                                zlib=True, complevel=self.complevel,
                                shuffle=True, chunksizes=chunks)
            ncvar = nc_f.variables["ktrack"]
            ncvar.setncattr("units", "fractional cell_index")
            # xtrack
            nc_f.createVariable("xtrack", "float32", ("pid", "time"),
                                zlib=True, complevel=self.complevel,
                                shuffle=True, chunksizes=chunks)
            ncvar = nc_f.variables["xtrack"]
            ncvar.setncattr("units", "GCM grid units")
            # ytrack
            nc_f.createVariable("ytrack", "float32", ("pid", "time"),
                                zlib=True, complevel=self.complevel,
                                shuffle=True, chunksizes=chunks)
            ncvar = nc_f.variables["ytrack"]
            ncvar.setncattr("units", "GCM grid units")
            # ztrack
            nc_f.createVariable("ztrack", "float32", ("pid", "time"),
                                zlib=True, complevel=self.complevel,
                                shuffle=True, chunksizes=chunks)
            ncvar = nc_f.variables["ztrack"]
            ncvar.setncattr("units", "GCM grid units")
        else:
            raise ValueError("Unrecognised case")

        nc_f.sync()

    # end _nc_init

    def _nc_write_one_time(self, nc_f, time, ids, ijk, xyz,
                           outc=None, init=False):
        # The particle ids start from 1, python index from 0
        ids -= 1
        if init:
            nc_f.variables["t_ini"][ids] = time
            nc_f.variables["outc"][ids] = 0
            nc_f.variables["i_ini"][ids] = ijk[:, 0]
            nc_f.variables["j_ini"][ids] = ijk[:, 1]
            nc_f.variables["k_ini"][ids] = ijk[:, 2]
            nc_f.variables["x_ini"][ids] = xyz[:, 0]
            nc_f.variables["y_ini"][ids] = xyz[:, 1]
            nc_f.variables["z_ini"][ids] = xyz[:, 2]
        elif np.any(outc > 0):  # particles exit
            nonzerooc = outc > 0
            nc_f.variables["t_end"][ids[nonzerooc]] = time
            nc_f.variables["outc"][ids[nonzerooc]] = outc[nonzerooc]
            nc_f.variables["i_end"][ids] = ijk[:, 0]
            nc_f.variables["j_end"][ids] = ijk[:, 1]
            nc_f.variables["k_end"][ids] = ijk[:, 2]
            nc_f.variables["x_end"][ids] = xyz[:, 0]
            nc_f.variables["y_end"][ids] = xyz[:, 1]
            nc_f.variables["z_end"][ids] = xyz[:, 2]
        else:
            tind = self._loc_time(nc_f, time)
            nc_f.variables["itrack"][ids, tind] = ijk[:, 0]
            nc_f.variables["jtrack"][ids, tind] = ijk[:, 1]
            nc_f.variables["ktrack"][ids, tind] = ijk[:, 2]
            nc_f.variables["xtrack"][ids, tind] = xyz[:, 0]
            nc_f.variables["ytrack"][ids, tind] = xyz[:, 1]
            nc_f.variables["ztrack"][ids, tind] = xyz[:, 2]

    def _loc_time(self, nc_f, time):
        ind = loc_time_fast(self.nc_time, self.nc_time.size,
                            time, False)
        if ind >= 0:
            return ind
        else:
            # We did not find the time we are looking for,
            # so we add it to the time axis
            # both in memory and on disk
            self.nc_time = np.append(self.nc_time, time)
            ind = self.nc_time.size - 1
            nc_f.variables["time"][ind] = time
            return ind

    def run(self):
        """
        Main function
        """

        from netCDF4 import Dataset
        from modules import loop_nogil, loopC_nogil

        from threading import Thread
        from Queue import Queue

        print("\nStarting ctracker")

        self.max_id = 0
        # Total number of active particles
        self._ntact = 0
        # Number of particles which exited the domain
        self._ntout = 0

        # array with particle ids
        active_ids = np.array([], dtype="int64")
        # input/output position of particles
        xyz = np.zeros((0, 3), dtype="float64")
        ijk = np.zeros((0, 3), dtype="int16")

        # this is a copy in memory of the NETcdf time
        # it is useful to find the index where to write
        # data without reading from disk
        self.nc_time = np.array([], dtype="float64")

        # determine size of output buffer
        if self.outfreq == 1:
            nvals = 1
        elif self.outfreq == 2:
            nvals = 10 * self.subiters
            will_write = 2
        elif self.outfreq == 3:
            nvals = 15  # this should be a safe maximum
            will_write = 3

        # buffer to store the codes telling if/how the particle
        # exited the computation
        out_code = np.zeros(self._ntact, dtype="short")

        out_tijk = np.zeros((self._ntact, nvals, 4),
                            dtype="f8")
        out_xyz = np.zeros((self._ntact, nvals, 3),
                           dtype="f8")

        # the queue for the input data
        # we do not want to load too many files,
        # otherwise we would fill in the memory very quickly
        INq = Queue(maxsize=3)

        # define the worker loading the input
        def INworker():
            for ngi in range(1, self.iters.size):
                it_old = self.iters[ngi - 1]
                it_new = self.iters[ngi]

                # define flux arrays
                uflux = np.zeros((2, self.kmax, self.jmax, self.imax + 1),
                                 "float64")

                vflux = np.zeros((2, self.kmax, self.jmax + 1, self.imax),
                                 "float64")
                wflux = np.zeros((2, self.kmax + 1, self.jmax, self.imax),
                                 "float64")

                # read old GCM step for interpolating
                fstamp_old = "%010d.data" % it_old
                # read new GCM step for interpolating
                fstamp_new = "%010d.data" % it_new
                ################
                ################
                ################
                if self.multivariable_gcm_out:
                    # old GCM step
                    # print('test')
                    gcm_velocity_old = np.fromfile(self.gcm_out_root + fstamp_old,
                                                   dtype=self.out_prec) \
                                           .reshape(self.gcm_data_shape)[:, ::-1,
                                       ...]  # chnaged by Abolfazl Irani Rahaghi
                    # print(gcm_velocity_old[0,:,200,200])
                    # print(gcm_velocity_old[0,...].shape)
                    # print(self.dzu[:,200,200])
                    # print(self.ff.shape)
                    uflux[0, :, :, :-1] = gcm_velocity_old[0, ...] * self.dzu * self.ff
                    vflux[0, :, :-1, :] = gcm_velocity_old[1, ...] * self.dzv * self.ff
                    # print(gcm_velocity_old[0,:,10,70])
                    # print('veloity',gcm_velocity_old[0,...].shape)
                    # print('dzv',self.dzv.shape)

                    if not self.is2D:
                        wflux[0, 1:, :, :] = gcm_velocity_old[2, ...] * self.dxdy * self.ff

                    # new GCM step
                    gcm_velocity_new = np.fromfile(self.gcm_out_root + fstamp_new,
                                                   dtype=self.out_prec) \
                                           .reshape(self.gcm_data_shape)[:, ::-1,
                                       ...]  # chnaged by Abolfazl Irani Rahaghi
                    uflux[1, :, :, :-1] = gcm_velocity_new[0, ...] * self.dzu * self.ff
                    vflux[1, :, :-1, :] = gcm_velocity_new[1, ...] * self.dzv * self.ff
                    if not self.is2D:
                        wflux[1, 1:, :, :] = gcm_velocity_old[2, ...] * self.dxdy * self.ff

                else:
                    # old GCM step
                    uflux[0, :, :, :-1] = np.fromfile(self.u_root + fstamp_old,
                                                      dtype=self.out_prec) \
                                              .reshape(self.grid_shape)[::-1, ...] * self.dzu * self.ff

                    vflux[0, :, :-1, :] = np.fromfile(self.v_root + fstamp_old,
                                                      dtype=self.out_prec) \
                                              .reshape(self.grid_shape)[::-1, ...] * self.dzv * self.ff

                    if not self.is2D:
                        wflux[0, 1:, :, :] = np.fromfile(self.w_root + fstamp_old,
                                                         dtype=self.out_prec) \
                                                 .reshape(self.grid_shape)[::-1, ...] * self.dxdy * self.ff

                    # new GCM step
                    uflux[1, :, :, :-1] = np.fromfile(self.u_root + fstamp_new,
                                                      dtype=self.out_prec) \
                                              .reshape(self.grid_shape)[::-1, ...] * self.dzu * self.ff

                    vflux[1, :, :-1, :] = np.fromfile(self.v_root + fstamp_new,
                                                      dtype=self.out_prec) \
                                              .reshape(self.grid_shape)[::-1, ...] * self.dzv * self.ff

                    if not self.is2D:
                        wflux[1, 1:, :, :] = np.fromfile(self.w_root + fstamp_new,
                                                         dtype=self.out_prec) \
                                                 .reshape(self.grid_shape)[::-1, ...] * self.dxdy * self.ff
                ################
                ################
                ################

                INq.put((ngi, uflux.copy(), vflux.copy(), wflux.copy()))

        # start the input thread
        INt = Thread(target=INworker)
        INt.daemon = True
        INt.start()

        # the queue where we put the output data
        # which will be processed by a separate thread
        # we set a maximum size to avoid using too much memory
        OUTq = Queue(maxsize=50)

        # define the worker which does the output work
        def OUTworker():
            ncall = 0
            while True:
                if OUTq.full():
                    print("Warning: output queue is full "
                          "(slows down the execution).")
                ncall += 1
                t0, out_tijk, out_xyz, active_ids, out_code, init = OUTq.get()
                if init:
                    self._nc_write_one_time(nc_inout, t0, active_ids,
                                            self.ijk_seed, self.xyz_seed,
                                            init=True)
                    OUTq.task_done()
                else:
                    # write to netCDF4 file
                    # identify writing times
                    # (usually, only 1 or a few different)
                    towrite = out_tijk[:, :, 0] >= 0
                    wtimes = np.unique(out_tijk[towrite, 0])
                    for wtime in wtimes:
                        # if outfreq is 1, we know there is only one record
                        # to be written
                        if self.outfreq == 1:
                            # particles running
                            positions = (out_tijk[:, 0, 0] == wtime) & \
                                        (out_code == 0)
                            if positions.sum() > 0:
                                self._nc_write_one_time(
                                    nc_run,
                                    t0 + wtime,
                                    active_ids[positions],
                                    out_tijk[positions, 0, 1:],
                                    out_xyz[positions, 0, :],
                                    outc=out_code[positions])
                            # particles exiting
                            positions = (out_tijk[:, 0, 0] == wtime) & \
                                        (out_code > 0)
                            if positions.sum() > 0:
                                self._nc_write_one_time(
                                    nc_inout,
                                    t0 + wtime,
                                    active_ids[positions],
                                    out_tijk[positions, 0, 1:],
                                    out_xyz[positions, 0, :],
                                    outc=out_code[positions])
                        else:
                            # TODO
                            raise ValueError("Unimplemented")
                    OUTq.task_done()

                # once in a while, make sure we sync to disk
                # this is important for reading the output
                # while a simulation is still running
                if (ncall % 100) == 0:
                    nc_inout.sync()
                    nc_run.sync()
                    ncall = 0

        # end def OUTworker

        # start the output thread
        OUTt = Thread(target=OUTworker)
        OUTt.daemon = True
        OUTt.start()

        # open file for output
        with Dataset(self.inoutfile, mode="w") as nc_inout, \
                Dataset(self.runfile, mode="r+") as nc_run:

            self._nc_init(nc_inout, "inout")
            self._nc_init(nc_run, "run")

            #
            # Start time loop over GCM output steps
            #
            print("\nStart main loop\n")
            for ngi in range(1, self.iters.size):
                # we are starting from the start!
                # even if we will read the following
                # gcm output for interpolating
                it_old = self.iters[ngi - 1]
                t0 = it_old * self.gcm_dt

                # get data from the queue
                ngiQ, uflux, vflux, wflux = INq.get()
                if ngiQ != ngi:
                    raise ValueError("Queue provides wrong timestep")

                if self.outfreq == 1:
                    # will we actually do output at this iteration?
                    # note that if a particle exits the domain,
                    # it is written in any case
                    if (ngi % self.out_gcmstep) == 0:
                        will_write = 1  # write
                    else:
                        will_write = 0  # don't write

                # remove particles that went out of the domain or other
                del_index = out_code == 0
                n_remove = (~del_index).sum()
                if n_remove:
                    active_ids = active_ids[del_index]
                    xyz = xyz[del_index, ...]
                    ijk = ijk[del_index, ...]
                    self._ntout += n_remove
                    self._ntact = active_ids.size
                    # the active particles number changed, so
                    # we update the buffer size
                    out_code = np.zeros(self._ntact, dtype="short")

                    out_tijk = np.zeros((self._ntact, nvals, 4),
                                        dtype="f8")
                    out_xyz = np.zeros((self._ntact, nvals, 3),
                                       dtype="f8")

                if it_old in self.iter_seed:
                    new_ids = np.arange(self.max_id + 1,
                                        self.max_id + 1 + self.n_seed,
                                        dtype="int64")
                    active_ids = np.append(active_ids, new_ids)
                    xyz = np.vstack([xyz,
                                     self.ijk_seed])
                    ijk = np.vstack([ijk,
                                     np.int16(self.ijk_indseed)])
                    self.max_id = self.max_id + self.n_seed
                    # we store the seeding position
                    OUTq.put((t0,
                              None,
                              None,
                              new_ids.copy(),
                              None,
                              True))
                    self._ntact = active_ids.size
                    # the active particles number changed, so
                    # we update the buffer size
                    out_code = np.zeros(self._ntact, dtype="short")

                    out_tijk = np.zeros((self._ntact, nvals, 4),
                                        dtype="f8")
                    out_xyz = np.zeros((self._ntact, nvals, 3),
                                       dtype="f8")

                print("\n", self.gcm_start +
                      np.timedelta64(int(t0 * 1000), "[ms]"),
                      "  |.... active: %d ....|....  out: %d ....|" %
                      (self._ntact, self._ntout))

                # compute tracks
                # The actual computation is done in C
                if self.n_procs > 1 and self._ntact >= 2 * self.n_min_part_per_thread:
                    n_threads = self._ntact // (self.n_min_part_per_thread)
                    n_part_per_thread = self._ntact // n_threads
                    if n_threads > self.n_procs:
                        n_threads = self.n_procs
                        n_part_per_thread = self._ntact // n_threads

                    print(n_part_per_thread, n_threads)

                    # spread the computation over multiple threads
                    # size of data per thread defined in configuration file
                    # chunk = self._ntact // self.n_procs
                    threads = []
                    for npr in range(n_threads):
                        # two indexes that cut the data
                        n0 = npr * n_part_per_thread
                        n1 = (npr + 1) * n_part_per_thread
                        # Make sure we don't forget some particle
                        if npr == (n_threads - 1):
                            n1 = None
                        if self.gcm_geometry in ("curvilinear",):
                            trd = Thread(
                                target=loop_nogil,
                                args=(xyz[n0:n1, ...],
                                      ijk[n0:n1, ...],
                                      out_tijk[n0:n1, ...],
                                      out_xyz[n0:n1, ...],
                                      out_code[n0:n1],
                                      self.imax,
                                      self.jmax,
                                      self.kmax,
                                      self.out_dt,
                                      self.dsmax,
                                      self.dxyz,
                                      uflux,
                                      vflux,
                                      wflux,
                                      self.dtmax,
                                      self.dstep,
                                      self.nend,
                                      self.iendw,
                                      self.iende,
                                      self.jends,
                                      self.jendn,
                                      will_write,
                                      nvals,
                                      self.subiters,
                                      self.xG,
                                      self.yG,
                                      self.dX,
                                      self.dY,
                                      self.CS,
                                      self.SN,
                                      self.Z))
                        else:
                            trd = Thread(
                                target=loopC_nogil,
                                args=(xyz[n0:n1, ...],
                                      ijk[n0:n1, ...],
                                      out_tijk[n0:n1, ...],
                                      out_xyz[n0:n1, ...],
                                      out_code[n0:n1],
                                      self.imax,
                                      self.jmax,
                                      self.kmax,
                                      self.out_dt,
                                      self.dsmax,
                                      self.dxyz,
                                      uflux,
                                      vflux,
                                      wflux,
                                      self.dtmax,
                                      self.dstep,
                                      self.nend,
                                      self.iendw,
                                      self.iende,
                                      self.jends,
                                      self.jendn,
                                      will_write,
                                      nvals,
                                      self.subiters,
                                      self.xG,
                                      self.yG,
                                      self.dX,
                                      self.dY,
                                      self.Z))
                        trd.daemon = True
                        trd.start()
                        threads.append(trd)
                    # wait for threads to finish (if they have not already)
                    [trd.join() for trd in threads]
                else:
                    if self.gcm_geometry in ("curvilinear",):
                        loop_nogil(xyz,
                                   ijk,
                                   out_tijk,
                                   out_xyz,
                                   out_code,
                                   self.imax,
                                   self.jmax,
                                   self.kmax,
                                   self.out_dt,
                                   self.dsmax,
                                   self.dxyz,
                                   uflux,
                                   vflux,
                                   wflux,
                                   self.dtmax,
                                   self.dstep,
                                   self.nend,
                                   self.iendw,
                                   self.iende,
                                   self.jends,
                                   self.jendn,
                                   will_write,
                                   nvals,
                                   self.subiters,
                                   self.xG,
                                   self.yG,
                                   self.dX,
                                   self.dY,
                                   self.CS,
                                   self.SN,
                                   self.Z)
                    else:
                        loopC_nogil(xyz,
                                    ijk,
                                    out_tijk,
                                    out_xyz,
                                    out_code,
                                    self.imax,
                                    self.jmax,
                                    self.kmax,
                                    self.out_dt,
                                    self.dsmax,
                                    self.dxyz,
                                    uflux,
                                    vflux,
                                    wflux,
                                    self.dtmax,
                                    self.dstep,
                                    self.nend,
                                    self.iendw,
                                    self.iende,
                                    self.jends,
                                    self.jendn,
                                    will_write,
                                    nvals,
                                    self.subiters,
                                    self.xG,
                                    self.yG,
                                    self.dX,
                                    self.dY,
                                    self.Z)

                # instead of doing the netCDF writing here, we rather
                # put the output in a queue, a separate thread will
                # deal with it
                # Note that we must store copies, otherwise we would be
                # changing the output before it's written
                OUTq.put((t0,
                          out_tijk.copy(),
                          out_xyz.copy(),
                          active_ids.copy(),
                          out_code.copy(),
                          False))
            # end for cycle over iterations

            # finish the output queue before closing the netcdf output file
            OUTq.join()
        # end with statement

        print("\nSimulation ended.")
    # end run
