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

from cython cimport wraparound, boundscheck, cdivision

from libc.math cimport log, exp, abs, fmin
from libc.stdio cimport printf
from libc.stdlib cimport exit, EXIT_FAILURE


@wraparound(False)
@boundscheck(False)
@cdivision(True)
cdef (double, double, double, double) \
    cross(int ijk,
          int ia,
          int ja,
          int ka,
          double i_futr,
          double i_past,
          double r0,
          double[:, :, :, ::1] flux) nogil:
    """
    Compute crossing times
    """
    # um: flux entering the cell
    # uu: flux exiting the cell
    # i_futr: interpolation coefficient for "future" GCM data
    # i_past: interpolation coefficient for "past" GCM data
    cdef double uu, um, ba, sp, sn, Du, ii
    cdef double UNDEF = 1.0e20

    um = (i_futr * flux[1, ka, ja, ia] +
          i_past * flux[0, ka, ja, ia])
    if (ijk == 1):
        ii = ia
        uu = (i_futr * flux[1, ka, ja, ia + 1] +
              i_past * flux[0, ka, ja, ia + 1])
    elif (ijk == 2):
        ii = ja
        uu = (i_futr * flux[1, ka, ja + 1, ia] +
              i_past * flux[0, ka, ja + 1, ia])
    elif (ijk == 3):
        ii = ka
        uu = (i_futr * flux[1, ka + 1, ja, ia] +
              i_past * flux[0, ka + 1, ja, ia])

    # east, north or upward crossing
    if (uu > 0.0) and (r0 != (ii + 1)):
        if um != uu:
            Du = (uu - um)
            ba = (r0 - ii) * Du + um
            if ba > 0.0:
                # eq. 7.9 & 7.6 Doos et al. 2013
                sp = (log(uu) - log(ba)) / Du
                # It can still be that there is a
                # zero flux somewhere between r0 and
                # the face
                if (sp <= 0.0):
                    sp = UNDEF
            else:
                sp = UNDEF
        else:
            sp = (1.0 + ii - r0) / uu
    else:
        sp = UNDEF

    # west, south or downward crossing
    if (um < 0.0) and (r0 != ii):
        if um != uu:
            # negative velocity, so we go from exit (uu)
            # to enter (um)
            Du = (um - uu)
            # (ii)-----(r0)---(ii+1)
            # (um)-----(ba)---(uu)
            ba = (1 + ii - r0) * Du + uu
            if ba < 0.0:
                sn = (log(-ba) - log(-um)) / Du
                # It can still be that there is a
                # zero flux somewhere between r0 and
                # the face
                if (sn <= 0.0):
                    sn = UNDEF
            else:
                sn = UNDEF
        else:
            sn = (ii - r0) / uu
    else:
        sn = UNDEF

    return uu, um, sp, sn


@cdivision(True)
cdef double newpos(int ii,
                   double r0,
                   double ds,
                   double uu,
                   double um) nogil:
    """
    Compute new position along ijk direction
    """
    # um: flux entering the cell
    # uu: flux exiting the cell
    cdef double ba, Du, Dum

    if um != uu:
        Du = uu - um
        mDu = um / Du
        # eq. 7.8 Doos et al 2013
        return (r0 - ii + mDu) * exp(Du * ds) + ii - mDu
    else:
        return r0 + uu * ds


@wraparound(False)
@boundscheck(False)
def transform_cartesian(double[:] ii,
                        double[:] jj,
                        double[:] kk,
                        float[:, ::1] X,
                        float[:, ::1] Y,
                        float[:, ::1] dX,
                        float[:, ::1] dY,
                        float[:, :, ::1] Z,
                        double[:, ::1] stxyz):

    cdef int m
    cdef int N = ii.size

    for m in range(N):
        transform_one(ii[m], jj[m], kk[m],
                        X, Y, dX, dY, Z,
                        stxyz[m, :])


@wraparound(False)
@boundscheck(False)
cdef void transform_one(
                  double i,
                  double j,
                  double k,
                  float[:, ::1] X,
                  float[:, ::1] Y,
                  float[:, ::1] dX,
                  float[:, ::1] dY,
                  float[:, :, ::1] Z,
                  double[:] stxyz) nogil:

    cdef int ii_int, jj_int, kk_int
    cdef double nx, ny, px, py, pxpy, pxny, nxpy, nxny, nz
    cdef double dxij, dyij

    ii_int = int(i)
    jj_int = int(j)
    kk_int = int(k)

    nx = i % 1
    ny = j % 1
    nz = k % 1

    # inside the cell
    if (nx >= 1e-9) & (ny >= 1e-9):
        dxij = dX[jj_int, ii_int]
        dyij = dY[jj_int, ii_int]

        # x
        stxyz[0] = X[jj_int, ii_int] + dxij * nx

        # y
        stxyz[1] = Y[jj_int, ii_int] + dyij * ny

        # z
        if nz >= 1e-9:
            stxyz[2] = (1.0 - nz) * Z[kk_int, jj_int, ii_int] + \
                        nz * Z[kk_int+1, jj_int, ii_int]
        else:
            stxyz[2] = Z[kk_int, jj_int, ii_int]
    # at the SW corner
    elif (nx < 1e-9) & (ny < 1e-9):
        # x
        stxyz[0] = X[jj_int, ii_int]

        # y
        stxyz[1] = Y[jj_int, ii_int]

        # z
        if nz >= 1e-9:
            stxyz[2] = (1.0 - nz) * Z[kk_int, jj_int, ii_int] + \
                        nz * Z[kk_int+1, jj_int, ii_int]
        else:
            stxyz[2] = Z[kk_int, jj_int, ii_int]
    # on W face
    elif nx < 1e-9:
        dyij = dY[jj_int, ii_int]

        # x
        stxyz[0] = X[jj_int, ii_int]

        # y
        stxyz[1] = Y[jj_int, ii_int] + dyij * ny

        # z
        if nz >= 1e-9:
            stxyz[2] = (1.0 - nz) * Z[kk_int, jj_int, ii_int] + \
                        nz * Z[kk_int+1, jj_int, ii_int]
        else:
            stxyz[2] = Z[kk_int, jj_int, ii_int]
    # on S face
    else:
        dxij = dX[jj_int, ii_int]

        # x
        stxyz[0] = X[jj_int, ii_int] + dxij * nx

        # y
        stxyz[1] = Y[jj_int, ii_int]

        # z
        if nz >= 1e-9:
            stxyz[2] = (1.0 - nz) * Z[kk_int, jj_int, ii_int] + \
                        nz * Z[kk_int+1, jj_int, ii_int]
        else:
            stxyz[2] = Z[kk_int, jj_int, ii_int]


@wraparound(False)
@boundscheck(False)
@cdivision(True)
cdef void loop_particle(
                  double[:, ::1] xyz,
                  short[:, ::1] ijk,
                  double[:, :, ::1] out_tijk,
                  double[:, :, ::1] out_xyz,
                  short[:] out_code,
                  int imax,
                  int jmax,
                  int kmax,
                  double out_dt,
                  float[:, :, ::1] Adsmax,
                  float[:, :, ::1] Adxyz,
                  double[:, :, :, ::1] uflux,
                  double[:, :, :, ::1] vflux,
                  double[:, :, :, ::1] wflux,
                  double dtmax,
                  double dstep,
                  int nend,
                  double[:] iendw,
                  double[:] iende,
                  double[:] jends,
                  double[:] jendn,
                  int outfreq,
                  int nvals,
                  int subiters,
                  float[:, ::1] xG,
                  float[:, ::1] yG,
                  float[:, ::1] dX,
                  float[:, ::1] dY,
                  float[:, :, ::1] Z) nogil:

    cdef double x0, y0, z0, x1, y1, z1, ts, tt, dt
    cdef double i_futr, i_past, in_futr, in_past
    cdef double dsmax, dxyz
    cdef double uu1, um1, vu1, vm1, wu1, wm1, uu
    cdef double ds, dse, dsw, dsn, dss, dsu, dsd
    cdef int ia, ja, ka, ib, jb, kb, rec_out, kk
    cdef int scrivi, kill_area
    cdef long ntrac, ntact = xyz.shape[0]

    for ntrac in range(ntact):

        # initialize stuff
        ts = 0.0  # normalised time between two GCM time steps
        tt = 0.0  # time in seconds between two GCM time steps

        # output init
        for kk in range(nvals):
            out_tijk[ntrac, kk, 0] = -1e20
        rec_out = 0

        # take coord
        # the coordinate system is as follows:
        #        cell i
        # |---------------------|
        # i    i <= x < i+1    i+1
        x1 = xyz[ntrac, 0]
        y1 = xyz[ntrac, 1]
        z1 = xyz[ntrac, 2]
        ib = ijk[ntrac, 0]
        jb = ijk[ntrac, 1]
        kb = ijk[ntrac, 2]

        scrivi = 0
        while True:
            # if we reached the new GCM output
            if tt == out_dt:
                xyz[ntrac, 0] = x1
                xyz[ntrac, 1] = y1
                xyz[ntrac, 2] = z1
                ijk[ntrac, 0] = ib
                ijk[ntrac, 1] = jb
                ijk[ntrac, 2] = kb
                break

            # time interpolation constants
            i_futr = ts % 1.0
            i_past = 1.0 - i_futr

            x0 = x1
            y0 = y1
            z0 = z1
            ia = ib
            ja = jb
            ka = kb

            dsmax = Adsmax[kb, jb, ib]
            dxyz = Adxyz[kb, jb, ib]

            # make sure we do not start from land
            #TODO do we still need this?
            if dxyz == 0.0:
                printf("\nx0: %f\n", x0)
                printf("y0: %f\n", y0)
                printf("z0: %f\n", z0)
                printf("ia: %d\n", ia)
                printf("ja: %d\n", ja)
                printf("ka: %d\n", ka)
                printf("x1: %f\n", x1)
                printf("y1: %f\n", y1)
                printf("z1: %f\n", z1)
                printf("ib: %d\n", ib)
                printf("jb: %d\n", jb)
                printf("kb: %d\n", kb)
                printf("ds: %f\n", ds)
                printf("dsmax: %f\n", dsmax)
                printf("dxyz: %f\n", dxyz)
                printf("dsn: %f\n", dsn)
                printf("dss: %f\n", dss)
                printf("dse: %f\n", dse)
                printf("dsw: %f\n", dsw)
                printf("dsu: %f\n", dsu)
                printf("dsd: %f\n", dsd)
                printf("Particle on land.")
                exit(EXIT_FAILURE)

            #
            # calculate the 3 crossing times over the box
            # choose the shortest time and calculate the
            # new positions
            #
            # solving the differential equations
            # note:
            # space variables (x,...) are dimensionless
            # time variables (ds,...) are in seconds/m^3
            #

            uu1, um1, dse, dsw = \
                cross(1, ia, ja, ka, i_futr, i_past, x0, uflux)
            vu1, vm1, dsn, dss = \
                cross(2, ia, ja, ka, i_futr, i_past, y0, vflux)
            wu1, wm1, dsu, dsd = \
                cross(3, ia, ja, ka, i_futr, i_past, z0, wflux)

            ds = fmin(fmin(fmin(fmin(fmin(fmin(dse, dsw), dsn), dss),
                      dsd), dsu), dsmax)
            if (ds == 0.0) or (ds == 1e20):
                printf("\nx0: %f\n", x0)
                printf("y0: %f\n", y0)
                printf("z0: %f\n", z0)
                printf("ia: %d\n", ia)
                printf("ja: %d\n", ja)
                printf("ka: %d\n", ka)
                printf("x1: %f\n", x1)
                printf("y1: %f\n", y1)
                printf("z1: %f\n", z1)
                printf("ib: %d\n", ib)
                printf("jb: %d\n", jb)
                printf("kb: %d\n", kb)
                printf("ds: %f\n", ds)
                printf("dsmax: %f\n", dsmax)
                printf("dxyz: %f\n", dxyz)
                printf("dsn: %f\n", dsn)
                printf("dss: %f\n", dss)
                printf("dse: %f\n", dse)
                printf("dsw: %f\n", dsw)
                printf("dsu: %f\n", dsu)
                printf("dsd: %f\n", dsd)
                printf("Cannot integrate track for unknown reasons\n")
                exit(EXIT_FAILURE)

            # now update the time
            dt = ds * dxyz

            # if time step makes the integration time
            # very close to the GCM output
            if (tt + dt) >= out_dt:
                dt = out_dt - tt
                tt = out_dt
                ts = 1.0
                ds = dt / dxyz
                end_loop = True
            else:
                end_loop = False
                tt += dt
                if dt == dtmax:
                    ts += dstep
                else:
                    ts += dt / out_dt

            # compute new time interpolation constant
            in_futr = ts % 1.0
            in_past = 1.0 - in_futr

            # now we actually compute the new position
            # eastward grid-cell exit
            if ds == dse:
                scrivi = 1
                # if at the "new time" the flow at the cell
                # face is positive the particle will exit
                uu = (in_futr * uflux[1, ka, ja, ia + 1] +
                      in_past * uflux[0, ka, ja, ia + 1])
                if uu > 0.0:
                    ib = ia + 1
                x1 = float(ia + 1)
                y1 = newpos(ja, y0, ds, vu1, vm1)
                z1 = newpos(ka, z0, ds, wu1, wm1)
                # deal with corner cases (including rounding error)
                #TODO: corner cases are extremely rare, but
                # may happen. Is there a better way to deal with them?
                if (y1 - ja) >= 1.0:  # y
                    uu = (in_futr * vflux[1, ka, ja + 1, ia] +
                          in_past * vflux[0, ka, ja + 1, ia])
                    if uu > 0.0:
                        jb = ja + 1
                    y1 = float(ja + 1)
                elif (y1 - ja) <= 0.0:
                    uu = (in_futr * vflux[1, ka, ja, ia] +
                          in_past * vflux[0, ka, ja, ia])
                    if uu < 0.0:
                        jb = ja - 1
                    y1 = float(ja)
                if (z1 - ka) >= 1.0:  # z
                    uu = (in_futr * wflux[1, ka + 1, ja, ia] +
                          in_past * wflux[0, ka + 1, ja, ia])
                    if uu > 0.0:
                        kb = ka + 1
                    z1 = float(ka + 1)
                elif (z1 - ka) <= 0.0:
                    uu = (in_futr * wflux[1, ka, ja, ia] +
                          in_past * wflux[0, ka, ja, ia])
                    if uu < 0.0:
                        kb = ka - 1
                    z1 = float(ka)
            # westward grid-cell exit
            elif ds == dsw:
                scrivi = 1
                uu = (in_futr * uflux[1, ka, ja, ia] +
                      in_past * uflux[0, ka, ja, ia])
                if uu < 0.0:
                    ib = ia - 1
                x1 = float(ia)
                y1 = newpos(ja, y0, ds, vu1, vm1)
                z1 = newpos(ka, z0, ds, wu1, wm1)
                # deal with corner cases (including rounding error)
                if (y1 - ja) >= 1.0:  # y
                    uu = (in_futr * vflux[1, ka, ja + 1, ia] +
                          in_past * vflux[0, ka, ja + 1, ia])
                    if uu > 0.0:
                        jb = ja + 1
                    y1 = float(ja + 1)
                elif (y1 - ja) <= 0.0:
                    uu = (in_futr * vflux[1, ka, ja, ia] +
                          in_past * vflux[0, ka, ja, ia])
                    if uu < 0.0:
                        jb = ja - 1
                    y1 = float(ja)
                if (z1 - ka) >= 1.0:  # z
                    uu = (in_futr * wflux[1, ka + 1, ja, ia] +
                          in_past * wflux[0, ka + 1, ja, ia])
                    if uu > 0.0:
                        kb = ka + 1
                    z1 = float(ka + 1)
                elif (z1 - ka) <= 0.0:
                    uu = (in_futr * wflux[1, ka, ja, ia] +
                          in_past * wflux[0, ka, ja, ia])
                    if uu < 0.0:
                        kb = ka - 1
                    z1 = float(ka)
            # northward grid-cell exit
            elif ds == dsn:
                scrivi = 1
                uu = (in_futr * vflux[1, ka, ja + 1, ia] +
                      in_past * vflux[0, ka, ja + 1, ia])
                if uu > 0.0:
                    jb = ja + 1
                x1 = newpos(ia, x0, ds, uu1, um1)
                y1 = float(ja + 1)
                z1 = newpos(ka, z0, ds, wu1, wm1)
                # deal with corner cases (including rounding error)
                if (x1 - ia) >= 1.0:  # x
                    uu = (in_futr * uflux[1, ka, ja, ia + 1] +
                          in_past * uflux[0, ka, ja, ia + 1])
                    if uu > 0.0:
                        ib = ia + 1
                    x1 = float(ia + 1)
                elif (x1 - ia) <= 0.0:
                    uu = (in_futr * uflux[1, ka, ja, ia] +
                          in_past * uflux[0, ka, ja, ia])
                    if uu < 0.0:
                        ib = ia - 1
                    x1 = float(ia)
                if (z1 - ka) >= 1.0:  # z
                    uu = (in_futr * wflux[1, ka + 1, ja, ia] +
                          in_past * wflux[0, ka + 1, ja, ia])
                    if uu > 0.0:
                        kb = ka + 1
                    z1 = float(ka + 1)
                elif (z1 - ka) <= 0.0:
                    uu = (in_futr * wflux[1, ka, ja, ia] +
                          in_past * wflux[0, ka, ja, ia])
                    if uu < 0.0:
                        kb = ka - 1
                    z1 = float(ka)
            # southward grid-cell exit
            elif ds == dss:
                scrivi = 1
                uu = (in_futr * vflux[1, ka, ja, ia] +
                      in_past * vflux[0, ka, ja, ia])
                if uu < 0.0:
                    jb = ja - 1
                x1 = newpos(ia, x0, ds, uu1, um1)
                y1 = float(ja)
                z1 = newpos(ka, z0, ds, wu1, wm1)
                # deal with corner cases (including rounding error)
                if (x1 - ia) >= 1.0:  # x
                    uu = (in_futr * uflux[1, ka, ja, ia + 1] +
                          in_past * uflux[0, ka, ja, ia + 1])
                    if uu > 0.0:
                        ib = ia + 1
                    x1 = float(ia + 1)
                elif (x1 - ia) <= 0.0:
                    uu = (in_futr * uflux[1, ka, ja, ia] +
                          in_past * uflux[0, ka, ja, ia])
                    if uu < 0.0:
                        ib = ia - 1
                    x1 = float(ia)
                if (z1 - ka) >= 1.0:  # z
                    uu = (in_futr * wflux[1, ka + 1, ja, ia] +
                          in_past * wflux[0, ka + 1, ja, ia])
                    if uu > 0.0:
                        kb = ka + 1
                    z1 = float(ka + 1)
                elif (z1 - ka) <= 0.0:
                    uu = (in_futr * wflux[1, ka, ja, ia] +
                          in_past * wflux[0, ka, ja, ia])
                    if uu < 0.0:
                        kb = ka - 1
                    z1 = float(ka)
            # upward grid-cell exit
            elif ds == dsu:
                scrivi = 1
                uu = (in_futr * wflux[1, ka + 1, ja, ia] +
                      in_past * wflux[0, ka + 1, ja, ia])
                if uu > 0.0:
                    kb = ka + 1
                x1 = newpos(ia, x0, ds, uu1, um1)
                y1 = newpos(ja, y0, ds, vu1, vm1)
                z1 = float(ka + 1)
                # deal with corner cases (including rounding error)
                if (x1 - ia) >= 1.0:  # x
                    uu = (in_futr * uflux[1, ka, ja, ia + 1] +
                          in_past * uflux[0, ka, ja, ia + 1])
                    if uu > 0.0:
                        ib = ia + 1
                    x1 = float(ia + 1)
                elif (x1 - ia) <= 0.0:
                    uu = (in_futr * uflux[1, ka, ja, ia] +
                          in_past * uflux[0, ka, ja, ia])
                    if uu < 0.0:
                        ib = ia - 1
                    x1 = float(ia)
                if (y1 - ja) >= 1.0:  # y
                    uu = (in_futr * vflux[1, ka, ja + 1, ia] +
                          in_past * vflux[0, ka, ja + 1, ia])
                    if uu > 0.0:
                        jb = ja + 1
                    y1 = float(ja + 1)
                elif (y1 - ja) <= 0.0:
                    uu = (in_futr * vflux[1, ka, ja, ia] +
                          in_past * vflux[0, ka, ja, ia])
                    if uu < 0.0:
                        jb = ja - 1
                    y1 = float(ja)
            # downward grid-cell exit
            elif ds == dsd:
                scrivi = 1
                uu = (in_futr * wflux[1, ka, ja, ia] +
                      in_past * wflux[0, ka, ja, ia])
                if uu < 0.0:
                    kb = ka - 1
                x1 = newpos(ia, x0, ds, uu1, um1)
                y1 = newpos(ja, y0, ds, vu1, vm1)
                z1 = float(ka)
                # deal with corner cases (including rounding error)
                if (x1 - ia) >= 1.0:  # x
                    uu = (in_futr * uflux[1, ka, ja, ia + 1] +
                          in_past * uflux[0, ka, ja, ia + 1])
                    if uu > 0.0:
                        ib = ia + 1
                    x1 = float(ia + 1)
                elif (x1 - ia) <= 0.0:
                    uu = (in_futr * uflux[1, ka, ja, ia] +
                          in_past * uflux[0, ka, ja, ia])
                    if uu < 0.0:
                        ib = ia - 1
                    x1 = float(ia)
                if (y1 - ja) >= 1.0:  # y
                    uu = (in_futr * vflux[1, ka, ja + 1, ia] +
                          in_past * vflux[0, ka, ja + 1, ia])
                    if uu > 0.0:
                        jb = ja + 1
                    y1 = float(ja + 1)
                elif (y1 - ja) <= 0.0:
                    uu = (in_futr * vflux[1, ka, ja, ia] +
                          in_past * vflux[0, ka, ja, ia])
                    if uu < 0.0:
                        jb = ja - 1
                    y1 = float(ja)
            elif end_loop or (ds == dsmax):
                scrivi = 1
                x1 = newpos(ia, x0, ds, uu1, um1)
                y1 = newpos(ja, y0, ds, vu1, vm1)
                z1 = newpos(ka, z0, ds, wu1, wm1)
                #TODO for 2D advection we will need to include
                # the check for convergence/divergence zones
            else:
                printf("\nUnrecognised ds\n")
                printf("ds: %f\n", ds)
                printf("dsmax: %f\n", dsmax)
                printf("dxyz: %f\n", dxyz)
                printf("dsn: %f\n", dsn)
                printf("dss: %f\n", dss)
                printf("dse: %f\n", dse)
                printf("dsw: %f\n", dsw)
                printf("dsu: %f\n", dsu)
                printf("dsd: %f\n", dsd)
                exit(EXIT_FAILURE)

            # check if particle entered a kill area
            kill_area = 0
            for kk in range(nend):
                if (iendw[kk] <= x1) and \
                   (x1 <= iende[kk]) and \
                   (jends[kk] <= y1) and \
                   (y1 <= jendn[kk]):
                    # we store the region where it was killed
                    kill_area = 10 + kk
                    break  # break the for loop over kill areas
            if kill_area > 0:
                # write to buffer
                if rec_out >= nvals:
                    printf("Kill region: buffer finished, make it larger!")
                    exit(EXIT_FAILURE)
                out_tijk[ntrac, rec_out, 0] = tt
                out_tijk[ntrac, rec_out, 1] = x1
                out_tijk[ntrac, rec_out, 2] = y1
                out_tijk[ntrac, rec_out, 3] = z1
                out_code[ntrac] = kill_area
                transform_one(x1, y1, z1,
                              xG, yG, dX, dY, Z,
                              out_xyz[ntrac, rec_out, :])
                rec_out += 1
                break  # break particle loop

            # check if the particle is still
            # inside the domain
            if (ib > imax) or (ib < 0) or \
               (jb > jmax) or (jb < 0):
                # write to buffer
                if rec_out >= nvals:
                    printf("Exit domain: buffer finished, make it larger!")
                    exit(EXIT_FAILURE)
                out_tijk[ntrac, rec_out, 0] = tt
                out_tijk[ntrac, rec_out, 1] = x1
                out_tijk[ntrac, rec_out, 2] = y1
                out_tijk[ntrac, rec_out, 3] = z1
                out_code[ntrac] = 1
                transform_one(x1, y1, z1,
                              xG, yG, dX, dY, Z,
                              out_xyz[ntrac, rec_out, :])
                rec_out += 1
                break
            # check if the particle exited through the top
            if (kb >= kmax):
                # write to buffer
                #TODO there should be a better way to deal with the 
                # top boundary condition
                if rec_out >= nvals:
                    printf("Airborne: buffer finished, make it larger!")
                    exit(EXIT_FAILURE)
                out_tijk[ntrac, rec_out, 0] = tt
                out_tijk[ntrac, rec_out, 1] = x1
                out_tijk[ntrac, rec_out, 2] = y1
                out_tijk[ntrac, rec_out, 3] = z1
                out_code[ntrac] = 2
                transform_one(x1, y1, z1,
                              xG, yG, dX, dY, Z,
                              out_xyz[ntrac, rec_out, :])
                rec_out += 1
                break

            # At the end of the cycle we can write the
            # position of the particle in buffer
            # gcmstep => outfreq == 1
            # always => outfreq == 2
            # cross => outfreq == 3
            if ((tt == out_dt) and (outfreq == 1)) \
               or ((outfreq == 3) and (scrivi == 1)) \
               or ((outfreq == 2) and (ts > 0.0)):
                # write to buffer
                if rec_out >= nvals:
                    printf("Normal write: buffer finished, make it larger!")
                    exit(EXIT_FAILURE)
                out_tijk[ntrac, rec_out, 0] = tt
                out_tijk[ntrac, rec_out, 1] = x1
                out_tijk[ntrac, rec_out, 2] = y1
                out_tijk[ntrac, rec_out, 3] = z1
                transform_one(x1, y1, z1,
                              xG, yG, dX, dY, Z,
                              out_xyz[ntrac, rec_out, :])
                rec_out += 1


def loopC_nogil(
                  double[:, ::1] xyz,
                  short[:, ::1] ijk,
                  double[:, :, ::1] out_tijk,
                  double[:, :, ::1] out_xyz,
                  short[:] out_code,
                  int imax,
                  int jmax,
                  int kmax,
                  double out_dt,
                  float[:, :, ::1] Adsmax,
                  float[:, :, ::1] Adxyz,
                  double[:, :, :, ::1] uflux,
                  double[:, :, :, ::1] vflux,
                  double[:, :, :, ::1] wflux,
                  double dtmax,
                  double dstep,
                  int nend,
                  double[:] iendw,
                  double[:] iende,
                  double[:] jends,
                  double[:] jendn,
                  int outfreq,
                  int nvals,
                  int subiters,
                  float[:, ::1] xG,
                  float[:, ::1] yG,
                  float[:, ::1] dX,
                  float[:, ::1] dY,
                  float[:, :, ::1] Z):
    """
    Kill the GIL
    """
    with nogil:
        loop_particle(
                  xyz,
                  ijk,
                  out_tijk,
                  out_xyz,
                  out_code,
                  imax,
                  jmax,
                  kmax,
                  out_dt,
                  Adsmax,
                  Adxyz,
                  uflux,
                  vflux,
                  wflux,
                  dtmax,
                  dstep,
                  nend,
                  iendw,
                  iende,
                  jends,
                  jendn,
                  outfreq,
                  nvals,
                  subiters,
                  xG,
                  yG,
                  dX,
                  dY,
                  Z)
