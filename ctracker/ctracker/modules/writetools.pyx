from cython cimport wraparound, boundscheck


@wraparound(False)
@boundscheck(False)
cpdef int loc_time_fast(double[:] timeax,
                        int N,
                        double time,
                        int monotonic):

    cdef int ii
    cdef int loc_time = -9999999

    # We go backwards since the time we are looking for
    # is usually towards the end
    for ii in range(N - 1, -1, -1):
        if abs(timeax[ii] - time) <= 1.0e-9:  # we found it
            return ii
        # the axis is monotonic, so if we find a value
        # smaller than time, it means there is no time in
        # the axis
        elif monotonic and (timeax[ii] < time):
            return loc_time

    return loc_time
