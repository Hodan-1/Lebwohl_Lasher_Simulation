# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
from libc.math cimport cos, exp
from numpy cimport ndarray

ctypedef np.float64_t DTYPE_t

########################################
# Compute the energy of a single cell  #
########################################
cpdef inline double one_energy(double[:, :] arr, int ix, int iy, int nmax) nogil:
    cdef double en = 0.0
    cdef int ixp = (ix + 1) % nmax
    cdef int ixm = (ix - 1 + nmax) % nmax  # ensure positive modulo
    cdef int iyp = (iy + 1) % nmax
    cdef int iym = (iy - 1 + nmax) % nmax
    cdef double ang, c_val

    # Neighbor to the right
    ang = arr[ix, iy] - arr[ixp, iy]
    c_val = cos(ang)
    en += 0.5 * (1.0 - 3.0 * c_val * c_val)
    
    # Neighbor to the left
    ang = arr[ix, iy] - arr[ixm, iy]
    c_val = cos(ang)
    en += 0.5 * (1.0 - 3.0 * c_val * c_val)
    
    # Neighbor above
    ang = arr[ix, iy] - arr[ix, iyp]
    c_val = cos(ang)
    en += 0.5 * (1.0 - 3.0 * c_val * c_val)
    
    # Neighbor below
    ang = arr[ix, iy] - arr[ix, iym]
    c_val = cos(ang)
    en += 0.5 * (1.0 - 3.0 * c_val * c_val)
    
    return en

########################################
# Calculate the energy of the lattice  #
########################################
def all_energy(np.ndarray[DTYPE_t, ndim=2] arr, int nmax):
    cdef double enall = 0.0
    cdef int i, j
    cdef double[:, :] arr_view = arr  # create a typed memoryview

    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr_view, i, j, nmax)
    return enall

########################################
# Calculate the order parameter of the lattice
########################################
def get_order(np.ndarray[DTYPE_t, ndim=2] arr, int nmax):
    """
    Here we avoid an explicit 4-loop over indices by using NumPy’s vectorized operations.
    Note that lab[2] is identically zero, so only the (0,0), (0,1), and (1,1) components need computing.
    """
    cdef np.ndarray[DTYPE_t, ndim=2] Qab = np.empty((3, 3), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] cos_arr = np.cos(arr)
    cdef np.ndarray[DTYPE_t, ndim=2] sin_arr = np.sin(arr)
    cdef double N2 = nmax * nmax
    cdef double s00 = 3.0 * np.sum(cos_arr * cos_arr) - N2
    cdef double s01 = 3.0 * np.sum(cos_arr * sin_arr)
    cdef double s11 = 3.0 * np.sum(sin_arr * sin_arr) - N2

    Qab[0, 0] = s00 / (2.0 * N2)
    Qab[0, 1] = s01 / (2.0 * N2)
    Qab[1, 0] = s01 / (2.0 * N2)
    Qab[1, 1] = s11 / (2.0 * N2)
    Qab[0, 2] = Qab[2, 0] = Qab[1, 2] = Qab[2, 1] = 0.0
    Qab[2, 2] = - N2 / (2.0 * N2)  # equals -0.5

    eigenvalues = np.linalg.eigvals(Qab)
    return eigenvalues.max()

########################################
# Perform one Monte Carlo step         #
########################################
def MC_step(np.ndarray[DTYPE_t, ndim=2] arr, float Ts, int nmax):
    """
    For each Monte Carlo step:
      - Pre-generate random indices and trial angle changes,
      - Compute the local energy before and after the trial move,
      - Decide to accept/reject using the Metropolis criterion.
    """
    cdef double scale = 0.1 + Ts
    cdef int accept = 0
    # Pre-generate random numbers in bulk:
    cdef np.ndarray[np.int32_t, ndim=2] xran = np.random.randint(0, nmax, size=(nmax, nmax), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2] yran = np.random.randint(0, nmax, size=(nmax, nmax), dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=2] aran = np.random.normal(scale=scale, size=(nmax, nmax))
    cdef np.ndarray[DTYPE_t, ndim=2] urn = np.random.rand(nmax, nmax)  # uniform [0,1)
    
    cdef int i, j, ix, iy
    cdef double ang, en0, en1, boltz
    cdef double[:, :] arr_view = arr  # memoryview for lattice
    cdef int[:, :] xran_view = xran
    cdef int[:, :] yran_view = yran
    cdef double[:, :] aran_view = aran
    cdef double[:, :] urn_view = urn

    for i in range(nmax):
        for j in range(nmax):
            ix = xran_view[i, j]
            iy = yran_view[i, j]
            ang = aran_view[i, j]
            en0 = one_energy(arr_view, ix, iy, nmax)
            arr_view[ix, iy] += ang
            en1 = one_energy(arr_view, ix, iy, nmax)
            if en1 <= en0:
                accept += 1
            else:
                boltz = exp(-(en1 - en0) / Ts)
                if boltz >= urn_view[i, j]:
                    accept += 1
                else:
                    arr_view[ix, iy] -= ang
    return accept / (nmax * nmax)
