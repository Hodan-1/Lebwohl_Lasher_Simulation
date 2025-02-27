# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport cos, exp
from cython.parallel import prange

ctypedef np.float64_t DTYPE_t

########################################
cpdef inline double one_energy(double[:, :] arr, int ix, int iy, int sub_nmax, int nmax) nogil:
    cdef int i
    cdef int dx[4], dy[4]
    cdef double en = 0.0, ang

    dx[0], dy[0] = 1, 0
    dx[1], dy[1] = -1, 0
    dx[2], dy[2] = 0, 1
    dx[3], dy[3] = 0, -1

    # Use np.roll for periodic boundary conditions
    neighbors = (
        np.roll(arr, shift=1, axis=0) + np.roll(arr, shift=-1, axis=0) +
        np.roll(arr, shift=1, axis=1) + np.roll(arr, shift=-1, axis=1)
    )
    ang = arr[ix, iy] - neighbors[ix, iy]

    return 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    
########################################
def all_energy(np.ndarray[DTYPE_t, ndim=2] arr, int sub_nmax, int nmax):

   """
    Compute the total energy of the lattice using vectorised operations.
    """

neighbors = (
        np.roll(arr, shift=1, axis=0) + np.roll(arr, shift=-1, axis=0) +
        np.roll(arr, shift=1, axis=1) + np.roll(arr, shift=-1, axis=1)
    )
    energy = 0.5 * (1.0 - 3.0 * np.cos(arr - neighbors) ** 2)
    return np.sum(energy)

########################################
cpdef get_order(np.ndarray[DTYPE_t, ndim=2] arr, int nmax):
    """
    Calculate the order parameter of the lattice.
    """

    Qab_local = np.sum(np.exp(1j * arr * 6))
    Qab_total = comm.allreduce(Qab_local, op=MPI.SUM)
    return np.abs(Qab_total) / (nmax * nmax)
    
    return np.linalg.eigvals(Qab).max()

########################################
def MC_step(np.ndarray[DTYPE_t, ndim=2] arr, float Ts, int nmax):
    """
    Perform one Monte Carlo step using vectorized operations.
    """
    scale = 0.1 + Ts

    # Generate random perturbations
    angles = np.random.normal(scale=scale, size=arr.shape)
    new_arr = arr + angles

    # Compute energy change (vectorized)
    neighbors_old = (
        np.roll(arr, shift=1, axis=0) + np.roll(arr, shift=-1, axis=0) +
        np.roll(arr, shift=1, axis=1) + np.roll(arr, shift=-1, axis=1)
    )
    neighbors_new = (
        np.roll(new_arr, shift=1, axis=0) + np.roll(new_arr, shift=-1, axis=0) +
        np.roll(new_arr, shift=1, axis=1) + np.roll(new_arr, shift=-1, axis=1)
    )

    old_energy = 1.0 - 3.0 * np.cos(arr - neighbors_old) ** 2
    new_energy = 1.0 - 3.0 * np.cos(new_arr - neighbors_new) ** 2
    delta_E = 0.5 * (new_energy - old_energy)

    # Metropolis acceptance criterion (vectorized)
    accept_mask = (delta_E <= 0) | (np.exp(-delta_E / Ts) >= np.random.rand(*arr.shape))
    arr[accept_mask] = new_arr[accept_mask]

    # Exchange boundaries
    exchange_boundaries(arr, nmax)

    return np.count_nonzero(accept_mask) / (arr.shape[0] * arr.shape[1])
