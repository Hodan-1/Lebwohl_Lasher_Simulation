# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport cos, exp
from cython.parallel import prange

ctypedef np.float64_t DTYPE_t

########################################
# Compute the energy of a single cell  #
########################################
cpdef inline double one_energy(double[:, :] arr, int ix, int iy, int sub_nmax, int nmax) nogil:
    cdef int i
    cdef int dx[4], dy[4]
    cdef double en = 0.0, ang
    cdef int new_ix, new_iy
    

    dx[0], dy[0] = 1, 0
    dx[1], dy[1] = -1, 0
    dx[2], dy[2] = 0, 1
    dx[3], dy[3] = 0, -1

    for i in range(4): 
        # Handle periodic boundaries manually
        new_ix = (ix + dx[i]) % sub_nmax
        new_iy = (iy + dy[i]) % nmax

        # Ensure indices are non-negative
        if new_ix < 0:
            new_ix += sub_nmax
        if new_iy < 0:
            new_iy += nmax

        ang = arr[ix, iy] - arr[new_ix, new_iy]
        en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    return en

########################################
# Calculate the energy of the lattice  #
########################################
def all_energy(np.ndarray[DTYPE_t, ndim=2] arr, int sub_nmax, int nmax):
    cdef double enall = 0.0
    cdef int i, j
    cdef double[:, :] arr_view = arr
# Debug: Print array shape
    print(f"Calculating energy: arr.shape = {arr.shape[0]}x{arr.shape[1]}, nmax = {nmax}")
    
    for i in prange(nmax, nogil=True):
        for j in range(nmax):
            enall += one_energy(arr_view, i, j, sub_nmax, nmax)
    
    return enall

########################################
# Calculate the order parameter        #
########################################
cpdef get_order(np.ndarray[DTYPE_t, ndim=2] arr, int nmax):
    cdef double N2 = nmax * nmax
    cdef np.ndarray[DTYPE_t, ndim=2] cos_arr = np.cos(arr)
    cdef np.ndarray[DTYPE_t, ndim=2] sin_arr = np.sin(arr)
    
    cdef double s00 = 3.0 * np.sum(cos_arr * cos_arr) - N2
    cdef double s01 = 3.0 * np.sum(cos_arr * sin_arr)
    cdef double s11 = 3.0 * np.sum(sin_arr * sin_arr) - N2
    
    Qab = np.array([[s00, s01, 0.0], [s01, s11, 0.0], [0.0, 0.0, -0.5]]) / (2.0 * N2)
    
    return np.linalg.eigvals(Qab).max()

########################################
# Perform one Monte Carlo step         #
########################################
def MC_step(np.ndarray[DTYPE_t, ndim=2] arr, float Ts, int nmax):
    cdef double scale = 0.1 + Ts
    cdef int accept = 0, i, j, ix, iy
    cdef double ang, en0, en1
    cdef double[:, :] arr_view = arr
    
    xran = np.random.randint(0, arr.shape[0], size=(arr.shape[0], arr.shape[1]), dtype=np.int32)
    yran = np.random.randint(0, nmax, size=(arr.shape[0], arr.shape[1]), dtype=np.int32)
    aran = np.random.normal(scale=scale, size=(arr.shape[0], arr.shape[1]))
    urn = np.random.rand(arr.shape[0], arr.shape[1])
    
    cdef int[:, :] xran_view = xran, yran_view = yran
    cdef double[:, :] aran_view = aran, urn_view = urn
    
    for i in prange(nmax, nogil=True):
        for j in range(arr.shape[1]):
            ix, iy = xran_view[i, j], yran_view[i, j]
            ang = aran_view[i, j]
            en0 = one_energy(arr_view, ix, iy,arr.shape[0], nmax)
            arr_view[ix, iy] += ang
            en1 = one_energy(arr_view, ix, iy, arr.shape[0], nmax)
            
            if en1 <= en0 or exp(-(en1 - en0) / Ts) >= urn_view[i, j]:
                accept += 1
            else:
                arr_view[ix, iy] -= ang
    
    return accept / (nmax * nmax)
