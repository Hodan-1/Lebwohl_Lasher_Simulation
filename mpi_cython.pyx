# monte_carlo.pyx
import numpy as np
cimport numpy as np
from libc.math cimport cos, exp

# Declare types for NumPy arrays
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# Energy calculation (Cython-optimized)
def compute_energy(np.ndarray[DTYPE_t, ndim=2] arr, int nmax):
    cdef int i, j
    cdef int nx = arr.shape[0]
    cdef int ny = arr.shape[1]
    cdef double energy = 0.0
    cdef double diff

    for i in range(nx):
        for j in range(ny):
            diff = arr[i, j] - arr[(i + 1) % nx, j]
            energy += 0.5 * (1.0 - 3.0 * cos(diff) ** 2)
            diff = arr[i, j] - arr[(i - 1) % nx, j]
            energy += 0.5 * (1.0 - 3.0 * cos(diff) ** 2)
            diff = arr[i, j] - arr[i, (j + 1) % ny]
            energy += 0.5 * (1.0 - 3.0 * cos(diff) ** 2)
            diff = arr[i, j] - arr[i, (j - 1) % ny]
            energy += 0.5 * (1.0 - 3.0 * cos(diff) ** 2)
    return energy

# Monte Carlo step (Cython-optimized)
def mc_step(np.ndarray[DTYPE_t, ndim=2] arr, double Ts):
    cdef int i, j
    cdef int nx = arr.shape[0]
    cdef int ny = arr.shape[1]
    cdef double scale = 0.1 + Ts
    cdef int accept = 0
    cdef double delta_E, prob

    # Generate random perturbations
    cdef np.ndarray[DTYPE_t, ndim=2] angles = np.random.normal(0.0, scale, (nx, ny))
    cdef np.ndarray[DTYPE_t, ndim=2] new_arr = arr + angles

    # Compute energy change
    for i in range(nx):
        for j in range(ny):
            delta_E = 0.5 * (
                (1.0 - 3.0 * cos(new_arr[i, j] - new_arr[(i + 1) % nx, j]) ** 2) +
                (1.0 - 3.0 * cos(new_arr[i, j] - new_arr[(i - 1) % nx, j]) ** 2) +
                (1.0 - 3.0 * cos(new_arr[i, j] - new_arr[i, (j + 1) % ny]) ** 2) +
                (1.0 - 3.0 * cos(new_arr[i, j] - new_arr[i, (j - 1) % ny]) ** 2) -
                (1.0 - 3.0 * cos(arr[i, j] - arr[(i + 1) % nx, j]) ** 2) -
                (1.0 - 3.0 * cos(arr[i, j] - arr[(i - 1) % nx, j]) ** 2) -
                (1.0 - 3.0 * cos(arr[i, j] - arr[i, (j + 1) % ny]) ** 2) -
                (1.0 - 3.0 * cos(arr[i, j] - arr[i, (j - 1) % ny]) ** 2)
            )

            if delta_E <= 0 or exp(-delta_E / Ts) >= np.random.rand():
                arr[i, j] = new_arr[i, j]
                accept += 1
    return accept
