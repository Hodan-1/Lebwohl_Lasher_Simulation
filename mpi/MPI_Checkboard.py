from mpi4py import MPI
import numpy as np
import sys
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl

# Initialise MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#=======================================================================
def decompose_lattice(nmax, size):
    """
    Decompose the lattice into sub-lattices for each MPI process.
    Each process gets a contiguous block of rows.

    Args:
    - nmax (int): The size of the lattice (nmax x nmax).
    - size (int): The number of MPI processes.

    Returns:
    - sub_nmax (int): The number of rows each process will handle.
    """
    sub_nmax = nmax // size
    return sub_nmax

#=======================================================================
def initdat(nmax):
    """
    Initialise the lattice with random angles between 0 and 2π.
    Each process initialises its own sub-lattice.

    Args:
    - nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    - arr (ndarray): A 2D array representing the initialised sub-lattice.
    """
    sub_nmax = decompose_lattice(nmax, size)
    return np.random.uniform(0, 2 * np.pi, (sub_nmax, nmax))

#=======================================================================
def exchange_boundaries(arr, nmax):
    """
    Exchange boundary rows with non-blocking MPI communication.

    Args:
    - arr (ndarray): The sub-lattice array for which boundaries need to be exchanged.
    - nmax (int): The size of the lattice (nmax x nmax).
    """
    sub_nmax = arr.shape[0]
    reqs = []

    # Send to left, receive from right
    if rank > 0:
        reqs.append(comm.Isend(arr[0, :], dest=rank - 1))
        reqs.append(comm.Irecv(arr[-1, :], source=rank - 1))

    # Send to right, receive from left
    if rank < size - 1:
        reqs.append(comm.Isend(arr[-1, :], dest=rank + 1))
        reqs.append(comm.Irecv(arr[0, :], source=rank + 1))

    MPI.Request.Waitall(reqs)

#=======================================================================
def all_energy(arr, nmax):
    """
    Compute total energy efficiently using vectorised operations.

    Args:
    - arr (ndarray): The sub-lattice array.
    - nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    - total_energy (float): The total energy of the lattice.
    """
    neighbours = (
        np.roll(arr, shift=1, axis=0) + np.roll(arr, shift=-1, axis=0) +
        np.roll(arr, shift=1, axis=1) + np.roll(arr, shift=-1, axis=1)
    )
    energy_local = np.sum(0.5 * (1.0 - 3.0 * np.cos(arr - neighbours) ** 2))
    
    total_energy = np.zeros(1, dtype=np.float64)
    comm.Allreduce(np.array([energy_local]), total_energy, op=MPI.SUM)
    return total_energy[0]

#=======================================================================
def get_order(arr, nmax):
    """
    Calculate the order parameter of the lattice.

    Args:
    - arr (ndarray): The sub-lattice array.
    - nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    - order (float): The order parameter of the lattice.
    """
    Qab_local = np.sum(np.exp(1j * arr * 6))
    Qab_total = comm.allreduce(Qab_local, op=MPI.SUM)
    return np.abs(Qab_total) / (nmax * nmax)

#=======================================================================
def MC_step(arr, Ts, nmax):
    """
    Perform one Monte Carlo step with checkerboard updates using NumPy vectorisation.

    Args:
    - arr (ndarray): The sub-lattice array.
    - Ts (float): The temperature parameter.
    - nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    - acceptance_ratio (float): The ratio of accepted moves.
    """
    scale = 0.1 + Ts
    accept = 0

    # Generate random perturbations
    angles = np.random.normal(scale=scale, size=arr.shape)
    new_arr = arr + angles

    # Compute energy change (vectorised)
    neighbours_old = (
        np.roll(arr, shift=1, axis=0) + np.roll(arr, shift=-1, axis=0) +
        np.roll(arr, shift=1, axis=1) + np.roll(arr, shift=-1, axis=1)
    )
    neighbours_new = (
        np.roll(new_arr, shift=1, axis=0) + np.roll(new_arr, shift=-1, axis=0) +
        np.roll(new_arr, shift=1, axis=1) + np.roll(new_arr, shift=-1, axis=1)
    )

    old_energy = 1.0 - 3.0 * np.cos(arr - neighbours_old) ** 2
    new_energy = 1.0 - 3.0 * np.cos(new_arr - neighbours_new) ** 2
    delta_E = 0.5 * (new_energy - old_energy)

    # Metropolis acceptance criterion (vectorised)
    accept_mask = (delta_E <= 0) | (np.exp(-delta_E / Ts) >= np.random.rand(*arr.shape))
    arr[accept_mask] = new_arr[accept_mask]

    accept = np.count_nonzero(accept_mask)

    # Exchange boundaries
    exchange_boundaries(arr, nmax)

    # MPI Reduction
    total_accept = np.zeros(1, dtype=np.int64)
    comm.Allreduce(np.array([accept]), total_accept, op=MPI.SUM)
    return total_accept[0] / (nmax * nmax)

#=======================================================================
def plotdat(arr, pflag, nmax):
    """
    Plot the lattice configuration (only by root process).

    Args:
    - arr (ndarray): The sub-lattice array.
    - pflag (int): Plot flag to determine the type of plot.
    - nmax (int): The size of the lattice (nmax x nmax).
    """
    if rank != 0 or pflag == 0:
        return

    u, v = np.cos(arr), np.sin(arr)
    x, y = np.meshgrid(np.arange(nmax), np.arange(nmax))

    if pflag == 1:
        mpl.rc('image', cmap='rainbow')
        cols = np.array([[one_energy(arr, i, j, nmax) for j in range(nmax)] for i in range(nmax)])
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gray')
        cols = np.ones((nmax, nmax))  # Default white background
        norm = plt.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots()
    ax.imshow(cols, cmap='rainbow', norm=norm, origin='lower')
    ax.quiver(x, y, u, v, scale=20, pivot='middle', colour='black')
    ax.set_aspect('equal')
    plt.show()

#=======================================================================
def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    """
    Save simulation data to a file (only by root process).

    Args:
    - arr (ndarray): The sub-lattice array.
    - nsteps (int): The number of Monte Carlo steps.
    - Ts (float): The temperature parameter.
    - runtime (float): The total runtime of the simulation.
    - ratio (ndarray): Array of acceptance ratios.
    - energy (ndarray): Array of energy values.
    - order (ndarray): Array of order parameter values.
    - nmax (int): The size of the lattice (nmax x nmax).
    """
    if rank != 0:
        return

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"LL-Output-{current_datetime}.txt"

    header = (
        f"# File created: {current_datetime}\n"
        f"# Lattice size: {nmax}x{nmax}\n"
        f"# MC steps: {nsteps}\n"
        f"# Temperature: {Ts:.3f}\n"
        f"# Runtime: {runtime:.6f} s\n"
        "# Step    Ratio    Energy    Order\n"
    )

    # Stack data for efficient saving
    data = np.column_stack((np.arange(nsteps + 1), ratio, energy, order))
    np.savetxt(filename, data, fmt="%5d %8.4f %12.4f %8.4f", header=header, comments='')

#=======================================================================
def main(nsteps, nmax, temp, pflag):
    """
    Run the Monte Carlo simulation.

    Args:
    - nsteps (int): The number of Monte Carlo steps.
    - nmax (int): The size of the lattice (nmax x nmax).
    - temp (float): The temperature parameter.
    - pflag (int): Plot flag to determine the type of plot.
    """
    # Initialise the sub-lattice
    lattice = initdat(nmax)
    if rank == 0:
        plotdat(lattice, pflag, nmax)

    # Arrays to store results
    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio = np.zeros(nsteps + 1, dtype=np.float64)
    order = np.zeros(nsteps + 1, dtype=np.float64)

    # Initial values
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)

    # Monte Carlo loop
    start_time = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    runtime = time.time() - start_time

    # Final output 
    if rank == 0:
        print(f"Size: {nmax}, Steps: {nsteps}, T*: {temp:.3f}, Order: {order[-1]:.3f}, Time: {runtime:.6f} s")
        plotdat(lattice, pflag, nmax)
        savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)

#=======================================================================
if __name__ == '__main__':
    if len(sys.argv) == 5:
        nsteps = int(sys.argv[1])
        nmax = int(sys.argv[2])
        temp = float(sys.argv[3])
        pflag = int(sys.argv[4])
        main(nsteps, nmax, temp, pflag)
    else:
        print(f"Usage: mpirun -np <num_processes> python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
