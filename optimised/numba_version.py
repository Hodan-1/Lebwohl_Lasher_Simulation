import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numba
from numba import njit, prange

#=======================================================================
@njit(cache=True)
def initdat(nmax):
    """
    Initialise the lattice with random orientations.

    Args:
    nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    np.ndarray: A 2D array representing the lattice with random angles.
    """
    return np.random.random_sample((nmax, nmax)) * 2.0 * np.pi

#=======================================================================
def plotdat(arr, pflag, nmax):
    """
    Plot the lattice using matplotlib.

    Args:
    arr (np.ndarray): The lattice array to be plotted.
    pflag (int): Plot flag to determine the type of plot.
    nmax (int): The size of the lattice (nmax x nmax).
    """
    if pflag == 0:
        return

    # Calculate the cosine and sine of the angles for quiver plot
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax, nmax))

    # Determine the colour scheme based on the plot flag
    if pflag == 1:
        mpl.rc('image', cmap='rainbow')
        for i in prange(nmax):
            for j in prange(nmax):
                cols[i, j] = one_energy(arr, i, j, nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    # Create the quiver plot
    quiveropts = dict(headlength=0, pivot='middle', headwidth=1, scale=1.1 * nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols, norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()

#=======================================================================
def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    """
    Save simulation data to a file.

    Args:
    arr (np.ndarray): The lattice array.
    nsteps (int): Number of Monte Carlo steps.
    Ts (float): Reduced temperature.
    runtime (float): Total runtime of the simulation.
    ratio (np.ndarray): Array of acceptance ratios.
    energy (np.ndarray): Array of energy values.
    order (np.ndarray): Array of order parameter values.
    nmax (int): The size of the lattice (nmax x nmax).
    """
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = f"LL-Output-{current_datetime}.txt"
    with open(filename, "w") as FileOut:
        print("#=====================================================", file=FileOut)
        print(f"# File created:        {current_datetime}", file=FileOut)
        print(f"# Size of lattice:     {nmax}x{nmax}", file=FileOut)
        print(f"# Number of MC steps:  {nsteps}", file=FileOut)
        print(f"# Reduced temperature: {Ts:5.3f}", file=FileOut)
        print(f"# Run time (s):        {runtime:8.6f}", file=FileOut)
        print("#=====================================================", file=FileOut)
        print("# MC step:  Ratio:     Energy:   Order:", file=FileOut)
        print("#=====================================================", file=FileOut)
        for i in range(nsteps + 1):
            print(f"   {i:05d}    {ratio[i]:6.4f} {energy[i]:12.4f}  {order[i]:6.4f} ", file=FileOut)

#=======================================================================
@njit (fastmath=True, cache=True)
def one_energy(arr, ix, iy, nmax):
    """
    Compute the energy of a single cell in the lattice.

    Args:
    arr (np.ndarray): The lattice array.
    ix (int): x-coordinate of the cell.
    iy (int): y-coordinate of the cell.
    nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    float: The energy of the cell at (ix, iy).
    """
    en = 0.0
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    ang = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    return en

#=======================================================================
@njit(fastmath=True, parallel=True, cache=True)
def all_energy(arr, nmax):
    """
    Compute the total energy of the entire lattice.

    Args:
    arr (np.ndarray): The lattice array.
    nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    float: The total energy of the lattice.
    """
    enall = 0.0
    for i in prange(nmax):
        for j in prange(nmax):
            enall += one_energy(arr, i, j, nmax)
    return enall

#=======================================================================
@njit(fastmath= True, parallel = True, cache=True)
def get_order(arr, nmax):
    """
    Calculate the order parameter of the lattice.

    Args:
    arr (np.ndarray): The lattice array.
    nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    float: The maximum eigenvalue of the Qab matrix, representing the order parameter.
    """
    Qab = np.zeros((3, 3))
    delta = np.eye(3)

    lab = np.stack((np.cos(arr), np.sin(arr), np.zeros_like(arr)), axis=0)
    for a in prange(3):
        for b in prange(3):
            Qab[a, b] = np.sum(3 * lab[a] * lab[b] - delta[a, b])
    Qab = Qab / (2 * nmax * nmax)
    eigenvalues = np.linalg.eigvalsh(Qab)
    return np.max(eigenvalues)

#=======================================================================
@njit(fastmath=True, parallel=True, cache=True)
def MC_step(arr, Ts, nmax):
    """
    Perform one Monte Carlo step on the lattice.

    Args:
    arr (np.ndarray): The lattice array.
    Ts (float): Reduced temperature.
    nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    float: The acceptance ratio of the Monte Carlo step.
    """
    scale = 0.1 + Ts
    accept = 0
    xran = np.random.randint(0, nmax, size=(nmax, nmax))
    yran = np.random.randint(0, nmax, size=(nmax, nmax))
    aran = np.random.normal(loc=0.0, scale=scale, size=(nmax, nmax))

    for i in prange(nmax):
        for j in prange(nmax):
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            if en1 <= en0:
                accept += 1
            else:
                boltz = np.exp(-(en1 - en0) / Ts)
                if boltz >= np.random.uniform(0.0, 1.0):
                    accept += 1
                else:
                    arr[ix, iy] -= ang
    return accept / (nmax * nmax)

#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    """
    Main function to run the simulation.

    Args:
    program (str): The name of the program.
    nsteps (int): Number of Monte Carlo steps.
    nmax (int): The size of the lattice (nmax x nmax).
    temp (float): Reduced temperature.
    pflag (int): Plot flag to determine the type of plot.
    """
    # Initialise the lattice
    lattice = initdat(nmax)
    plotdat(lattice, pflag, nmax)

    # Initialise arrays to store energy, ratio, and order values
    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio = np.zeros(nsteps + 1, dtype=np.float64)
    order = np.zeros(nsteps + 1, dtype=np.float64)

    # Calculate initial energy, ratio, and order
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)

    # Perform Monte Carlo steps
    initial = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    final = time.time()
    runtime = final - initial

    # Print and save the results
    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}: Order: {order[nsteps - 1]:5.3f}, Time: {runtime:8.6f} s")
    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)

#=======================================================================
if __name__ == '__main__':
    if len(sys.argv) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")