import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set a seed for reproducibility in random number generation
np.random.seed(42)

#=======================================================================
def initdat(nmax):
    """
    Initialise the lattice with random orientations in the range [0, 2π].

    Args:
    - nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    - arr (numpy.ndarray): A 2D array representing the lattice with random orientations.
    """
    return np.random.random_sample((nmax, nmax)) * 2.0 * np.pi

#=======================================================================
def plotdat(arr, pflag, nmax):
    """
    Plot the lattice data using quiver plots.

    Args:
    - arr (numpy.ndarray): The lattice data to be plotted.
    - pflag (int): A flag to determine the type of plot:
        - 0: No plot.
        - 1: Colour the arrows according to energy.
        - 2: Colour the arrows according to angle.
        - Other: Black plot.
    - nmax (int): The size of the lattice (nmax x nmax).
    """
    if pflag == 0:
        return

    # Compute the x and y components of the arrows
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros_like(arr)

    if pflag == 1:  # Colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        # Compute energy for all cells using np.roll
        right = np.roll(arr, -1, axis=0)
        left = np.roll(arr, 1, axis=0)
        up = np.roll(arr, -1, axis=1)
        down = np.roll(arr, 1, axis=1)

        ang1 = arr - right
        ang2 = arr - left
        ang3 = arr - up
        ang4 = arr - down

        cols = 0.5 * (1.0 - 3.0 * np.cos(ang1)**2) + \
               0.5 * (1.0 - 3.0 * np.cos(ang2)**2) + \
               0.5 * (1.0 - 3.0 * np.cos(ang3)**2) + \
               0.5 * (1.0 - 3.0 * np.cos(ang4)**2)

        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:  # Colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:  # Black plot
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

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
    - arr (numpy.ndarray): The lattice data.
    - nsteps (int): The number of Monte Carlo steps.
    - Ts (float): The reduced temperature.
    - runtime (float): The total runtime of the simulation.
    - ratio (numpy.ndarray): The acceptance ratio at each step.
    - energy (numpy.ndarray): The energy of the lattice at each step.
    - order (numpy.ndarray): The order parameter at each step.
    - nmax (int): The size of the lattice (nmax x nmax).
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
def all_energy(arr, nmax):
    """
    Compute the energy of the entire lattice using np.roll.

    Args:
    - arr (numpy.ndarray): The lattice data.
    - nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    - en (float): The total energy of the lattice.
    """
    right = np.roll(arr, -1, axis=0)
    left = np.roll(arr, 1, axis=0)
    up = np.roll(arr, -1, axis=1)
    down = np.roll(arr, 1, axis=1)

    ang1 = arr - right
    ang2 = arr - left
    ang3 = arr - up
    ang4 = arr - down

    en = 0.5 * (1.0 - 3.0 * np.cos(ang1)**2) + \
         0.5 * (1.0 - 3.0 * np.cos(ang2)**2) + \
         0.5 * (1.0 - 3.0 * np.cos(ang3)**2) + \
         0.5 * (1.0 - 3.0 * np.cos(ang4)**2)

    return np.sum(en)

#=======================================================================
def MC_step(arr, Ts, nmax):
    """
    Perform one Monte Carlo step using vectorised operations.

    Args:
    - arr (numpy.ndarray): The lattice data.
    - Ts (float): The reduced temperature.
    - nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    - accept (float): The acceptance ratio for this step.
    """
    scale = 0.1 + Ts
    accept = 0

    # Generate random numbers for all cells at once
    xran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    yran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    aran = np.random.normal(scale=scale, size=(nmax, nmax))

    # Use np.roll to get the neighbours with periodic boundary conditions
    right = np.roll(arr, -1, axis=0)
    left = np.roll(arr, 1, axis=0)
    up = np.roll(arr, -1, axis=1)
    down = np.roll(arr, 1, axis=1)

    # Compute the energy before the change
    en0 = 0.5 * (1.0 - 3.0 * np.cos(arr - right)**2) + \
          0.5 * (1.0 - 3.0 * np.cos(arr - left)**2) + \
          0.5 * (1.0 - 3.0 * np.cos(arr - up)**2) + \
          0.5 * (1.0 - 3.0 * np.cos(arr - down)**2)

    # Make the change
    arr[xran, yran] += aran

    # Compute the energy after the change
    en1 = 0.5 * (1.0 - 3.0 * np.cos(arr - right)**2) + \
          0.5 * (1.0 - 3.0 * np.cos(arr - left)**2) + \
          0.5 * (1.0 - 3.0 * np.cos(arr - up)**2) + \
          0.5 * (1.0 - 3.0 * np.cos(arr - down)**2)

    # Compute the Boltzmann factor
    boltz = np.exp(-(en1 - en0) / Ts)

    # Accept or reject the change
    accept_mask = (en1 <= en0) | (boltz >= np.random.uniform(0.0, 1.0, size=(nmax, nmax)))
    arr[~accept_mask] -= aran[~accept_mask]
    accept = np.sum(accept_mask)

    return accept / (nmax * nmax)

#=======================================================================
def get_order(arr, nmax):
    """
    Compute the order parameter using the Q tensor approach.

    Args:
    - arr (numpy.ndarray): The lattice data.
    - nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    - order (float): The order parameter of the lattice.
    """
    lab = np.vstack((np.cos(arr), np.sin(arr), np.zeros_like(arr))).reshape(3, nmax, nmax)
    Qab = np.zeros((3, 3))
    delta = np.eye(3, 3)
    for a in range(3):
        for b in range(3):
            Qab[a, b] = np.sum(3 * lab[a] * lab[b] - delta[a, b])
    Qab = Qab / (2 * nmax * nmax)
    eigenvalues = np.linalg.eigvalsh(Qab)
    return np.max(eigenvalues)

#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    """
    Main function to run the simulation.

    Args:
    - program (str): The name of the program.
    - nsteps (int): The number of Monte Carlo steps.
    - nmax (int): The size of the lattice (nmax x nmax).
    - temp (float): The reduced temperature.
    - pflag (int): The plot flag to determine the type of plot.
    """
    lattice = initdat(nmax)
    plotdat(lattice, pflag, nmax)

    energy = np.zeros(nsteps + 1)
    ratio = np.zeros(nsteps + 1)
    order = np.zeros(nsteps + 1)

    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5  # Ideal value
    order[0] = get_order(lattice, nmax)

    initial = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    final = time.time()
    runtime = final - initial

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