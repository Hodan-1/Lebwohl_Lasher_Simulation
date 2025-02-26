import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit

#=======================================================================
@njit
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax)) * 2.0 * np.pi
    return arr

#=======================================================================
def plotdat(arr, pflag, nmax):
    """
    Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
      The angles plot uses a cyclic color map representing the range from
      0 to pi.  The energy plot is normalised to the energy range of the
      current frame.
    Returns:
      NULL
    """
    if pflag == 0:
        return

    u, v = np.cos(arr), np.sin(arr)
    x, y = np.arange(nmax), np.arange(nmax)
    cols = np.zeros((nmax, nmax))

    if pflag == 1:  # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        cols = one_energy(arr, nmax)  # Use vectorized one_energy
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:  # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:  # black and white plot
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
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename, "w")
    # Write a header with run parameters
    print("#=====================================================", file=FileOut)
    print("# File created:        {:s}".format(current_datetime), file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax, nmax), file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps), file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts), file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime), file=FileOut)
    print("#=====================================================", file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:", file=FileOut)
    print("#=====================================================", file=FileOut)
    # Write the columns of data
    for i in range(nsteps + 1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i, ratio[i], energy[i], order[i]), file=FileOut)
    FileOut.close()

#=======================================================================
@njit
def one_energy(arr, nmax, x, y):
    """
    Computes the energy of a single cell at position (x, y).
    """
    # Handle periodic boundary conditions
    left = arr[x, (y - 1) % nmax]
    right = arr[x, (y + 1) % nmax]
    top = arr[(x - 1) % nmax, y]
    bottom = arr[(x + 1) % nmax, y]

    # Compute energy contributions from neighbors
    energy = 0.5 * (1.0 - 3.0 * np.cos(arr[x, y] - left)**2) + \
             0.5 * (1.0 - 3.0 * np.cos(arr[x, y] - right)**2) + \
             0.5 * (1.0 - 3.0 * np.cos(arr[x, y] - top)**2) + \
             0.5 * (1.0 - 3.0 * np.cos(arr[x, y] - bottom)**2)

    return energy

#=======================================================================
@njit
def all_energy(arr, nmax):
    """
    Vectorized version of all_energy using slicing.
    Computes the total energy of the lattice.
    """
    return np.sum(one_energy(arr, nmax))

#=======================================================================
@njit
def get_order(arr, nmax):
    """
    Vectorized version of get_order using slicing and broadcasting.
    Computes the order parameter for the lattice.
    """
    # Generate a 3D unit vector for each cell (i,j)
    cos_arr = np.cos(arr)
    sin_arr = np.sin(arr)
    zeros_arr = np.zeros_like(arr)
    
    # Stack to create a (3, nmax, nmax) array
    lab = np.stack((cos_arr, sin_arr, zeros_arr), axis=0)

    # Compute Qab tensor using broadcasting
    Qab = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            Qab[a, b] = np.sum(3 * lab[a] * lab[b] - (a == b))

    Qab = Qab / (2 * nmax * nmax)
    eigenvalues = np.linalg.eigvals(Qab)
    return np.max(eigenvalues)

#=======================================================================
@njit
def MC_step(arr, Ts, nmax):
    """
    Performs one Monte Carlo step for all cells in the lattice.
    """
    scale = 0.1 + Ts
    accept_count = 0  # Counter for accepted moves

    # Loop over all cells
    for _ in range(nmax * nmax):
        # Randomly select a cell
        x = np.random.randint(0, nmax)
        y = np.random.randint(0, nmax)

        # Propose a new angle for the selected cell
        new_angle = arr[x, y] + np.random.normal(0.0, scale)

        # Compute the energy difference for the selected cell
        en0 = one_energy(arr, nmax, x, y)  # Current energy
        old_angle = arr[x, y]  # Save the old angle
        arr[x, y] = new_angle  # Temporarily update the angle
        en1 = one_energy(arr, nmax, x, y)  # New energy
        delta_E = en1 - en0

        # Acceptance/rejection rule
        if delta_E <= 0 or np.exp(-delta_E / Ts) >= np.random.uniform(0.0, 1.0):
            accept_count += 1  # Accept the move
        else:
            arr[x, y] = old_angle  # Reject the move (revert the change)

    # Calculate acceptance ratio
    accept_ratio = accept_count / (nmax * nmax)
    return accept_ratio
#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    # Create and initialise lattice
    lattice = initdat(nmax)
    # Plot initial frame of lattice
    plotdat(lattice, pflag, nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio = np.zeros(nsteps + 1, dtype=np.float64)
    order = np.zeros(nsteps + 1, dtype=np.float64)
    # Set initial values in arrays
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5  # ideal value
    order[0] = get_order(lattice, nmax)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    final = time.time()
    runtime = final - initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax, nsteps, temp, order[nsteps - 1], runtime))
    # Plot final frame of lattice and generate output file
    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
    plotdat(lattice, pflag, nmax)

#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))