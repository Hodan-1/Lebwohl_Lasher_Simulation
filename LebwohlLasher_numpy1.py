"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#=======================================================================
def initdat(nmax):
"""
    Initialise the lattice with random orientations in the range [0, 2π].

    Parameters:
    - nmax (int): The size of the lattice (nmax x nmax).

    Returns:
    - arr (numpy.ndarray): A 2D array representing the lattice with random orientations.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
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

#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
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
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
def one_energy
Vectorized energy calculation for the entire lattice.           """                                                             # Roll the array to get neighbors (periodic boundary condit>    right = np.roll(arr, -1, axis=0)                                left = np.roll(arr, 1, axis=0)                                  up = np.roll(arr, -1, axis=1)                                   down = np.roll(arr, 1, axis=1)                                                                                                  ang1 = arr - right                                              ang2 = arr - left                                               ang3 = arr - up                                                 ang4 = arr - down                                                                                                               # Compute energy contributions from all four neighbors          energy = 0.5 * (1.0 - 3.0 * np.cos(arr - right)**2)             energy += 0.5 * (1.0 - 3.0 * np.cos(arr - left)**2)             energy += 0.5 * (1.0 - 3.0 * np.cos(arr - up)**2)               energy += 0.5 * (1.0 - 3.0 * np.cos(arr - down)**2)
 """
    # Roll the array to get neighbors (periodic boundary condit>
    right = np.roll(arr, -1, axis=0)
    left = np.roll(arr, 1, axis=0)
    up = np.roll(arr, -1, axis=1)
    down = np.roll(arr, 1, axis=1)
    ang1 = arr - right                                              ang2 = arr - left                                               ang3 = arr - up                                                 ang4 = arr - down

    # Compute energy contributions from all four neighbors
    energy = 0.5 * (1.0 - 3.0 * np.cos(arr - right)**2)
    energy += 0.5 * (1.0 - 3.0 * np.cos(arr - left)**2)
    energy += 0.5 * (1.0 - 3.0 * np.cos(arr - up)**2)
    energy += 0.5 * (1.0 - 3.0 * np.cos(arr - down)**2)                                                                    ang1 = arr - right                                              ang2 = arr - left                                               ang3 = arr - up                                                 ang4 = arr - down                                                                                                               # Compute energy contributions from all four neighbors          energy = 0.5 * (1.0 - 3.0 * np.cos(arr - right)**2)             energy += 0.5 * (1.0 - 3.0 * np.cos(arr - left)**2)             energy += 0.5 * (1.0 - 3.0 * np.cos(arr - up)**2)               energy += 0.5 * (1.0 - 3.0 * np.cos(arr - down)**2)    
   return energy
def all_energy(arr, nmax):
    """
    Vectorized energy calculation for the entire lattice.
    """
	
    
    # Sum the energy over the entire lattice
    return np.sum(one_energy)

#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>#==============================================================>

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
def MC_step(arr, Ts, nmax):
    """
    Vectorized Monte Carlo step for the entire lattice.
    """
    scale = 0.1 + Ts
    
    # Generate random changes for all lattice sites
    xran = np.random.randint(0, nmax, size=(nmax, nmax))
    yran = np.random.randint(0, nmax, size=(nmax, nmax))
    aran = np.random.normal(scale=scale, size=(nmax, nmax))
    
    # Compute the energy before changes
    en0 = all_energy(arr, nmax)
    
    # Apply proposed changes to the lattice
    proposed_arr = arr.copy()
    proposed_arr[xran, yran] += aran
    
    # Compute the energy after changes
    en1 = all_energy(proposed_arr, nmax)
    
    # Calculate the Boltzmann factor for all sites
    boltz = np.exp(-(en1 - en0) / Ts)
    
    # Accept or reject changes based on the Metropolis criterion
    accept_mask = (en1 <= en0) | (boltz >= np.random.uniform(0.0, 1.0, size=(nmax, nmax)))
    
    # Update the lattice with accepted changes
    arr[accept_mask] = proposed_arr[accept_mask]
    
    # Calculate the acceptance ratio
    accept_ratio = np.mean(accept_mask)
    return accept_ratio
#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    """
    Main function to run the Lebwohl-Lasher simulation.
    """
    # Initialize the lattice
    lattice = initdat(nmax)
    
    # Plot the initial frame
    plotdat(lattice, pflag, nmax)
    
    # Arrays to store energy, acceptance ratio, and order parameter
    energy = np.zeros(nsteps + 1)
    ratio = np.zeros(nsteps + 1)
    order = np.zeros(nsteps + 1)
    
    # Set initial values
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5  # Ideal value
    order[0] = get_order(lattice, nmax)
    
    # Perform Monte Carlo steps
    start_time = time.time()
    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
    runtime = time.time() - start_time
    
    # Final outputs
    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:.3f}, Order: {order[nsteps - 1]:.3f}, Time: {runtime:.6f} s")
    
    # Save data and plot the final frame
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
#=======================================================================
