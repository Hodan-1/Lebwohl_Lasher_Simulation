from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from monte_carlo import compute_energy, mc_step

# Initialise MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def plotdat(arr, pflag, nmax):
    """
    Plot the lattice data using quiver plots.
    """
    if pflag == 0:
        return

    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros_like(arr)

    if pflag == 1:  # Colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
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

def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    """
    Save simulation data to a file.
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

def main(nsteps, nmax, temp, pflag):
    # Initialise the lattice
    sub_nmax = nmax // size
    lattice = np.random.uniform(0, 2 * np.pi, (sub_nmax, nmax))

    # Monte Carlo loop
    for it in range(nsteps):
        accept = mc_step(lattice, temp)
        # Exchange boundaries (pure Python + MPI)
        exchange_boundaries(lattice)
        # Compute total energy (Cython-optimized)
        energy = compute_energy(lattice)
        # MPI reduction (pure Python + MPI)
        total_energy = comm.allreduce(energy, op=MPI.SUM)

    if rank == 0:
        print(f"Total energy: {total_energy}")
        plotdat(lattice, pflag, nmax)
        savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        nsteps = int(sys.argv[1])
        nmax = int(sys.argv[2])
        temp = float(sys.argv[3])
        pflag = int(sys.argv[4])
        main(nsteps, nmax, temp, pflag)
    else:
        print(f"Usage: mpirun -np <num_processes> python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")
