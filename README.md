# Lebwohl-Lasher Model Simulation

This repository contains implementations of the Lebwohl-Lasher model using various methods:
- NumPy Vectorization
- Cython (Serial and Parallel
- MPI (Message Passing Interface)

The code simulates a lattice model for liquid crystals with parameters such as iterations, size, temperature, and plot flags.

---

Code Structure
- Original Code:
  - LebwohlLasher_Origin.py  Called via command line with:        python numpy_vectorization.py <iterations> <size> <temperature> <plotflag>
- NumPy Implementation:
  - `Numpy_Vectorisation.py`: Called via command line with:
   python numpy_vectorization.py <iterations> <size> <temperature> <plotflag>
  
  - `numabs.py`: Called similarly via command line.

- Cython Implementation:
  - Serial:
    - `LebwohlLasher_cython.pyx`: Cython source file.
    - `cython_setup.py`: Setup file to compile the Cython code.
    - `LebwohlLasher_cython.py`: Python script to run the serial implementation.

  - Parallel:
    - `opti_parallel_cython.pyx`: Cython source file for parallel implementation.
    - `parallel_setup.py`: Setup file to compile the parallel Cython code.
    - `LebwohlLasher_parallel_cython.py`: Python script to run the parallel Cython implementation.

- MPI Implementation:
  - `MPI_Checkboard.py`: Run locally with:
    `mpiexec -n <num_processes> python MPI_Checkboard.py <iterations> <size> <temperature> <plotflag>`
  - On MPI vs Cython:
  -     compile with mpi_cython_setup.py. The pyx file is already present in file. To compile use: ` python mpi_cython_setup.py build_ext --inplace
      then Run locally with:
          `mpiexec -n <num_processes> python MPI_Cython.py <iterations> <size> <temperature> <plotflag>`
  - On Blue Crystal (using SLURM):
    - `slurm.sh`: SLURM submission script. Ensure `#SBATCH --ntasks-per-node=<num_processes>` matches the number of processes.
    - Change to the submission directory : 
      'cd /user/home/s/<>'
    - Submit the job with:
      `sbatch slurm.sh`
    - Check job status with:
      `sacct`
    - View output with:
      `nano output.txt`

---

Requires
- Python 3.x
- NumPy
- Cython
- MPI (e.g., OpenMPI or MPICH)
- SLURM (for Blue Crystal)

Compiling Cython Code
1. Navigate to the directory containing the Cython files.
2. Compile the serial Cython code:
   ` python cython_setup.py build_ext --inplace
