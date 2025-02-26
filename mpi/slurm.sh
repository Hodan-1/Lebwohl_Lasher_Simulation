#!/bin/bash
# =================
# run_slurm.sh
# =================

#SBATCH --job-name=mpi_test_job       # Job name
#SBATCH --partition=teach_cpu         # Partition (queue) to use
#SBATCH --account=PHYS033186          # Account to charge resou$
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=28          # Number of MPI tasks per$#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --time=0:10:00                # Maximum runtime (HH:MM:$#SBATCH --mem-per-cpu=100M            # Memory per CPU core
#SBATCH --output=mpi_slurm_%j.sh    # Output file (%j will be r$#SBATCH --error=mpi_error_%j.log      # Error file (%j will be $

# Load required modules
module purge
module load languages/python/3.12.3
module load openmpi/4.1.6-mzfg

# Change to the submission directory
cd /user/home/< >

# Run the MPI program

mpiexec -n 28 python3 MPI_Checkboard.py 80 80 0.5 0
