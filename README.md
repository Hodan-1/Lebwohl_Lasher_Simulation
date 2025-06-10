# Lebwohl-Lasher Model Simulation

## Overview
This project implements the Lebwohl-Lasher model for liquid crystal simulation using various optimisation techniques. It provides multiple implementations to compare performance and scalability across different computing approaches, from basic Python to parallel processing with MPI.

## Key Features
- **Multiple Implementation Methods**:
  - Base Python implementation
  - NumPy vectorisation and numbas optimisarion for improved performance
  - Cython optimisation (both serial and parallel)
  - MPI for distributed computing
  - Hybrid MPI-Cython implementation
- **Performance Analysis Tools**
- **Scalable Processing**
- **HPC Support** (Blue Crystal compatibility)

---
## Project Structure
```
Lebwoh_Lasher_Simulation/
├── base/
│   ├── LebwohlLasher_Origin.py    # Original implementation
│   └── performance_analysis.ipynb     # Performance analysis
├── optimised/
│   ├── numpy_vectorisation.py  # NumPy optimised version
│   └── numba_version.py        # Numba implemntation
├── serial/                     # Serial Cython implementations
│   ├── LebwohlLasher_cython.py
│   ├── LebwohlLasher_cython.pyx
│   └── cython_setup.py
├── parallel/                       # Parallel Cython implementations
│   ├── LebwohlLasher_parallel_cython.py
│   ├── opti_parallel_cython.pyx
│   └── parallel_setup.py
└── mpi/                         #MPI implementations
│   ├── MPI_Checkerboard.py
│   └── slurm.sh
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- [Python](https://www.python.org/downloads/) (3.7 or higher)
- [NumPy](https://numpy.org/install/)
- [Cython](https://cython.org/#download)
- [MPI](https://www.open-mpi.org/software/ompi/v4.1/) (OpenMPI or MPICH)
- [SLURM](https://slurm.schedmd.com/download.html) (for HPC usage)

---

### Installation

1. Clone the Repository:
```bash
git clone <repository-url>
cd Lebwohl_Lasher_Simulation
```

2. Install Dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Base Implementation
```bash
python base/LebwohlLasher_Origin.py <iterations> <size> <temperature> <plotflag>
```

### 2. NumPy Vectorized or Numba Version
```bash
python optimised/numpy_vectorisation.py <iterations> <size> <temperature> <plotflag>
```
Note: Replace `numpy_vectorisation.py` with `numba_version.py` for the Numba implementation.

### 3. Cython Implementations

#### Serial Version
```bash
cd serial_cython_optimisation
python cython_setup.py build_ext --inplace
python LebwohlLasher_cython.py <iterations> <size> <temperature> <plotflag>
```

#### Parallel Version
```bash
cd parallel_cython_optimisation
python parallel_setup.py build_ext --inplace
python LebwohlLasher_parallel_cython.py <iterations> <size> <temperature> <plotflag>
```

### 4. MPI Implementation
```bash
mpiexec -n <num_processes> python mpi/MPI_Checkerboard.py <iterations> <size> <temperature> <plotflag>
```

### Running on HPC (Blue Crystal)

1. Navigate to SLURM job directory:
```bash
cd HPC
```

2. Submit the job:
```bash
sbatch src/mpi/slurm.sh
```

3. Monitor and view results:

```bash
sacct                # Check job status
nano output.txt      # View job output
```

Make sure to update `#SBATCH --ntasks-per-node=` to match your process count.

## Performance Analysis
The performance analysis notebook (`base/performance_analysis.ipynb`) contains:
- Execution time comparisons
- Speedup calculations
- Scaling analysis across implementations

---

## Parameter Description
- `iterations`: Number of Monte Carlo steps (e.g., 10000)
- `size`: Lattice size (NxN) (e.g., 50 creates a 50x50 lattice)
- `temperature`: Simulation temperature (e.g., 0.5)
- `plotflag`: Enable/disable visualisation (1 for plotting, 0 for no plots)

### Basic Example
```bash
python base/LebwohlLasher_Origin.py 1000 50 0.5 1
```
This command will:

- Run 1000 Monte Carlo steps
- Use a 50x50 lattice
- Set temperature to 0.5
- Show visualisation (plotflag=1)

### MPI Example

``` bash
mpiexec -n 4 python mpi/MPI_Checkerboard.py 5000 80 0.4 0
```
This command will:
- Use 4 MPI processes
- Run 5000 Monte Carlo steps
- Use an 80x80 lattice
- Set temperature to 0.4
- Run without visualisation


---
## Troubleshooting

### Common Issues
1. **MPI Installation**
   - Error: "mpiexec command not found"
   - Solution: Ensure MPI is properly installed and added to PATH
   - Remember always match the number of processes with available CPU cores

2. **Cython Compilation**
   - Error: "Unable to find vcvarsall.bat"
   - Solution: Install appropriate C compiler (e.g., Microsoft Visual C++ for Windows)


