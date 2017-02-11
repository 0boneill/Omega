#!/bin/tcsh 
#
#SBATCH --job-name d21
#SBATCH --account=FY150001
#SBATCH -N 12
#SBATCH --time=0:10:00
#SBATCH -p ec


set OMP_NUM_THREADS = 1
set NCPU = 40 
if ( ${?SLURM_NNODES} ) then   # redsky
    set NCPU = $SLURM_NNODES
    @ NCPU *= 16
    @ NCPU /= $OMP_NUM_THREADS
endif

# NH model
#set EXEC = ../../../test_execs/theta-nlev60/theta-nlev60    
#set namelist = namelist-nh-default.nl


# hydrostatic theta
set EXEC = ../../../test_execs/theta-nlev60/theta-nlev60    
set namelist = namelist-default.nl


# hydrostatic preqx
#set EXEC = ../../../test_execs/preqx-nlev60-interp/preqx-nlev60-interp        # set name of executable
#set namelist = namelist-default.nl

\cp -f $namelist input.nl
date
mpirun -np $NCPU $EXEC < input.nl
date
ncl plot_lon_vs_z.ncl
