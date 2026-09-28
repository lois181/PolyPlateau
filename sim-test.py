# code to run GROMACS MD simulation in 10ns blocks- check for equilibration after each block
# user input length of simulations, desired length of production region and maximum simulation time 

# following methods:

# 1. find an mdp file, topology file, index file and pre-existing .xtc file, raise error if not there
# 2. if xtc length is less than required equilibration time, run a continuation 
# 3. if xtc length is greater than required equilibration, check for equilibration and pass or fail 
# 4. keep a counter of total time for runs and warn user not equilibrated or stop if equilibrated and output plots + final data