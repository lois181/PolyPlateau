# code to run GROMACS MD simulation in 10ns blocks- check for equilibration after each block
# user input length of simulations, desired length of production region and maximum simulation time 

# following methods:

# 1. find an mdp file, topology file, index file and pre-existing .xtc file, raise error if not there
# 2. if xtc length is less than required equilibration time, run a continuation 
# 3. if xtc length is greater than required equilibration, check for equilibration and pass or fail 
# 4. keep a counter of total time for runs and warn user not equilibrated or stop if equilibrated and output plots + final data

import subprocess
from pathlib import Path
import os 
import glob
import red
import numpy as np

class detect_equilibration():

    NS_TO_PS = 1000 # conversion from ns to ps- not neccessary I just like using ns  

    @classmethod
    def ns_to_ps(cls, value_ns):
        return value_ns * cls.NS_TO_PS

    def __init__(self, max_time_ns, sim_len_ns, prod_len, workdir="."):

        self.max_time_ns = detect_equilibration.ns_to_ps(max_time_ns) # maximum total simulation time
        self.sim_len_ns = detect_equilibration.ns_to_ps(sim_len_ns) # length of each sub-simulation before checking equilibration
        self.prod_len = detect_equilibration.ns_to_ps(prod_len) # minimum required time in production region

        self.workdir = Path(workdir)

         
    def initialise_files(self):
        """Prepare files for first simulation"""

        extensions = ("mdp", "top", "ndx", "gro")
        self.files = {}

        for extension in extensions:

            files = list(self.workdir.glob(f"*.{extension}"))

            if not files:
                raise FileNotFoundError("Some files are missing. Required files are: *xtc, *tpr, *mdp, *top and *ndx")

            if len(files) > 1:
                print(f"More than one {extension} file detected. Using first one found.")

            self.files[extension] = files[0]

        # modify run-time of mdp file

        time_step = 0.001
        n_steps = round((self.sim_len_ns) / 0.001)

        with open(f"{self.files["mdp"]}", "r") as file:
            lines = file.readlines() # list of lines

        for index, line in enumerate(lines):
            if line.strip().startswith("nsteps"):
                lines[index] = f"nsteps = {n_steps}\n"
                break
        else: # execute when for loop finished
            raise ValueError("Missing nsteps. Check your mdp file.")

        with open(f"{self.files["mdp"]}", "w") as file:
            file.writelines(lines)



    def perform_grompp(self):
        """Create a tpr file to begin simulation. Uses default naming for GROMACS files."""

        output = "topol.tpr"

        command = [
            "gmx_mpi",
            "grompp",
            "-f", self.files["mdp"],
            "-c", self.files["gro"],
            "-p", self.files["top"],
            "-n", self.files["ndx"],
            "-o", output
        ]

        try:
            subprocess.run(
                command, 
                cwd=self.workdir,
                check=True # non-zero exit code is an error
            )
        except subprocess.CalledProcessError as error: 
            print(f"grompp failed with exit code {error.returncode}.")
            return False

        self.tpr = self.workdir / "topol.tpr"

        return True 

    def perform_mdrun(self, continuation=True):
        """Perform mdrun"""

        if continuation:
            command_tpr = [
                "gmx_mpi",
                "convert-tpr",
                "-s", self.tpr,
                "-extend", self.sim_len_ns,
                "-o", "extend.tpr"
            ] 

            try:
                subprocess.run(
                    command_tpr,
                    cwd=self.workdir,
                    check=True
                    )
            except subprocess.CalledProcessError as error: 
                print(f"convert-tpr failed with exit code {error.returncode}.")
                return False

            command_mdrun = [
                "gmx_mpi",
                "mdrun",
                "-s", "extend.tpr",
                "-cpi", "state.cpi"
            ] 

            try:
                subprocess.run(
                    command_mdrun,
                    cwd=self.workdir,
                    check=True
                    )
            except subprocess.CalledProcessError as error: 
                print(f"continuation mdrun failed with exit code {error.returncode}.")
                return False

        else:

            command_mdrun = [
                "gmx_mpi",
                "mdrun",
                "-s", self.tpr
            ]   

            try:
                subprocess.run(
                    command_mdrun,
                    cwd=self.workdir,
                    check=True
                    )
            except subprocess.CalledProcessError as error: 
                print(f"mdrun failed with exit code {error.returncode}.")
                return False

        return True 

    def gmx_polystat_energy(self):
        """Extract density, Rg and energy data.
          GROMACS saves polystat.xvg and energy.xvg to cwd. gmx polystat accepts 0th index 
          group."""

        # find .xtc 
        extensions = ("xtc","edr")

        for extension in extensions:

            files = list(self.workdir.glob(f"*.{extension}"))

            if not files:
                raise FileNotFoundError("Some files are missing. Required files are: *xtc, *tpr, *mdp, *top and *ndx")

            if len(files) > 1:
                print(f"More than one {extension} file detected. Using first one found.")

            self.files[extension] = files[0]


        command_polystat = [
            "gmx_mpi",
            "polystat",
            "-f", self.files["xtc"],
            "-s", self.tpr
        ]

        try:
            subprocess.run(
                command_polystat,
                cwd=self.workdir,
                check=True,
                input = "0\n",
            )
        except subprocess.CalledProcessError as error:
            print(f"gmx polystat failed with exit code {error.returncode}.")
            return False


        command_energy = [
            "gmx_mpi",
            "energy",
            "-f", self.files["edr"]
        ]

        energy_groups = ["Density", "Potential"]
        selected_groups = "\n".join(energy_groups) + "\n0\n" # 0 should select no more groups?

        try:
            subprocess.run(
                command_energy,
                cwd=self.workdir,
                input=selected_groups,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as error:
            print(f"gmx energy failed with exit code {error.returncode}.")
            return False

        return True 

    def check_equilibration_red(self):
        """Check (currently only) density for equilibration using RED library. 
        If self.max_time_ns exists beyond equilibration time, return True, else return False. 
        """

        # check if energy.xvg exists
        energy_file = self.workdir / "energy.xvg"

        if not energy_file.exists():
            raise FileNotFoundError("energy.xvg file not found.")

        # read energy.xvg file
        energy_data = np.loadtxt(energy_file, comments=['@', '#'])  # Skip header rows

        time = energy_data[:,0] # numpy slice all rows, first column
        density_data = energy_data[:,1] 

        eq_time, g, ess = red.detect_equilibration_window(
                density_data, method="min_sse", times=time,plot=True,data_y_label="Density",plot_name=f"equil_test_{time[-1]}.png", time_units="ps")

        if time[-1] - eq_time >= self.prod_len:
            return True # equilibration detected through minimum production length 
        else:
            return False 


class equilibration_workflow:

    def __init__(self, simulation):
        """Accepts instance of detect_equilibration"""

        self.simulation = simulation

    def run(self):

        self.simulation.perform_grompp() # grompp
        self.simulation.perform_mdrun(continuation=False)

        max_time = self.simulation.max_time_ns
        current_time = self.simulation.sim_time_ns
        prod_time = self.simulation.max_prod

        # should handle the case where they try a max time less than block size 


        while current_time <= max_time: 
            self.simulation.perform_mdrun(continuation=True)

            if current_time > prod_time:
                



        
