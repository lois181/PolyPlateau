# code to run GROMACS MD simulation in 10ns blocks- check for equilibration after each block
# user input length of simulations, desired length of production region and maximum simulation time 

# following methods:

# 1. find an mdp file, topology file, index file and pre-existing .xtc file, raise error if not there
# 2. if xtc length is less than required equilibration time, run a continuation 
# 3. if xtc length is greater than required equilibration, check for equilibration and pass or fail 
# 4. keep a counter of total time for runs and warn user not equilibrated or stop if equilibrated and output plots + final data

# To-do for my own quality of life:
# Options to specify specific command "gmx_mpi", "mpirun gmx_mpi -nt" etc.
# If index file not found, make one 
# Default mdp file 
# On-the-fly version? (larger work)
# Run using a pre-exisisting state file from another simulation - use current logic in new run with -noappend flag 
# Print out plot regardless of equilibration outcome
# Also print raw data 
# option to provide .xtc names 
# add option for full mdrun command in case users have tables 

# plan:
# 1. property calculation should be done based on property selected and then results written to a monitoring csv which can then
# be read and each property in there can be tracked. The tracking will have a single version and multiple version

from importlib.resources import contents
from operator import index
import subprocess
from pathlib import Path
import os 
import glob
from tokenize import group
import red
import numpy as np
import pandas as pd 
from datetime import datetime, timedelta
from mannkendall import mk_white
from scipy.stats import kendalltau
import warnings 

class equilibrate_polymer():

    NS_TO_PS = 1000 # conversion from ns to ps- not neccessary I just like using ns  

    @classmethod
    def ns_to_ps(cls, value):
        try:
            return float(value) * 1000
        except (TypeError, ValueError):
            raise ValueError(f"{value!r} is not a valid number")

    @classmethod
    def confidence_to_threshold(cls, conf):
        """Convert a confidence level (%) to a significance threshold."""
        try:
            in_range = 1 <= conf <= 100 # boolean true or false 
        except TypeError:
            raise ValueError(f"{conf!r} is not a valid number") from None
        if not in_range:
            raise ValueError("The confidence level must be between 1% and 100%. We strongly recommend > 90%.")
        return np.round(1 - conf / 100, 3)


    def __init__(self, max_time_ns:float, sim_len_ns:float, prod_len:float, equil_properties: list[str], g_command:str="gmx", workdir=".", confidence:float=95):

        self.max_time_ns = self.ns_to_ps(max_time_ns) # maximum total simulation time
        self.sim_len_ns = self.ns_to_ps(sim_len_ns) # length of each sub-simulation before checking equilibration
        self.prod_len = self.ns_to_ps(prod_len) # minimum required time in production region
        self.workdir = Path(workdir).resolve()
        self.g_command = g_command 
        self.confidence = self.confidence_to_threshold(confidence)
        self.equil_properties = equil_properties

        if self.prod_len >= self.max_time_ns:
            raise ValueError("You must specify a minimum production time less than the maximum allowed simulation length.")

        if not isinstance(equil_properties, list): # safety checks below for providing equilibration properties
            raise TypeError("Equilibration properties must be lists of strings")

        if equil_properties is None or len(equil_properties) == 0:
            raise ValueError("Specify at least one property to monitor for equilibration.")

        allowed = {"density", "rg", "potential"} # currently supported equilibration options

        for prop in equil_properties:
            if prop not in allowed:
                raise ValueError(f"Invalid parameter. Choose from: {allowed}")

         
    def initialise_files(self):
        """Prepare files for first simulation"""

        extensions = ("mdp", "top", "ndx", "gro")
        self.files = {}

        for extension in extensions:

            files = list(self.workdir.glob(f"*.{extension}"))

            if not files:
                raise FileNotFoundError("Some files are missing. Required files are: *tpr, *mdp, *top and *ndx")

            if len(files) > 1:
                print(f"More than one {extension} file detected. Using first one found.")

            self.files[extension] = files[0]

        # modify run-time of mdp file

        time_step = 0.001
        n_steps = round((self.sim_len_ns) / 0.001)

        with open(self.files["mdp"], "r") as file:
            lines = file.readlines() # list of lines

        for index, line in enumerate(lines):
            if line.strip().startswith("nsteps"):
                lines[index] = f"nsteps = {n_steps}\n"
                break
        else: # execute when for loop finished
            raise ValueError("Missing nsteps. Check your mdp file.")

        with open(self.files["mdp"], "w") as file:
            file.writelines(lines)



    def perform_grompp(self):
        """Create a tpr file to begin simulation. Uses default naming for GROMACS files."""

        output = "topol.tpr"

        command = [
            "gmx",
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

        self.tpr = self.workdir / output
        self.tpr_number = 0 # keep track of tpr files for continuations 

        return True 

    def perform_mdrun(self, continuation=True):
        """Perform mdrun"""

        if continuation:
            next_number = self.tpr_number + 1
            next_tpr = self.workdir / f"topol{next_number}.tpr"
            command_tpr = [
                "gmx",
                "convert-tpr",
                "-s", self.tpr,
                "-extend", str(self.sim_len_ns),
                "-o", next_tpr
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
                self.g_command,
                "mdrun",
                "-s", next_tpr,
                "-cpi", "state.cpt",
                "-nt", str(8)
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

            self.tpr = next_tpr
            self.tpr_number = next_number

        else:

            command_mdrun = [
                self.g_command,
                "mdrun",
                "-s", self.tpr,
                "-nt", str(8)
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

        if "rg" in self.equil_properties:

            command_polystat = [
                "gmx",
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
                    text = True,
                )
            except subprocess.CalledProcessError as error:
                print(f"gmx polystat failed with exit code {error.returncode}.")
                return False


        energy_groups = []

        if "density" in self.equil_properties:
            energy_groups.append("Density")

        if "potential" in self.equil_properties:
            energy_groups.append("Potential")

        
        if energy_groups:

            command_energy = [
                "gmx",
                "energy",
                "-f", self.files["edr"]
            ]

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

        # add results to csv 

        columns = []
        headers = ["time"]

        if "rg" in self.equil_properties:
            rg_file = self.workdir / "polystat.xvg"
            if not rg_file.exists():
                raise FileNotFoundError(
                    "polystat.xvg file not found. Something went wrong with gmx polystat."
                )

            rg_data = np.loadtxt(rg_file, comments=["@", "#"], ndmin=2)

            columns.append(rg_data[:, 0])
            columns.append(rg_data[:, 2])
            headers.append("rg")

        if "density" in self.equil_properties or "potential" in self.equil_properties:
            energy_file = self.workdir / "energy.xvg"
            if not energy_file.exists():
                raise FileNotFoundError(
                    "energy.xvg file not found. Something went wrong with gmx energy."
                )

            energy_data = np.loadtxt(energy_file, comments=["@", "#"], ndmin=2)

            if not columns:
                columns.append(energy_data[:, 0])

            for index, prop in enumerate(energy_groups, start=1):
                columns.append(energy_data[:, index])
                headers.append(prop)

        with open(self.workdir / "equil_properties.csv", "w") as file:
            np.savetxt(
                file,
                np.column_stack(columns),
                delimiter=",",
                header=",".join(headers),
                comments="",
            )

        return True 

    def check_equilibration_red(self):
        """Check (currently only) density for equilibration using RED library. 
        If self.max_time_ns exists beyond equilibration time, return True, else return False. 
        """

        equil_file = self.workdir / "equil_properties.csv"

        if not equil_file.exists():
            raise FileNotFoundError("equil_properties.csv file not found.")

        equil_data = pd.read_csv(equil_file)

        # Match lowercase property names to CSV headers such as "Density".
        equil_data.columns = equil_data.columns.str.lower()

        time = equil_data["time"].to_numpy()
        equil_results = {}

        for prop in self.equil_properties:
            prop_data = equil_data[prop].to_numpy()

            eq_time, g, ess = red.detect_equilibration_window(
                prop_data,
                method="min_sse",
                times=time,
                plot=True,
                data_y_label=prop,
                plot_name=str(self.workdir / f"equil_test_{prop}_{time[-1]}.png"),
                time_units="ps",
            )

            equil_results[prop] = {
                "eq_time": eq_time,
                "g": g,
                "ess": ess,
            }

        eq_time = max(
            equil_results[prop]["eq_time"]
            for prop in self.equil_properties
        )

        if time[-1] - eq_time >= self.prod_len:
            return eq_time

        return None

    def perform_mann_kendall_test(self, data):
        """Input: data array (1,).
        Performs a Mann-Kendall test with trend-free pre-whitening according to Wang & Swail (2008).
        Returns: MK p-value. Less than the confidence threshold rejects the null hypothesis (finds evidence of a trend)"""

        # data is passed as an N-dimensional array depending on number of properties being monitored. MK test is performed on each property separately.

        # pre-whiten data first

        data = np.asarray(data, dtype=float)

        times = data[:, 0] # first col is time 

        if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0): # safety check for something in the future 
            raise ValueError("Time values must be finite and increasing. The problem may be with the GROMACS extension handling.")

        date_list = []

        # following is because mannkendall requires datetime objects for time series, so we convert with even intervals 
        for t in (times-times[0]) / 1000: # elapsed time in ns
            offset = timedelta(days=float(t)) # number of extra days
            date = datetime(2000, 1, 1) + offset # convert to datetime starting January 1st 2000
            date_list.append(date)

        dates = np.array(date_list, dtype=object) # store new times as datetime objects 

        results = {}

        for column in range(1, data.shape[1]):
            values = data[:, column]
            valid = np.isfinite(values)

            if not np.all(valid):
                warnings.warn("Encountered non-finite values in data. These will be ignored.", RuntimeWarning, stacklevel=2)

            if np.count_nonzero(valid) < 10:
                raise ValueError("Not enough data for MK test.")

            whitened = mk_white.prewhite(
                values[valid].copy(),
                dates[valid],
                resolution=1e-4, # for ties
                alpha_ak=95, # confidence level for detecting autocorrelation
            )["tfpw_ws"]

            whitened = np.asarray(whitened, dtype=float)
            valid = np.isfinite(whitened)

            if np.count_nonzero(~valid) > 0:
                warnings.warn(
                    f"Column {column}: discarded {np.count_nonzero(~valid)} of {len(whitened)} "
                    "points after pre-whitening because they were NaN or infinite.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            whitened = whitened[valid]

            if np.all(whitened == whitened[0]): # if all values are the same 
                raise ValueError("All data values are identical. Something has gone wrong.")
            else:
                tau, p_value = kendalltau(
                    np.arange(len(whitened)), whitened, method="auto"
                )

            if not np.isfinite(tau) or not np.isfinite(p_value):
                raise ValueError(f"Column {column}: MK test returned an undefined result.")

            results[column] = {
                "tau": float(tau),
                "p": float(p_value),
                "significant": bool(p_value <= self.confidence),
            }

        for column, result in results.items():
            print(
                f"Column {column}: tau={result['tau']:.3f}, "
                f"p={result['p']:.3f}, significant={result['significant']}"
            )

        return any(result["significant"] for result in results.values())        
            

class equilibration_workflow:

    def __init__(self, simulation:equilibrate_polymer, new_sim = True):
        """Accepts instance of detect_equilibration"""

        self.simulation = simulation

    def run(self):

        self.simulation.initialise_files()
        if not self.simulation.perform_grompp(): # grompp
            return False

        if not self.simulation.perform_mdrun(continuation=False):
            return False

        max_time = self.simulation.max_time_ns
        current_time = self.simulation.sim_len_ns
        prod_time = self.simulation.prod_len

        while current_time < max_time:
            if not self.simulation.perform_mdrun(continuation=True):
                return False

            current_time += self.simulation.sim_len_ns

            if current_time >= prod_time:
                if not self.simulation.gmx_polystat_energy(): # extract data
                    return False

                eq_time = self.simulation.check_equilibration_red()

                if eq_time is not None:
                    print("Minimum production length detected. Running Mann-Kendall test...")

                    equil_file = self.simulation.workdir / "equil_properties.csv"
                    equil_data = pd.read_csv(equil_file)
                    equil_data.columns = equil_data.columns.str.lower()

                    # Only test the production region and selected properties
                    production_data = equil_data.loc[
                        equil_data["time"] >= eq_time,
                        ["time", *self.simulation.equil_properties]
                    ]

                    data_array = production_data.to_numpy()
                    mk_results = self.simulation.perform_mann_kendall_test(data_array)

                    if mk_results:
                        print("Mann-Kendall test indicates a significant trend. System is not equilibrated.")
                        continue
                    else:
                        print("Mann-Kendall test does not indicate a significant trend. System is equilibrated.")
                        return True
                else:
                    print("Equilibration not detected. Continuing simulation...")
                    continue
            else:
                print("Simulation not long enough to encompass required minimum production region. Continuing...")
                continue
        else:
            print("Maximum simulation time reached or exceeded. System is not equilibrated.")
            return False

        # start current time 10 max time 30
        # current time less than max time so perform continuation mdrun
        # current time greater than production time, check for equilibration and return true if true 
        # if false, continue
        # if max time exceeded return false 

        # if current time is less than minimum required production time, we should continue through loop


# create instance of equilibrate_polymer 


polymer = equilibrate_polymer(max_time_ns=0.6, sim_len_ns=0.1, prod_len=0.2)

equilibrate = equilibration_workflow(polymer)

equilibrate.run()



        
