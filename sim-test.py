from importlib.resources import contents
from operator import index
import subprocess
import shlex
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
import matplotlib.pyplot as plt

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


    def __init__(self, max_time_ns:float, sim_len_ns:float, prod_len:float, equil_properties: list[str], g_ver:str="gmx", mdrun_preamble: str="", workdir=".", confidence:float=95):

        self.max_time_ns = self.ns_to_ps(max_time_ns) # maximum total simulation time
        self.sim_len_ns = self.ns_to_ps(sim_len_ns) # length of each sub-simulation before checking equilibration
        self.prod_len = self.ns_to_ps(prod_len) # minimum required time in production region
        self.workdir = Path(workdir).resolve()
        self.g_ver = g_ver 
        self.mdrun_preamble = mdrun_preamble
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

        with open(self.files["mdp"], "r") as file:
            lines = file.readlines() # list of lines

        # read time step from mdp file
        for index, line in enumerate(lines):
            if line.strip().startswith("dt"):
                parts = line.split(";", 1)[0].split("=", 1)
                if len(parts) == 2:
                    try:
                        time_step = float(parts[1].strip())
                        n_steps = round((self.sim_len_ns) / time_step)
                    except ValueError:
                        raise ValueError(f"Invalid dt value in mdp file: {parts[1].strip()}")
                break

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
            self.g_ver,
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

        preamble = [
            os.path.expandvars(arg)
            for arg in shlex.split(self.mdrun_preamble)
        ]

        if continuation:
            next_number = self.tpr_number + 1
            next_tpr = self.workdir / f"topol{next_number}.tpr"
            command_tpr = [
                self.g_ver,
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

            command_mdrun = preamble + [
                self.g_ver,
                "mdrun",
                "-s", next_tpr,
                "-cpi", "state.cpt",
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

            command_mdrun = preamble + [
                self.g_ver,
                "mdrun",
                "-s", self.tpr,
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
                self.g_ver,
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
                self.g_ver,
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

            np.savetxt(self.workdir / "equil_rg.csv", rg_data[:, [0, 2]], delimiter=",", header="time,rg", comments="")

        if "density" in self.equil_properties or "potential" in self.equil_properties:
            energy_file = self.workdir / "energy.xvg"
            if not energy_file.exists():
                raise FileNotFoundError(
                    "energy.xvg file not found. Something went wrong with gmx energy."
                )

            energy_data = np.loadtxt(energy_file, comments=["@", "#"], ndmin=2)

            for index, prop in enumerate(energy_groups, start=1):
                prop = prop.lower()

                np.savetxt(self.workdir / f"equil_{prop}.csv", energy_data[:, [0, index]], delimiter=",", header=f"time,{prop}", comments="")


        return True 

    def check_equilibration_red(self):
        """Return the common production start, or None if insufficient data remains."""

        equil_results = {}
        property_times = {}

        for prop in self.equil_properties:
            equil_file = self.workdir / f"equil_{prop}.csv"

            if not equil_file.exists():
                raise FileNotFoundError(f"equil_{prop}.csv file not found.")

            equil_data = pd.read_csv(equil_file)
            equil_data.columns = equil_data.columns.str.lower()

            time = equil_data["time"].to_numpy()
            prop_data = equil_data[prop].to_numpy()
            property_times[prop] = time

            eq_time, g, ess = red.detect_equilibration_window(
                prop_data,
                method="min_sse",
                times=time,
                plot=False,
                data_y_label=prop,
                plot_name=str(self.workdir / f"equil_test_{prop}_{time[-1]}.png"),
                time_units="ps",
            )

            equil_results[prop] = {
                "eq_time": eq_time,
                "g": g,
                "ess": ess,
                "time": time,
                "values": prop_data,
            }

        eq_time = max(
            equil_results[prop]["eq_time"]
            for prop in self.equil_properties
        )

        failures = {}

        for prop in self.equil_properties:
            production_times = property_times[prop][
                property_times[prop] >= eq_time
            ]

            if len(production_times) < 2:
                production_duration = 0.0
            else:
                production_duration = (
                    production_times[-1] - production_times[0]
                )

            if len(production_times) < 2 or production_duration < self.prod_len:
                failures[prop] = {
                    "reason": "Production too short",
                    "eq_time": equil_results[prop]["eq_time"],
                    "production_start": eq_time,
                    "production_duration": production_duration,
                    "required_duration": self.prod_len,
                }

        # Retain every property's RED results for plotting
        self.equil_results = equil_results

        if failures:
            return failures

        return eq_time

        # eq_time = max(
        #     equil_results[prop]["eq_time"]
        #     for prop in self.equil_properties
        # )

        # for prop in self.equil_properties:
        #     production_times = property_times[prop][
        #         property_times[prop] >= eq_time
        #     ]

        #     if len(production_times) < 2:
        #         return None

        #     if production_times[-1] - production_times[0] < self.prod_len:
        #         return None

        # return eq_time

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

            if len(whitened) < 10:
                raise ValueError(f"Column {column}: Fewer than 10 valid data points after pre-whitening.")

            print(f"Analysing {len(whitened)} valid data points for property number {column} after pre-whitening...")

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

    def plot_equilibration(self, prop, time, values, eq_time, reason):

        results_dir = self.simulation.workdir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        time_ns = time / 1000

        fig, ax = plt.subplots()

        ax.plot(time_ns, values, linewidth=0.8)
        ax.axvline(
            eq_time / 1000,
            color="red",
            linestyle=":",
            label=f"RED equilibration: {eq_time / 1000:g} ns",
        )

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(prop)
        ax.set_title(
            f"{prop}: {reason}\n"
            f"{time_ns[0]:g}–{time_ns[-1]:g} ns"
        )
        ax.legend()

        filename = f"{prop}_{time_ns[0]:g}-{time_ns[-1]:g}ns.png"

        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=200)
        plt.close(fig)

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

                red_result = self.simulation.check_equilibration_red()

                if isinstance(red_result, dict):
                    failures = red_result
                    eq_time = max(
                        result["eq_time"]
                        for result in self.simulation.equil_results.values()
                    )
                else:
                    failures = {}
                    eq_time = red_result
                    print("Minimum production length detected. Running Mann-Kendall test...")

                property_results = {}

                for prop in self.simulation.equil_properties:
                    equil_file = self.simulation.workdir / f"equil_{prop}.csv"
                    equil_data = pd.read_csv(equil_file)
                    equil_data.columns = equil_data.columns.str.lower()

                    if prop in failures:
                        failure = failures[prop]
                        passed = False
                        reason = (
                            "FAIL — production too short "
                            f"({failure['production_duration'] / 1000:g} ns; "
                            f"required {failure['required_duration'] / 1000:g} ns)"
                        )
                    else:
                        production_data = equil_data.loc[
                            equil_data["time"] >= eq_time,
                            ["time", prop]
                        ]

                        print(f"Running Mann-Kendall test for {prop}...")

                        data_array = production_data.to_numpy()
                        has_trend = self.simulation.perform_mann_kendall_test(
                            data_array
                        )

                        passed = not has_trend

                        if has_trend:
                            reason = "FAIL — significant trend detected by MK"
                        else:
                            reason = "PASS"

                    property_results[prop] = {
                        "passed": passed,
                        "reason": reason,
                    }

                    self.plot_equilibration(
                        prop,
                        equil_data["time"].to_numpy(),
                        equil_data[prop].to_numpy(),
                        self.simulation.equil_results[prop]["eq_time"],
                        reason,
                    )

                self.property_results = property_results

                if all(result["passed"] for result in property_results.values()):
                    print("All properties passed. System is equilibrated.")
                    return True
                else:
                    print("One or more properties failed. Continuing simulation...")
                    continue
            else:
                print("Simulation not long enough to encompass required minimum production region. Continuing...")
                continue
        else:
            print("Maximum simulation time reached or exceeded. System is not equilibrated.")
            return False


# example usage:
# Code performs simulations in increments of sim_len_ns. After each simulation it checks for the 
# minimum long run variance equilibration time using the RED package and requires a production period
# of at least prod_len. If production period is too short, simulations are continued. If long enough, it
# performs a modified version of the Mann-Kendall test with confidence level (confidence) of default
# 95%. If no evidence of trend is found, halt simulation. 

# For multiple properties (supported: density, rg, potential), the maximum equilibration time is used as 
# the common production start. If any property fails, simulation continues until either all properties pass or max_time_ns is reached.

# Assumed version of gromacs is non-mpi. Options below to specify gmx_mpi and also mdrun preamble e.g. mpirun

# Code searches for .top, .mdp, .ndx and .gro files in working directory. I'm assuming use of default file names for now and a 
# simulation that isn't a continuation of a pre-exisiting simulation (it relies on appending to the current xtc on every
# simulation extension which is problematic if you've previously moved around files in GROMACS).

polymer = equilibrate_polymer(
    max_time_ns=50,
    sim_len_ns=10,
    prod_len=10,
    equil_properties=["density", "potential"],
    g_ver="gmx_mpi",
    mdrun_preamble="mpirun",
    confidence=99,
    workdir=".",
)

equilibrate = equilibration_workflow(polymer)
equilibrate.run()
        
