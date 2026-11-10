# PolyPlateau

## Introduction

PolyPlateau is an automated GROMACS simulation procedure, designed for polymers, which monitors the behaviour of key system properties (currently supported: potential energy, density and radius of gyration) over time. By determining when the monitored properties reach a plateau of user-specified minimum duration, PolyPlateau identifies a time ($t_0$), the data after which forms a production period which can be used for accurate property extraction.

## Installation 

PolyPlateau requires the GROMACS molecular dynamics package. This code was tested on version 2024.6. 

From the project directory, run the following command to install the required Python packages:

```bash
python -m pip install -r requirements.txt
```
Alternatively, install the following packages into your own environment:

### Python dependencies

| Package | Version |
| --- | --- |
| `numpy` | `2.2.6` |
| `pandas` | `2.3.3` |
| `matplotlib` | `3.10.7` |
| `red-molsim` | `0.2.0` |
| `pymannkendall` | `1.4.3` |

## Usage and Example

### Concept 

The workflow is summarised in the chart below:

![PolyPlateau flowchart](./polyplateau-workflow.png)

#### Step 1

PolyPlateau relies on user-created pre-existing GROMACS files (.mdp, .top, .gro). If `use_index=True`, the first .ndx file in the working directory will be used for all analysis, selecting the first index group as default. If `use_index=False`, a new .ndx file will be created using the user-provided .gro file. 

Input files should be provided according to the system of interest to the user, although since PolyPlateau searches for the absence of trends/bias in the data, some settings should be avoided e.g. temperature annealing. 

#### Step 2

Provided the current elapsed simulation time is less than `max_time_ns`, which is the user-provided maximum simulation time, a GROMACS simulation of length `sim_len_ns` will be performed. This continues until the elapsed simulation time is at least the minimum production time set by the user, `prod_len`.

#### Step 3

GROMACS is used to extract the properties PolyPlateau tracks. These are currently: potential energy, density and radius of gyration. If more than one property is selected, all of them must pass production period checks before the procedure halts. 

#### Step 4

To calculate an initial guess of $t_0$, we implement the methodology as seen in the following work: Clark, F.; Cole, D. J.; Michel, J. Robust Automated Truncation Point Selection for Molecular Simulations. J. Chem. Theory Comput. 2024. https://doi.org/10.1021/acs.jctc.4c01359.

This calculates the minimum squared standard error (SSE) for different choices of $t_0$ using a Bartlett window method. We use the recommended window size of $\sqrt{N}$, where $N$ is the number of data points. The value of $t_0$ which minimises the SSE balances minimising the bias in the data with maximising the size of the identified production region. For more information, consult the work by Clark et al. which includes thorough testing on simulation data.

#### Step 5

If the length of the production period identified is greater than `prod_len`, we verify that it does contain a significant trend using the Mann-Kendall statistic p-value. To account for the potentially high autocorrelation in the data, we use the Hamed-Rao variance correction which has shown to be effective in these cases. (https://pmc.ncbi.nlm.nih.gov/articles/PMC12775442/) The procedure is halted only when the production period of each property satisfies both the minimum duration requirement and also the Mann-Kendall test.

Since different system properties relax over different timescales, the output is the maximum value of $t_0$ across all properties tested, along with individual values of $t_0$ per property. 

### Usage 

## Example usage

Run this example from the directory containing your GROMACS `.gro`, `.top`,
and `.mdp` files. The first four arguments are required. The optional
arguments are shown with their defaults.

```python
from polyplateau import equilibrate_polymer, equilibration_workflow

simulation = equilibrate_polymer(
    max_time_ns=50,                            # Time limit in ns
    sim_len_ns=10,                             # Length of each segment in ns
    prod_len=10,                               # Minimum production window in ns
    equil_properties=["density", "potential", "rg"], # Properties to monitor

    g_ver="gmx",                               # GROMACS executable i.e. gmx or gmx_mpi
    mdrun_preamble="",                         # Optional launcher, e.g. "mpirun"
    workdir=".",                               # Directory containing the input files
    confidence=99,                             # Mann–Kendall confidence level (%)
    use_index=False,                           # Generate an index from the .gro file
)

workflow = equilibration_workflow(simulation)
equilibrated = workflow.run()
```

`workflow.run()` returns `True` when all selected properties pass the
equilibration checks. Assessment plots and the CSV summary are saved in
`results/` inside `workdir`.



