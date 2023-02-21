# Production Simulation and Analyses

<img alt="GitHub" src="https://img.shields.io/github/license/alexmuetze/Production-Simulation-and-Analyses"> <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/alexmuetze/Production-Simulation-and-Analyses">

The tool Production Simulation and Analyses is a simulation library, which enables the user to simulate and evaluate an arbitrarily configurable production area. The focus of the simulation library is on production planning and control (PPC) and in particular the tasks: Throughput planning, order release, capacity control and dispatching.
The model contains a number of well-known PPC heuristics for these tasks, each of which can be freely parameterized.

In addition to the flow simulation, which is realized as a discrete event simulation using the Python framework *SimPy* (https://simpy.readthedocs.io/en/latest/), the simulation library offers the user a variety of possible data exports and integrated analyses, which are provided using *Matplotlib* (https://matplotlib.org/) and *seaborn* (https://seaborn.pydata.org/).

Futhermore, the model includes an experimental layer for parallel and sequential simulation using *SLURM Workload Manager* (https://slurm.schedmd.com/).

## Install the Required Dependencies
Production Simulation and Analyses is completely based on Python. The packages needed for the simulation are documented in `requirements.txt`.
To give an overview, the required packages are listed here as well.

| Package | Version |
| --: | --: |
| `numpy` | 1.19.1 |
| `pandas` | 1.1.0 |
| `simpy` | 4.0.1 |
| `scipy` | 1.6.0 |
| `matplotlib` | 3.6.0 |
| `seaborn` | 0.12.1 |


## Usage
The simulation library is mainly configured via the classes *ModelPanel* and *PolicyPanel*, which are included in `control_panel.py`. While *ModelPanel* contains, in particular, more basic structural configuration decisions of the production area to be considered, *PolicyPanel* contains the concrete configuration of the individual PPC tasks. In dependence on the use of the so-called *batch_manager*, the settings which can be varied can be specified by a parameter dictionary, which is iterated by the `exp_batch_manager.py`  and provided by `exp_parameters.py`.

The simulation library runs automatically if a user starts the `exp_batch_manager.py` and defines the upper and lower experiment limits. During the run, various modules are loaded, and data is generated and processed. For the data evaluation, the setting in `exp_manager.py` is to be considered in particular. Here it can be set which exports/analyses are to be carried out by the simulation library.


##
For further information, please refer to the annotation of the code in the individual files.
Furthermore the author thanks *Arno Kasper* for providing the tool *Process Sim* (https://github.com/ArnoKasper/ProcessSim), which is the basis for the created simulation library, and for the permission for its further usage.


