# Production-Simulation-and-Analyses

Production Simulation and Analyses is a simulation library which can simulate a production shop with different 
procedures for Production Planning and Control (PPC) focusing Order Release, Capacity Control and Dispatching. 
It can be fully controlled by any control systems build by the modeller. 
Futhermore, the model includes an experimental layer for parallel and sequential simulation experiments using
SLURM.

## Install the Required Dependencies
Install the Python 3 packages listed in the following table. There are two ways to install the required dependencies. Firstly, use the
`requirements.txt` to install packages easily using `pip`. Secondly, you
can download packages manually.

| Package | Version |
| --: | --: |
| `numpy` | 1.23.0 |
| `pandas` | 1.4.3 |
| `simpy` | 4.0.1 |
| `scipy` | 1.8.1 |

MatPlotLib mit reinnehmen!

## Use
The model settings can be changes are stored in `control_panel.py`. That file includes two classes. The first class `ModelPanel` contains the basic settigns which cannot be changed during simulations. The second class `PolicyPanel` are options which can be changed at any point during the simulation. Alternativly, one can specificy additional functionality in `customized_settings.py` and aplly the setting Customized in the correct settingsfields in `control_panel.py`. 

## Documentation
Das Simulationsmodell erlaubt die folgenden Simulationen:

Auftragsfreigabeverfahren
Poolsequencing rule
Durchlaufterminierungsart
Durchlaufzeitbestimmung...




##
Für weitere Informationen Verweis auf Dissertation Alexander Mütze


