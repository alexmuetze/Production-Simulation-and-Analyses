# Production Simulation and Analyses

Das Tool Production Simulation und Analyses ist eine Simulationsbibliothek, welche 


Production Simulation and Analyses is a simulation library which can simulate a production shop with different 
procedures for Production Planning and Control (PPC) focusing Order Release, Capacity Control and Dispatching. 
It can be fully controlled by any control systems build by the modeller. 
Futhermore, the model includes an experimental layer for parallel and sequential simulation experiments using
SLURM.

## Install the Required Dependencies
Production Simulation and Analyses ist komplett auf Python basiert. 


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


