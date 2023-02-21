"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Management of the experiments
- Calls the Experiment Manager
- Execution of all Experiements in the Range (Lower_Limit, Upper_Limit)
- Prints basic information (Start, End)
"""

from exp_manager import Experiment_Manager
import time
import sys

# track run time
start_time = time.time()

# set the range of the experiments that needs to be run
# Sys needed in case of batch execution at cluster system, e.g. SLURM
lower_limit = 0 #((int(sys.argv[1])-1)) * 8 +2400
upper_limit = 7 #((int(sys.argv[1])-1)) * 8 + 7 +2400 #lower_limit+1

# activate the simulation (automatic model)
Experiment_Manager(lower_limit, upper_limit)

# provide essential experimental information
t_time = (time.time() - start_time)
t_hours = t_time // 60 // 60
t_min = (t_time - (t_hours * 60 * 60)) // 60
t_seconds = (t_time - (t_min * 60) - (t_hours * 60 * 60))

print(f"\n\nExperiment {lower_limit} till {upper_limit} are finished"
      f"\nThe total run time"
      f"\n\tHours:      {t_hours}"
      f"\n\tMinutes:    {t_min}"
      f"\n\tSeconds:    {round(t_seconds, 2)}")
