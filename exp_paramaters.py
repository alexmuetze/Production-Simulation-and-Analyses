"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Module creates experiment list which afterwards is input for the exp_batch_manager
"""

import numpy as np

#utilization = [*range(80, 100, 1)]
#utilization = [x / 100 for x in utilization]

utilization = [80,85,90,95,96,97,98,99]
utilization = [x / 100 for x in utilization]
cv = [0, 0.25, 0.5, 0.75, 1]
due_date_setting = [[20,60], [30,50], [10,70]]
workcontent = ["2_erlang", "exponential"]
norm = [150]
shoplayout = ["RJSe75","RJS", "PFS"]


dispatching = [['SPT','SPT','SPT','SPT','SPT','SPT'],['FCFS','FCFS','FCFS','FCFS','FCFS','FCFS'],['True_Random','True_Random','True_Random','True_Random','True_Random','True_Random'],['EODD','EODD','EODD','EODD','EODD','EODD']]
sequencing = ["PRD"]

experimental_params_dict = []

# Creating the Parameter Dictionary-------------------------------------------------------------------------------------
def get_interactions():

    for shoplayout_i in shoplayout:
        for workcontent_i in workcontent:
            for due_date_setting_i in due_date_setting:
                for cv_i in cv:
                    for norm_i in norm:
                        for dispatching_i in dispatching:
                            for sequencing_i in sequencing:
                                for utilization_i in utilization:
                                    params_dict=dict()
                                    params_dict["release_rule"]="CONWIP"
                                    params_dict["release_norm"]= norm_i
                                    params_dict["utilization"]=utilization_i
                                    params_dict["cv"]= round(cv_i,1)
                                    params_dict["workcontent"]= workcontent_i
                                    params_dict["shoplayout"]=shoplayout_i
                                    params_dict["Due_Date_Setting"]=due_date_setting_i
                                    params_dict["Dispatching"]= dispatching_i
                                    params_dict["Sequencing"] = sequencing_i
                                    experimental_params_dict.append(params_dict)


    # Experimental_Params_Dict is read by batch manager and defines the possible upper bound of a batch
    #print(experimental_params_dict)
    return experimental_params_dict


# activate the code
if __name__ == '__main__':
    experimental_params_dict = get_interactions()