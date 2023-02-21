"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Defines the overall model
- ModelPanel defines shoplayout, throughput time, etc. -> General Organizational Descisions
- PolicyPanel defines PPC tasks and their configuration --> Dispatching, Release, Pool-Sequencing, ...
"""

import numpy as np

from typing import Dict, List, ClassVar
from generalfunctions import GeneralFunctions
import exp_paramaters as parameters
from simpy import FilterStore, PriorityResource

# Class defining the overall simulation modell like shoplayout, throughput times, etc.----------------------------------
class ModelPanel(object):

    # Initialisation of ModelPanel--------------------------------------------------------------------------------------
    def __init__(self, experiment_number: int, simulation: ClassVar) -> None:

        # basic variables-----------------------------------------------------------------------------------------------
        self.experiment_number: int = experiment_number
        self.sim: ClassVar = simulation
        self.print_info: bool = True
        self.print_results: bool = True
        experimental_params_dict = parameters.get_interactions()
        self.params_dict: Dict[...] = experimental_params_dict[self.experiment_number]
        self.general_functions: GeneralFunctions = GeneralFunctions(simulation=self.sim)

        # experiment names and folder name definition in order to create a new space environment for saving experiments-
        self.indexes = ["release_rule",
                        "release_norm",
                        "Sequencing",
                        "cv",
                        "shoplayout",
                        "workcontent",
                        "Due_Date_Setting",
                        "utilization"
                        ]
        self.experiment_name: str = ""

        for i in self.indexes:
            self.experiment_name += str(self.params_dict[i]) + "_"

        #self.new_folder: str=""

        self.indexes2=["release_rule","shoplayout","cv","workcontent","Due_Date_Setting", "Sequencing", "Dispatching"]
        self.new_folder: str= ""
        for i in self.indexes2:
            self.new_folder +=str(self.params_dict[i]) + "/"

        # simulation parameters-----------------------------------------------------------------------------------------
        self.WARM_UP_PERIOD: int = 3000    # warm-up period simulation model
        self.RUN_TIME: int = 10000   # run time simulation model
        self.NUMBER_OF_RUNS: int = 5       # number of replications

        # manufacturing process and order characteristics / basic settings----------------------------------------------
        self.NUMBER_OF_WORKCENTRES: int = 6
        self.MANUFACTURING_FLOOR_LAYOUT: List[str, ...] = []
        for i in range(0, self.NUMBER_OF_WORKCENTRES):
            self.MANUFACTURING_FLOOR_LAYOUT.append(f'WC{i}')

        self.ORDER_POOL: FilterStore = FilterStore(self.sim.env)
        self.ORDER_POOLS: Dict[...] = {}
        self.ORDER_QUEUES: Dict[...] = {}
        self.MANUFACTURING_FLOOR: Dict[...] = {}  # The manufacturing floor
        self.NUMBER_OF_MACHINES: int = 1

        for i, WC in enumerate(self.MANUFACTURING_FLOOR_LAYOUT):
            self.ORDER_POOLS[WC]: FilterStore = FilterStore(self.sim.env)
            self.ORDER_QUEUES[WC]: FilterStore = FilterStore(self.sim.env)
            self.MANUFACTURING_FLOOR[WC]: PriorityResource = \
                PriorityResource(self.sim.env, capacity=self.NUMBER_OF_MACHINES)

        # capacity settings---------------------------------------------------------------------------------------------
        self.STANDARDCAPACITY = 8
        self.UPPERBARRIER = 10
        self.LOWERBARRIER = 6
        self.ACTUALCAPA: Dict[str,float] = {}

        for i, WC in enumerate(self.MANUFACTURING_FLOOR_LAYOUT):
            self.ACTUALCAPA[WC] = self.STANDARDCAPACITY

        # manufacturing model configuration-----------------------------------------------------------------------------
        """
        Options for the configuration of flows: According to Oostermann et al. 2000
            1. PJS: pure job shop
            2. GFS: general flow shop
            3. PFS: pure flow shop
            4. RJS: restricted job shop
            
        Plus special routing settings enabling the definition of a fixed first work center and routing strengths
        """

        if self.params_dict["shoplayout"] == "RJS":
            self.WC_AND_FLOW_CONFIGURATION: str = 'RJS'
            self.WC_AND_FLOW_CONFIGURATION_SORTING: bool = False
            self.WC_AND_FLOW_CONFIGURATION_SORTINGCLASSIC : bool = False
            self.WC_AND_FLOW_CONFIGURATION_SORTINGFIRST: bool = False
            self.WC_AND_FLOW_CONFIGURATION_FIRSTWORKINGSYSTEM: bool = False
            self.WC_AND_FLOW_CONFIGURATION_SORTINGVALUE: float = 0

        if self.params_dict["shoplayout"] == "RJSe":
            self.WC_AND_FLOW_CONFIGURATION: str = 'RJS'
            self.WC_AND_FLOW_CONFIGURATION_SORTING: bool = True
            self.WC_AND_FLOW_CONFIGURATION_SORTINGCLASSIC : bool = False
            self.WC_AND_FLOW_CONFIGURATION_SORTINGFIRST: bool = False
            self.WC_AND_FLOW_CONFIGURATION_FIRSTWORKINGSYSTEM: bool = True
            self.WC_AND_FLOW_CONFIGURATION_SORTINGVALUE: float = 0

        if self.params_dict["shoplayout"] == "RJS75":
            self.WC_AND_FLOW_CONFIGURATION: str = 'RJS'
            self.WC_AND_FLOW_CONFIGURATION_SORTING: bool = True
            self.WC_AND_FLOW_CONFIGURATION_SORTINGCLASSIC : bool = True
            self.WC_AND_FLOW_CONFIGURATION_SORTINGFIRST: bool = False
            self.WC_AND_FLOW_CONFIGURATION_FIRSTWORKINGSYSTEM: bool = False
            self.WC_AND_FLOW_CONFIGURATION_SORTINGVALUE: float = 0.75

        if self.params_dict["shoplayout"] == "RJSe75":
            self.WC_AND_FLOW_CONFIGURATION: str = 'RJS'
            self.WC_AND_FLOW_CONFIGURATION_SORTING: bool = True
            self.WC_AND_FLOW_CONFIGURATION_SORTINGCLASSIC : bool = True
            self.WC_AND_FLOW_CONFIGURATION_SORTINGFIRST: bool = False
            self.WC_AND_FLOW_CONFIGURATION_FIRSTWORKINGSYSTEM: bool = True
            self.WC_AND_FLOW_CONFIGURATION_SORTINGVALUE: float = 0.75

        if self.params_dict["shoplayout"] == "PFS":
            self.WC_AND_FLOW_CONFIGURATION: str = 'PFS'
            self.WC_AND_FLOW_CONFIGURATION_SORTING: bool = False
            self.WC_AND_FLOW_CONFIGURATION_SORTINGCLASSIC : bool = False
            self.WC_AND_FLOW_CONFIGURATION_SORTINGFIRST: bool = False
            self.WC_AND_FLOW_CONFIGURATION_FIRSTWORKINGSYSTEM: bool = False
            self.WC_AND_FLOW_CONFIGURATION_SORTINGVALUE: float = 0


        # process and arrival times-------------------------------------------------------------------------------------
        """
        process time distribution versions
            - lognormal:    Lognormal Distribution
            - normal:       Normal Distribution
            - weibull:      Weibull Distribution
            - 2_erlang:     2-Erlang Distribution
            - exponential:  Exponential (1-Erlang Distribution)
            - constant:     constant process time value
        """
        self.PROCESS_TIME_DISTRIBUTION: str = self.params_dict["workcontent"]
        self.AIMED_UTILIZATION: float = self.params_dict["utilization"]
        self.VARIATION_ARRIVAL: float = self.params_dict["cv"]
        self.MEAN_PROCESS_TIME: int = 1
        self.STD_DEV_PROCESS_TIME: float = 1*0.5
        self.TRUNCATION_POINT_PROCESS_TIME: any = 4* self.MEAN_PROCESS_TIME
        self.TRUNCATION: bool = True

        self.MEAN_PROCESS_TIME_ADJ: float = 0
        self.STD_DEV_PROCESS_TIME_ADJ: float = 0
        self.MEAN_PROCESS_TIME_TRUE: float = 0
        self.STD_DEV_PROCESS_TIME_TRUE: float = 0
        self.MEAN_TIME_BETWEEN_ARRIVAL_Recalculation: bool = True

        # option for smoothing the arrival of jobs, i.e. creating a covariance between workcontent and interarrival time
        self.SMOOTHING = False

        # calculate mean time between arrival---------------------------------------------------------------------------
        self.MEAN_TIME_BETWEEN_ARRIVAL: float = \
            self.general_functions.arrival_time_calculator(
                wc_and_flow_config=self.WC_AND_FLOW_CONFIGURATION,
                manufacturing_floor_layout=self.MANUFACTURING_FLOOR_LAYOUT,
                aimed_utilization=self.AIMED_UTILIZATION,
                mean_process_time=self.MEAN_PROCESS_TIME,
                number_of_machines=self.NUMBER_OF_MACHINES)

        # variables used for workload calculations----------------------------------------------------------------------
        self.PROCESSED: Dict[str, float] = {}  # keeps record of the processed orders/load
        self.RELEASED: Dict[str, float] = {}  # keeps record of the released orders/load
        self.LOAD_ACCOUNT: Dict[str, float] = {}  # keeps record of the load account in case LOOR

        for WC in self.MANUFACTURING_FLOOR_LAYOUT:
            self.PROCESSED[WC] = 0.0
            self.RELEASED[WC] = 0.0
            self.LOAD_ACCOUNT[WC] = 0.0

        # activate the appropriate data collection methods--------------------------------------------------------------
        self.COLLECT_BASIC_DATA: bool = False


# Class defining the planning and control policies of PPC---------------------------------------------------------------
class PolicyPanel(object):

    # Initialisation of PolicyPanel--------------------------------------------------------------------------------------
    def __init__(self, experiment_number: int, simulation: ClassVar) -> None:
        self.experiment_number: int = experiment_number
        self.params_dict: Dict[...] = parameters.experimental_params_dict[self.experiment_number]
        self.sim: ClassVar = simulation

        # customer enquiry management - Due Date and Operational Due Date Determination---------------------------------
        """
        - The following options exist for DueDate-Setting
        
            -   random: adds a random Due Date with a uniform continuous distribution
                    Location: GeneralFunctions.random_value_DD
            -   constant: adds a constant due date to the input data
                    Location: GeneralFunctions.add_constant_DD 
            -   work_content_and_allowance: adds the cumulative process time and a constant allowance
                    Location: GeneralFunctions.work_content_and_allowance_DD
            -   work_content_mean_times: adds the cumulative process time and a constant allowance per work system
                    Location: GeneralFunctions.work_content_mean_times_DD
            -   mean_operation_times: adds a constant allowance per work system
                    Location: GeneralFunctions.mean_operation_times_DD(order=self) 
            -   x_operation_times: adds the cumulative process time multiplied by factor k to the input (+ 2 basic values)
                    Location: GeneralFunctions.x_operation_times_DD         
             
        - The following options exist for Operational DueDate-Setting
        
            -   work_content_and_allowance: adds the work content and the divided constant DD allowance (by routing length)
                    Location: GeneralFunctions.work_content_and_allowance_ODD
            -   work_content_mean_times: adds the work content and the mean operation time value
                    Location: GeneralFunctions.work_content_mean_times_ODD
            -   mean_operation_times: adds the mean operation time value
                    Location: GeneralFunctions.mean_operation_times_ODD
            -   x_operation_times: adds the process time multiplied by factor k
                    Location: GeneralFunctions.x_operation_times_ODD
        """
                    

        self.due_date_method: str = "random"
        self.DD_random_min_max: List[int, int] = self.params_dict["Due_Date_Setting"]
        self.DD_factor_K_value: int = 12
        self.DD_basic_value_allowance: int = 4
        self.DD_constant_value: float = 40
        self.DD_total_work_content_value: float = 8
        self.DD_mean_operation_time_value: float = 5
        self.operational_due_date_method = "mean_operation_times"
        self.planned_pool_time = 5
        self.safety_time = 5

        # Release control ----------------------------------------------------------------------------------------------
        self.capacityflexibilty: bool = True
        self.backlog_control = False
        self.WIP_control = False

        # pool sequencing rules-----------------------------------------------------------------------------------------
        """
        Release sequencing rules 
            - FCFS
            - PRD
            - SPT_Total
            - SPT
            - EDD
            - MODCS
            - SLACK
            - "Once_Random", "True_Random"
        """
        self.sequencing_rule: str = self.params_dict["Sequencing"]

        # available release rules for the simulation--------------------------------------------------------------------
        """
        Release rules 
        - LUMS_COR
        - WLC_AL
        - WLC_CAL
        - WLC_LD
        - LOOR
        - DOOR
        - CONWIP
        - CONLOAD
        - UBR, UBRLB, UBR_Trigger, UBRLB_Trigger
        - Immediate
        """
        """
            Key for the release control 
                Workload Control options using corrected aggregate load (Oosterman et al., 2000)
                -   LUMS_COR: combines both periodic and continuous release (Thürer et al., 2012)
                -   pure_periodic: periodic release (Land et al., 2004) [WLC_AL, WLC_CAL]
                -   pure_continuous:
                    - UBR: upper bound release (Thürer et al., 2014)
                    - UBR_Trigger: UBR with continuous trigger (Fernandes et al., 2017)
                    - UBRLB: upper bound release with load balancing (Thürer et al., 2014)
                    - UBRLB_Trigger: UBRLB with continuous trigger (Fernandes et al., 2017)

                Other options
                -   CONWIP: Constant Work In Process
                -   CONLOAD: Constant Workload In Process
                -   DOOR: Due-Date Oriented Order Release
                -   LOOR: Load-Oriented Order Release
                -   WLC_LD: Workload-Control (acc. to Lödding 2016)
        """
        self.release_control: bool = True
        self.release_norm: float = self.params_dict["release_norm"]
        self.release_control_method: str = self.params_dict["release_rule"]

        # Enabling and Defining time limit for release
        self.release_time_limit: bool = False
        self.release_time_limit_value: float = 0 #in Hours

        # Enabling and Defining time limit for starvation avoidance
        self.starvation_avoidance_time_limit: bool = False
        self.starvation_avoidance_limit_value: float = 200

        # Parameters for Periodic Release
        self.check_period: float = 4  # Periodic release time
        self.continuous_trigger: int = 0  # Initiale Determine the trigger point

        # Special Parameters for LOOR, needed to be defined; no influence if LOOR is not choosen!
        self.LOOR_Abfa: Dict[str, float] = {}  # Dictionary for Percentage Ratio LOOR
        self.LOOR_Load_Limit: Dict[str, float] = {}  # Dictionary of LOOR-Load Limits

        self.LOOR_EPS = [800,800,800,800,800,800]
        self.LOOR_EPS_Dict = {}
        self.LOOR_EPS_Dict = dict(zip(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT, self.LOOR_EPS))

        for _, work_centre in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            self.LOOR_Abfa[work_centre] = (1 / self.LOOR_EPS_Dict[work_centre]) * 100
            self.LOOR_Load_Limit[work_centre] = (self.LOOR_EPS_Dict[
                                                 work_centre] / 100) * self.check_period * self.sim.model_panel.AIMED_UTILIZATION

        # dispatching---------------------------------------------------------------------------------------------------
        """
        dispatching mode determines how to take the dispatching decision:
            - system_state_dispatching: use a PPC model to control the dispatching decision, deactivated!
            - priority_rule: use priority rules to control the dispatching decision
        """

        self.dispatching_mode: str = 'priority_rule'

        """ 
            - FISFO         -- First-In-System-First-Out --> Order Number
            - FCFS          -- First-Come-First-Sever --> Input Date
            - SPT           -- Processing Time
            - MODD          -- (Deactivated)
            - EODD          -- Earliest Operation Due Date
            - Once_Random   -- Random Number Order Entry
            - True_Random   -- True Random every time a draw is made
            - SLACK         -- Slack Rule (DueDate minus Work Content)
        """

        # priority rule; option for defining a print_name for dispatching if required
        self.dispatching_rules = self.params_dict["Dispatching"]
        self.dispatching_rules_print= "See Name"


        # deactivated (system_state_dispatching) --> (see Kasper et al. 2023: 10.1016/j.omega.2022.102726)--------------
        """
        system state dispatching method available
            - FOCUS
            - DRACO
        
        self.system_state_dispatching_version = "DRACO"
        self.system_state_dispatching_release = self.params_dict["ssd_order_release"]
        self.system_state_dispatching_authorization = self.params_dict["ssd_order_authorization"]
        self.system_state_dispatching_release_authorization = self.system_state_dispatching_authorization or \
                                                              self.system_state_dispatching_release
        self.WIP_target: float = self.params_dict["release_target"]
        self.loop_WIP_target_jk: float = self.params_dict["authorization_target"]
        """