"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Defines the major functions of the simulation model
- Handels run time and warm-up time
- Triggers Data export
- Manages terminal print options
"""

from simpy import Environment, Event
from random import Random
import numpy as np
from scipy import stats
from typing import Generator
from controlpanel import ModelPanel, PolicyPanel
from data_collection_and_storage import Data_Continous_Run, Data_Experiment
from generalfunctions import GeneralFunctions
from simsource import Source
from process import Process
from releasecontrol import ReleaseControl
# from systemstatedispatching import SystemStateDispatching (Deactivated, see notice)

# Class of Simulation Model---------------------------------------------------------------------------------------------
class SimulationModel(object):
    """
    class containing the simulation model function
    the simulation instance (i.e. self) is passed in the other function outside this class as sim
    """

    # Initialisation of the entire Simulation Model---------------------------------------------------------------------
    def __init__(self, exp_number: int = 1) -> None:

        # setup general params
        self.inspect_process = None
        self.sim_started: bool = False
        self.exp_number: int = exp_number
        self.warm_up: bool = False

        # set seed for specifically process times and other random generators
        self.random_generator: Random = Random()
        self.random_generator.seed(1000000)

        # import the Simpy environment
        self.env: Environment = Environment()

        # add general functionality to the model
        self.general_functions: GeneralFunctions = GeneralFunctions(simulation=self)

        # get the model and policy control panel
        self.model_panel: ModelPanel = ModelPanel(experiment_number=self.exp_number, simulation=self)
        self.policy_panel: PolicyPanel = PolicyPanel(experiment_number=self.exp_number, simulation=self)
        self.print_info: bool = self.model_panel.print_info

        # set up the list containing all run databases
        self.detailled_run_db = list()
        self.continous_run_db = list()

        # initiate the data export procedure for continous data
        self.data_continous_run: Data_Continous_Run = Data_Continous_Run(simulation=self)

        # initiate the data export procedure for run and aggregate dara
        self.data: Data_Experiment = Data_Experiment(simulation=self)

        # add plot method - Needs Development
        #self.plot: run_plot = run_plot(simulation=self)

        # calculate adjusted inputs for process times
        if self.model_panel.TRUNCATION:
            self.general_functions.calculate_adjusted_process_time()

        # import source
        self.source: Source = Source(simulation=self)

        # import release, authorization control and state dependent dispatching
        self.release_control: ReleaseControl = ReleaseControl(simulation=self)
        #self.system_state_dispatching: SystemStateDispatching = SystemStateDispatching(simulation=self)

        # import process
        self.process: Process = Process(simulation=self)

        # declare variables
        self.release_periodic: any = None
        self.source_process: any = None
        self.run_manager: any = None

    # The actual simulation function with all required SimPy settings---------------------------------------------------
    def sim_function(self) -> None:
        """
        initialling and timing of the generator functions
        """
        self.sim_started = True

        # activate release control, LOOR and WLC_LD are own processes, triggered if choosen!
        if self.policy_panel.release_control:
            if self.policy_panel.release_control_method in ["LUMS_COR", "WLC_AL", "WLC_CAL"]:
                self.release_periodic: Process[Event, None, None] = \
                    self.env.process(self.release_control.periodic_release())
            if self.policy_panel.release_control_method == "LOOR":
                self.release_periodic: Process[Event, None, None] = \
                    self.env.process(self.release_control.LOOR())
            if self.policy_panel.release_control_method == "WLC_LD":
                self.release_periodic: Process[Event, None, None] = \
                    self.env.process(self.release_control.WLC_LD())

        # initialize processes
        self.source_process: Process[Event, None, None] = self.env.process(self.source.generate_random_arrival_exp())

        # initialize backlog_control
        if self.policy_panel.backlog_control:
            self.inspect_process: Process[Event, None, None] = self.env.process(self.process.backlog_control)

        # initialize WIP_control
        if self.policy_panel.WIP_control:
            self.inspect_process: Process[Event, None, None] = self.env.process(self.process.WIP_control)

        # initialize release_time_limit
        if self.policy_panel.release_time_limit:
            self.release_time_trigger: Process[Event, None, None] = \
                self.env.process(self.release_control.release_time_limit_trigger())

        # activate data collection methods
        self.run_manager: Process[Event, None, None] = self.env.process(SimulationModel.run_manager(self))

        # activate continous data tracking
        self.continous_tracking: Process[Event, None, None] = \
            self.env.process(self.data_continous_run.continous_data_getter())

        # start and end of simulation + info_print
        if self.print_info:
            self.print_start_info()

        # start simulation
        sim_time = (self.model_panel.WARM_UP_PERIOD + self.model_panel.RUN_TIME) * \
                   self.model_panel.NUMBER_OF_RUNS + 0.001

        self.env.run(until=sim_time)

        # simulation finished, print final info
        if self.print_info:
            self.print_end_info()

    # Run manager running the simulation--------------------------------------------------------------------------------
    def run_manager(self) -> Generator[Event, None, None]:
        """
        The run manager managing processes during the simulation. Can perform the same actions in through cyclic
        manner. Currently, the run_manager managers printing information and the saving and processing of information.
        """
        while self.env.now < (
                self.model_panel.WARM_UP_PERIOD + self.model_panel.RUN_TIME) * self.model_panel.NUMBER_OF_RUNS:
            yield self.env.timeout(self.model_panel.WARM_UP_PERIOD)

            # change the warm_up status
            self.warm_up = True

            # print run info if required
            if self.print_info:
                self.print_warmup_info()

            # update data
            self.data_continous_run.run_update(warmup=self.warm_up)
            self.data.run_update(warmup=self.warm_up)

            # pause for run_time
            yield self.env.timeout(self.model_panel.RUN_TIME)

            # change the warm_up status after run_time is over
            self.warm_up = False

            # update data
            self.data_continous_run.run_update(warmup=self.warm_up)
            self.data.run_update(warmup=self.warm_up)

            # add a new run database es list entry to list of all run database which is exported in the end
            self.continous_run_db.append(self.data_continous_run.continous_database)
            self.detailled_run_db.append(self.data.run_database)

            # update values in case of backlog_control
            if self.policy_panel.backlog_control:
                self.source.update_df_planned_values()

            # print run info if required
            if self.print_info:
                self.print_run_info()

            self.source.initial_pause=True

    # Function that prints information to the console-------------------------------------------------------------------
    def print_start_info(self) -> None:
        print(f"Simulation starts with experiment: {self.model_panel.experiment_name}")
        # print(f"Mean time between arrival: {self.model_panel.MEAN_TIME_BETWEEN_ARRIVAL}")
        # add more information if needed
        return

    def print_warmup_info(self) -> None:
        return print('Warm-up period finished')

    # Function for printing run information-----------------------------------------------------------------------------
    def print_run_info(self) -> None:

        # vital simulation results are given
        run_number = int(self.env.now / (self.model_panel.WARM_UP_PERIOD + self.model_panel.RUN_TIME))
        index = run_number - 1

        # make progress bar
        progress = "["
        step = 100/self.model_panel.NUMBER_OF_RUNS

        for i in range(1, 101):
            if run_number * step > i:
                progress = progress + "="
            elif run_number * step == i:
                progress = progress + ">"
            else:
                progress = progress + "."
        progress = progress + f"] {round(run_number/self.model_panel.NUMBER_OF_RUNS*100,2)}%"

        # compute replication confidence
        try:
            print(f"run number {run_number}", progress)
            result = self.replication_confidence_interval(run_number=run_number, criteria="mean_throughput_time")
            statistics = f"replication statistics (Mean_TTP): {result}"
            print(statistics)

            result = self.replication_confidence_interval(run_number=run_number, criteria="std_throughput_time")
            statistics = f"replication statistics (Std_TTP): {result}"
            print(statistics)

        except (KeyError, IndexError):
            print("could not print simulation results")
        return

    def print_end_info(self) -> None:
        print("Simulation ends")
        return

    # Function for calculation replication statistics-------------------------------------------------------------------
    def replication_confidence_interval(self, run_number, criteria):

        # defining confidence_int and value to be observed
        running_mean = self.data.experiment_database.loc[:, criteria].mean()
        current_std_dev = self.data.experiment_database.loc[:, criteria].std()
        confidence_int = stats.t.ppf(1 - 0.025, df=self.data.experiment_database.shape[0] - 1) * \
                         (current_std_dev / np.sqrt(run_number))
        lower_confidence = round(running_mean - confidence_int, 4)
        upper_confidence = '%.4f' % round(running_mean + confidence_int, 4)
        deviation = '%.4f' % round(((running_mean - lower_confidence) / running_mean) * 100, 4)
        lower_confidence = '%.4f' % lower_confidence
        running_mean = '%.4f' % round(running_mean, 4)
        return [running_mean, lower_confidence, upper_confidence, deviation]

