"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Comprehends everything for the order flow
- Determination of Processing Times (via call)
- Determination of order due dates (via call)
- Determination of operational due dates (via call)
"""

import pandas as pd

class Order(object):
    # Initialisation----------------------------------------------------------------------------------------------------
    def __init__(self, simulation):
        """
        object having all attributes of the flow item
        also contains the sorting algorithms for the specified shop layout
        """

        self.sim = simulation
        self.identifier = None

        # source params
        self.entry_time = 0
        self.interarrival_time = 0

        # pool params
        self.location = "pool"
        self.release = False
        self.first_entry = True
        self.release_time = 0
        self.inter_release_time = 0
        self.pool_time = 0

        # routing sequence params
        if self.sim.model_panel.WC_AND_FLOW_CONFIGURATION in ["GFS", "PJS"]:
            self.routing_sequence = self.sim.random_generator.sample(
                self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT,
                self.sim.random_generator.randint(1, len(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT)))

            # Sort the routing if necessary
            if self.sim.model_panel.WC_AND_FLOW_CONFIGURATION == "GFS":
                self.routing_sequence.sort()  # GFS or PFS require sorted list of stations

        elif self.sim.model_panel.WC_AND_FLOW_CONFIGURATION == "PFS":
            self.routing_sequence = self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT.copy()

        elif self.sim.model_panel.WC_AND_FLOW_CONFIGURATION == "RJS":
            self.routing_sequence = self.sim.random_generator.sample(
                self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT,
                self.sim.random_generator.randint(len(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT),
                                                  len(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT)))

        else:
            raise Exception("no valid manufacturing process selected")

        if self.sim.model_panel.WC_AND_FLOW_CONFIGURATION_SORTING == True:
            a = self.sim.random_generator.random()
            if a <= self.sim.model_panel.WC_AND_FLOW_CONFIGURATION_SORTINGVALUE:
                if self.sim.model_panel.WC_AND_FLOW_CONFIGURATION_SORTINGCLASSIC:
                    self.routing_sequence.sort() # Classical Sort
                if self.sim.model_panel.WC_AND_FLOW_CONFIGURATION_SORTINGFIRST:
                    first = min(self.routing_sequence)
                    self.routing_sequence.remove(first)
                    self.routing_sequence.insert(0, first)

            if self.sim.model_panel.WC_AND_FLOW_CONFIGURATION_FIRSTWORKINGSYSTEM:
                first = min(self.routing_sequence)
                self.routing_sequence.remove(first)
                self.routing_sequence.insert(0, first)


        # definition of needed variables
        # make a variable independent from routing sequence to allow for queue switching
        self.routing_sequence_data = self.routing_sequence[:]

        # process time
        self.process_time = {}
        self.process_time_cumulative = 0
        self.remaining_process_time = 0

        # priority
        self.dispatching_priority = {}

        # data collection variables
        self.operation_queue_entry_time = {}
        self.operation_inter_arrival_time = {}
        self.operation_inter_depature_time = {}
        self.operation_finish_time = {}
        self.operation_time_in_queue = {}
        self.operation_start_time = {}
        self.wc_state = {}

        # work content--------------------------------------------------------------------------------------------------
        for WC in self.routing_sequence:
            # type of process time distribution
            if self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "2_erlang":
                self.process_time[WC] = self.sim.general_functions.two_erlang_truncated()
            elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "lognormal":
                self.process_time[WC] = self.sim.general_functions.log_normal_truncated()
            elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "exponential":
                self.process_time[WC] = self.sim.general_functions.exponential()
            elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "constant":
                self.process_time[WC] = self.sim.model_panel.MEAN_PROCESS_TIME
            elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "normal":
                self.process_time[WC] = self.sim.general_functions.normal_truncated()
            elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "weibull":
                self.process_time[WC] = self.sim.general_functions.weibull_truncated()
            else:
                raise Exception("no valid process time distribution selected")


            # calculate cum
            self.process_time_cumulative += self.process_time[WC]
            self.remaining_process_time += self.process_time[WC]

            self.dispatching_priority[WC] = 0

            # data collection variables
            self.operation_queue_entry_time[WC] = 0
            self.operation_finish_time[WC] = 0
            self.operation_time_in_queue[WC] = 0
            self.operation_start_time[WC] = 0
            self.wc_state[WC] = "NOT_PASSED"

        self.due_date = 0
        self.operation_due_date = {}
        self.operation_planned_input_date = {}
        self.planned_release_date = 0

        # due date------------------------------------------------------------------------------------------------------
        if self.sim.policy_panel.due_date_method == "random":
            self.due_date = self.sim.general_functions.random_value_DD(order=self)
        elif self.sim.policy_panel.due_date_method == "constant":
            self.due_date = self.sim.general_functions.add_constant_DD(order=self)
        elif self.sim.policy_panel.due_date_method == "work_content_and_allowance":
            self.due_date = self.sim.general_functions.work_content_and_allowance_DD(order=self)
        elif self.sim.policy_panel.due_date_method == "work_content_mean_times":
            self.due_date = self.sim.general_functions.work_content_mean_times_DD(order=self)
        elif self.sim.policy_panel.due_date_method == "mean_operation_times":
            self.due_date = self.sim.general_functions.mean_operation_times_DD(order=self)
        elif self.sim.policy_panel.due_date_method == "x_operation_times":
            self.due_date = self.sim.general_functions.x_operation_times_DD(order=self)
        else:
            raise Exception("no valid due date procedure selected")


        # operational due dates-----------------------------------------------------------------------------------------
        self.production_due_date= self.due_date - self.sim.general_functions.shiftcalender_reverse(
                                                    self.due_date,self.sim.policy_panel.safety_time)

        if self.sim.policy_panel.operational_due_date_method == "work_content_and_allowance":
            self.planned_release_date = self.sim.general_functions.work_content_and_allowance_ODD(order=self)
        elif self.sim.policy_panel.operational_due_date_method == "work_content_mean_times":
            self.planned_release_date = self.sim.general_functions.work_content_mean_times_ODD(order=self)
        elif self.sim.policy_panel.operational_due_date_method == "mean_operation_times":
            self.planned_release_date = self.sim.general_functions.mean_operation_times_ODD(order=self)
        elif self.sim.policy_panel.operational_due_date_method == "x_operation_times":
            self.planned_release_date = self.sim.general_functions.x_operation_times_ODD(order=self)
        else:
            raise Exception("no valid operational due date procedure selected")

        self.planned_entry_date= self.planned_release_date -self.sim.general_functions.shiftcalender_reverse(
                                                    self.planned_release_date,self.sim.policy_panel.planned_pool_time)


        # data collection for backlog control
        if self.sim.sim_started and self.sim.policy_panel.backlog_control:
                df = pd.DataFrame([self.planned_release_date], columns=['Time_Sys_Input'])
                df["Order_Sys_Input"] = 1
                df["Load_Sys_Input"] = self.process_time_cumulative
                df["Time_Sys_Output"] = self.production_due_date
                df["Order_Sys_Output"] = 1
                df["Load_Sys_Output"] = self.process_time_cumulative

                for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                    if work_centre in self.routing_sequence:
                        df[f"Time_WC_Input_{work_centre}"] = self.operation_planned_input_date[work_centre]
                        df[f"Order_WC_Input_{work_centre}"] = 1
                        df[f"Load_WC_Input_{work_centre}"] = self.process_time[work_centre]
                        df[f"Time_WC_Output_{work_centre}"] = self.operation_due_date[work_centre]
                        df[f"Order_WC_Output_{work_centre}"] = 1
                        df[f"Load_WC_Output_{work_centre}"] = self.process_time[work_centre]
                    else:
                        df[f"Time_WC_Input_{work_centre}"] = 0.0
                        df[f"Order_WC_Input_{work_centre}"] = 0
                        df[f"Load_WC_Input_{work_centre}"] = 0.0
                        df[f"Time_WC_Output_{work_centre}"] = 0.0
                        df[f"Order_WC_Output_{work_centre}"] = 0
                        df[f"Load_WC_Output_{work_centre}"] = 0.0

                self.sim.source.df_planned_values = pd.concat([self.sim.source.df_planned_values, df])

        self.finishing_time = 0
        self.first_queue_dispatching_time = None

        # other
        self.cards = []
        self.continuous_trigger = False
        self.pool_priority = 0
        return

    def __eq__(self, other):
        return self.identifier == other

