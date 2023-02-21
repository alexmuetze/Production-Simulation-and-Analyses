"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Module comprehends all general functions needed for computation
- Is essential for the calculation of distributions etc.
- Calculates Due Dates etc.
- Has Shift Calender Functionality
"""

import random
import numpy as np
import csv

class GeneralFunctions(object):
    # Init Function for e.g. Random Generator Seed----------------------------------------------------------------------
    def __init__(self, simulation):
        self.sim = simulation
        self.random_generator = random.Random()
        self.random_generator.seed(18183)
        self.random_generator2=random.Random()
        self.random_generator2.seed(18184)
        self.numpy_generator= np.random.default_rng(18183)

    # Arrival Time Calculation------------------------------------------------------------------------------------------
    def arrival_time_calculator(self, wc_and_flow_config, manufacturing_floor_layout, aimed_utilization,
                                mean_process_time, number_of_machines):
        """
        compute the inter arrival time
        """

        mean_amount_work_centres = 0
        if wc_and_flow_config == "GFS" or wc_and_flow_config == "PJS":
            mean_amount_work_centres = (len(manufacturing_floor_layout) + 1) / 2

        elif wc_and_flow_config == "PFS" or wc_and_flow_config == "RJS":
            mean_amount_work_centres = len(manufacturing_floor_layout)

        # calculate the mean inter-arrival time
        inter_arrival_time = mean_amount_work_centres / len(manufacturing_floor_layout) * \
                             1 / aimed_utilization * mean_process_time / number_of_machines

        # round the float to five digits accuracy
        inter_arrival_time = round(inter_arrival_time, 5)
        return inter_arrival_time

    # Calculation of Processing Times (recognizing Truncation via value setted by other function)-----------------------
    # exponential
    def exponential(self):
        truncation_point = self.sim.model_panel.TRUNCATION_POINT_PROCESS_TIME
        if truncation_point == np.inf:
            return_value = self.numpy_generator.exponential(self.sim.model_panel.MEAN_PROCESS_TIME)
            return return_value

        # pull truncated value
        return_value = np.inf
        while return_value > truncation_point:
            return_value = self.numpy_generator.exponential(self.sim.model_panel.MEAN_PROCESS_TIME_ADJ)
        return return_value

    # log-normal
    def log_normal_truncated(self):
        truncation_point = self.sim.model_panel.TRUNCATION_POINT_PROCESS_TIME
        if truncation_point == np.inf:
            return_value = 0
            while return_value <= 0:
                return_value = self.random_generator.lognormvariate(self.sim.model_panel.MEAN_PROCESS_TIME,
                                                                self.sim.model_panel.STD_DEV_PROCESS_TIME)
            return return_value

        return_value = np.inf
        while return_value > truncation_point or return_value <= 0:
            return_value = self.random_generator.lognormvariate(self.sim.model_panel.MEAN_PROCESS_TIME_ADJ,
                                                                self.sim.model_panel.STD_DEV_PROCESS_TIME_ADJ)
        return return_value

    # normal
    def normal_truncated(self):
        truncation_point = self.sim.model_panel.TRUNCATION_POINT_PROCESS_TIME
        if truncation_point == np.inf:
            return_value = 0
            while return_value <= 0:
                return_value = self.random_generator.normalvariate(self.sim.model_panel.MEAN_PROCESS_TIME,
                                                                self.sim.model_panel.STD_DEV_PROCESS_TIME)
            return return_value

        return_value = np.inf
        while return_value > truncation_point or return_value <= 0:
            return_value = self.random_generator.normalvariate(self.sim.model_panel.MEAN_PROCESS_TIME_ADJ,
                                                                self.sim.model_panel.STD_DEV_PROCESS_TIME_ADJ)
        return return_value

    # weibull
    def weibull_truncated(self):
        truncation_point = self.sim.model_panel.TRUNCATION_POINT_PROCESS_TIME
        if truncation_point == np.inf:
            return_value = 0
            while return_value <= 0:
                return_value = self.random_generator.weibullvariate(self.sim.model_panel.MEAN_PROCESS_TIME,
                                                                self.sim.model_panel.STD_DEV_PROCESS_TIME)
            return return_value

        return_value = np.inf
        while return_value > truncation_point or return_value <= 0:
            return_value = self.random_generator.weibullvariate(self.sim.model_panel.MEAN_PROCESS_TIME_ADJ,
                                                                self.sim.model_panel.STD_DEV_PROCESS_TIME_ADJ)
        return return_value

    # two-erlang
    def two_erlang_truncated(self):
        truncation_point = self.sim.model_panel.TRUNCATION_POINT_PROCESS_TIME
        if truncation_point == np.inf:
            return_value = self.numpy_generator.gamma(2,self.sim.model_panel.MEAN_PROCESS_TIME/2)
            return return_value

        return_value = np.inf
        while return_value > truncation_point:
            return_value = self.numpy_generator.gamma(2,self.sim.model_panel.MEAN_PROCESS_TIME_ADJ/2)
        return return_value


    # Shiftcalender Functions (Forward, Reverse, Duration)--------------------------------------------------------------
    # forward
    def shiftcalender_forward(self, start, duration):
        self.sc_duration = duration
        self.sc_modified_duration: float=0
        self.sc_start = start

        if not self.sim.policy_panel.capacityflexibilty:
            return self.sc_duration

        else:
            self.startoffset = 0
            if self.sc_start % 10 >= self.sim.model_panel.STANDARDCAPACITY:
                self.startoffset = 10 - (self.sc_start % 10)
                self.sc_start = 0
            self.sc_modified_duration += self.startoffset

            while self.sc_duration > 0:
                if self.sc_start % 10 + self.sc_duration <= self.sim.model_panel.STANDARDCAPACITY:
                    self.sc_modified_duration += self.sc_duration
                    self.sc_duration= 0
                    continue

                if self.sc_start % 10 + self.sc_duration > self.sim.model_panel.STANDARDCAPACITY:
                    self.sc_duration -= (self.sim.model_panel.STANDARDCAPACITY - (self.sc_start% 10))
                    self.sc_modified_duration += 10-(self.sc_start % 10)
                    self.sc_start = 0

            return self.sc_modified_duration

    # reverse
    def shiftcalender_reverse(self, end, duration):
        self.sc_duration = duration
        self.sc_modified_duration: float = 0
        self.sc_end = end

        if not self.sim.policy_panel.capacityflexibilty:
            return self.sc_duration

        else:
            self.endoffset = 0
            if self.sc_end % 10 >= self.sim.model_panel.STANDARDCAPACITY:
                self.endoffset = self.sc_end % 10 - self.sim.model_panel.STANDARDCAPACITY
                self.sc_end = self.sim.model_panel.STANDARDCAPACITY #warum start...
            self.sc_modified_duration += self.endoffset

            while self.sc_duration > 0:
                if self.sc_end % 10 - self.sc_duration >= 0:
                    self.sc_modified_duration += self.sc_duration
                    self.sc_duration = 0
                    continue

                if self.sc_end % 10 - self.sc_duration < 0:
                    self.sc_duration -= self.sc_end % 10
                    self.sc_modified_duration += (self.sc_end % 10 + 10 - self.sim.model_panel.STANDARDCAPACITY)
                    self.sc_end = self.sim.model_panel.STANDARDCAPACITY

            return self.sc_modified_duration

    # duration
    def shiftcalender_duration(self, start, end):
        self.sc_end = end
        self.sc_start = start
        self.sc_duration: float = end-start

        if not self.sim.policy_panel.capacityflexibilty:
            return self.sc_duration

        else:
            self.sc_duration = 0
            if self.sc_end % 10 >= 0:
                self.sc_duration = self.sc_end % 10
                self.sc_end = self.sc_end - self.sc_duration

            while self.sc_end > self.sc_start:
                if self.sc_end - self.sc_start >= 10:
                    self.sc_duration += self.sim.model_panel.STANDARDCAPACITY
                    self.sc_end -= 10
                    continue

                if self.sc_end - self.sc_start < 10:
                    self.sc_end -= 2
                    self.sc_duration += (self.sc_end-self.sc_start)
                    self.sc_end = self.sc_start

            return self.sc_duration

    # Due Date Calculation for entire order (determination of Customer_Due_Date)----------------------------------------
    # random
    def random_value_DD(self, order):

        self.duration =self.random_generator.uniform(self.sim.policy_panel.DD_random_min_max[0],
            self.sim.policy_panel.DD_random_min_max[1])
        self.modified_duration = self.shiftcalender_forward(self.sim.env.now, self.duration)

        ReturnValue = self.sim.env.now + self.modified_duration
        return ReturnValue

    # constant
    def add_constant_DD(self, order):

        self.duration = self.sim.policy_panel.DD_constant_value
        self.modified_duration = self.shiftcalender_forward(self.sim.env.now, self.duration)

        ReturnValue = self.sim.env.now + self.modified_duration
        return ReturnValue

    # work content and constant allowance
    def work_content_and_allowance_DD(self, order):

        self.duration = order.process_time_cumulative + self.sim.policy_panel.DD_basic_value_allowance
        self.modified_duration = self.shiftcalender_forward(self.sim.env.now, self.duration)

        ReturnValue = self.sim.env.now + self.modified_duration
        return ReturnValue

    # total work content and inter-operation time
    def work_content_mean_times_DD(self, order):

        self.duration = order.process_time_cumulative +\
                        (self.sim.policy_panel.DD_mean_operation_time_value * (len(order.routing_sequence)+2))
        self.modified_duration = self.shiftcalender_forward(self.sim.env.now, self.duration)

        ReturnValue = self.sim.env.now + self.modified_duration
        return ReturnValue

    # constant throughput time per operation
    def mean_operation_times_DD(self, order):

        self.duration = self.sim.policy_panel.DD_mean_operation_time_value * (len(order.routing_sequence) + 2)
        self.modified_duration = self.shiftcalender_forward(self.sim.env.now, self.duration)

        ReturnValue = self.sim.env.now + self.modified_duration
        return ReturnValue

    # operation time times multiplier (also known as factor k)
    def x_operation_times_DD(self, order):

        self.duration = order.process_time_cumulative * self.sim.policy_panel.DD_factor_K_value + \
                        self.sim.policy_panel.DD_mean_operation_time_value*2
        self.modified_duration = self.shiftcalender_forward(self.sim.env.now, self.duration)

        ReturnValue = self.sim.env.now + self.modified_duration
        return ReturnValue

    # Calculation of operational due dates for all operations of an order-----------------------------------------------
    # basic value + work content
    def work_content_and_allowance_ODD(self, order):

        self.basic_value = self.sim.policy_panel.DD_basic_value_allowance/(len(order.routing_sequence)+1)
        self.due_date = order.production_due_date

        for WC in reversed(order.routing_sequence):
            self.end = self.due_date
            order.operation_due_date[WC] = self.due_date
            self.duration = (order.process_time[WC] + self.basic_value)

            self.modified_duration = self.shiftcalender_reverse(self.end, self.duration)
            order.operation_planned_input_date[WC] = self.end - self.modified_duration
            self.due_date = order.operation_planned_input_date[WC]

        order.planned_release_date = self.due_date

        return order.planned_release_date

    # mean work content + inter-operation time (see e.g. Wiendahl 2019)
    def work_content_mean_times_ODD(self, order): #ZUE + ZDF

        self.due_date = order.production_due_date

        for WC in reversed(order.routing_sequence):
            self.end = self.due_date
            order.operation_due_date[WC] = self.due_date
            self.duration = order.process_time[WC] + self.sim.policy_panel.DD_mean_operation_time_value

            self.modified_duration = self.shiftcalender_reverse(self.end, self.duration)
            order.operation_planned_input_date[WC] = self.end - self.modified_duration
            self.due_date = order.operation_planned_input_date[WC]

        order.planned_release_date = self.due_date

        return order.planned_release_date

    # mean operation time (see e.g. Wiendahl 2019)
    def mean_operation_times_ODD(self, order):

        self.due_date = order.production_due_date

        for WC in reversed(order.routing_sequence):
            self.end = self.due_date
            order.operation_due_date[WC] = self.due_date
            self.duration = self.sim.policy_panel.DD_mean_operation_time_value

            self.modified_duration = self.shiftcalender_reverse(self.end, self.duration)
            order.operation_planned_input_date[WC] = self.end - self.modified_duration
            self.due_date = order.operation_planned_input_date[WC]

        order.planned_release_date = self.due_date

        return order.planned_release_date

    # x-times operation time (see e.g. Wiendahl 2019)
    def x_operation_times_ODD(self, order):
        """
        allocate due date to order by total work content
        :param order:
        :return: Due Date value
        """

        self.due_date = order.production_due_date

        for WC in reversed(order.routing_sequence):
            self.end = self.due_date
            order.operation_due_date[WC] = self.due_date
            self.duration = self.sim.policy_panel.DD_factor_K_value * order.process_time[WC]

            self.modified_duration = self.shiftcalender_reverse(self.end, self.duration)
            order.operation_planned_input_date[WC] = self.end - self.modified_duration
            self.due_date = order.operation_planned_input_date[WC]

        order.planned_release_date = self.due_date

        return order.planned_release_date

    # Priority recalculation modules (MODD)-----------------------------------------------------------------------------
    def MODD_load_control(self, queue_list, work_center):

        # get the orders from the queue
        for i, order_queue in enumerate(queue_list):
            result_MODD = max(
                (self.sim.env.now + order_queue[0].process_time[work_center]),
                order_queue[0].ODDs[work_center]
            )
            order_queue[0].dispatching_priority[work_center] = result_MODD
            order_queue[1] = result_MODD
        return queue_list

    # Priority recalculation modules (Random)---------------------------------------------------------------------------
    def randomize_dispatching(self, queue_list, work_center):
        # get the orders from the queue
        for i, order_queue in enumerate(queue_list):
            order_queue[0].dispatching_priority[work_center] = self.random_generator2.random()
            order_queue[1] = order_queue[0].dispatching_priority[work_center]
        return queue_list

    # Calculation of Modified_Capacity_Slack----------------------------------------------------------------------------
    def modified_capacity_slack(self, order):
        """
        Modified capacity slack pool rule: combines capacity slack with PRD to switch between low and peak load periods
        see thurer et al., 2015 for more details
        """

        order.pool_priority = 0

        # compute capacity slack corrected
        for WC in order.routing_sequence:
            current_load = self.sim.model_panel.RELEASED[WC] - self.sim.model_panel.PROCESSED[WC]
            load_contribution_i = (order.process_time[WC] / (order.routing_sequence.index(WC) + 1))
            order.pool_priority += load_contribution_i / (self.sim.policy_panel.release_norm - current_load)
        # adjust for routing length
        order.pool_priority = order.pool_priority / len(order.routing_sequence)

        # load switching mechanism of modified capacity slack
        if order.pool_priority < 0:
            order.pool_priority = 10000 - order.pool_priority  # if negative

        # categorize orders
        if order.planned_release_date - self.sim.policy_panel.check_period > self.sim.env.now:
            # not urgent
            order.pool_priority = order.planned_release_date
        else:
            # urgent
            order.pool_priority = -10000000000 + order.pool_priority
        return

    # Calculation of Adjusted Distribution Times if Truncation is enabled-----------------------------------------------
    def calculate_adjusted_process_time(self):
        if self.sim.model_panel.PROCESS_TIME_DISTRIBUTION in ["2_erlang", "exponential"]:
            if self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "2_erlang":
                reader = csv.reader(open("Distributions/Erlang-2.csv", 'r'), delimiter=",")
            else:
                reader = csv.reader(open("Distributions/Erlang-1.csv", 'r'), delimiter=",")
            next(reader)
            mean_adj: float = 1000
            mean_true: float = 1000
            diff: float = 1000
            truncation = self.sim.model_panel.TRUNCATION_POINT_PROCESS_TIME

            for data in reader:
                if int(data[2]) == truncation:
                    if abs(self.sim.model_panel.MEAN_PROCESS_TIME - float(data[1])) < diff:
                        mean_true = float(data[1])
                        mean_adj = float(data[0])
                        diff = abs(self.sim.model_panel.MEAN_PROCESS_TIME - float(data[1]))

            if diff > 0.1:
                raise Exception('Truncation Value does not exist in lookup table or configuration not feasible!')

            self.sim.model_panel.MEAN_PROCESS_TIME_ADJ = mean_adj
            self.sim.model_panel.MEAN_PROCESS_TIME_TRUE = mean_true

            if self.sim.model_panel.MEAN_TIME_BETWEEN_ARRIVAL_Recalculation:
                self.sim.model_panel.MEAN_TIME_BETWEEN_ARRIVAL= \
                    self.arrival_time_calculator(
                        wc_and_flow_config=self.sim.model_panel.WC_AND_FLOW_CONFIGURATION,
                        manufacturing_floor_layout=self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT,
                        aimed_utilization=self.sim.model_panel.AIMED_UTILIZATION,
                        mean_process_time=self.sim.model_panel.MEAN_PROCESS_TIME_TRUE,
                        number_of_machines=self.sim.model_panel.NUMBER_OF_MACHINES)

            return

        elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION in ["lognormal", "normal", "weibull"]:
            if self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "lognormal":
                reader = csv.reader(open("Distributions/LogNormal.csv", 'r'), delimiter=",")
            elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "normal":
                reader = csv.reader(open("Distributions/Normal.csv", 'r'), delimiter=",")
            else:
                reader = csv.reader(open("Distributions/Weibull.csv", 'r'), delimiter=",")
            next(reader)
            mean_adj: float = 1000
            mean_true: float = 1000
            std_adj: float = 1000
            std_true: float = 1000
            diff: float = 1000

            truncation = self.sim.model_panel.TRUNCATION_POINT_PROCESS_TIME

            for data in reader:
                if int(data[4]) == truncation:
                    if abs(self.sim.model_panel.MEAN_PROCESS_TIME - float(data[2])) + abs(self.sim.model_panel.STD_DEV_PROCESS_TIME - float(data[3])) < diff:
                        mean_true = float(data[2])
                        mean_adj = float(data[0])
                        std_true = float(data[3])
                        std_adj = float(data[1])
                        diff = abs(self.sim.model_panel.MEAN_PROCESS_TIME - float(data[2])) + abs(self.sim.model_panel.STD_DEV_PROCESS_TIME - float(data[3]))

            if diff > 0.1:
                raise Exception('Truncation Value does not exist in lookup table or configuration not feasible!')

            self.sim.model_panel.MEAN_PROCESS_TIME_ADJ = mean_adj
            self.sim.model_panel.MEAN_PROCESS_TIME_TRUE = mean_true
            self.sim.model_panel.STD_DEV_PROCESS_TIME_ADJ = std_adj
            self.sim.model_panel.STD_DEV_PROCESS_TIME_TRUE = std_true

            if self.sim.model_panel.MEAN_TIME_BETWEEN_ARRIVAL_Recalculation:
                self.sim.model_panel.MEAN_TIME_BETWEEN_ARRIVAL= \
                    self.arrival_time_calculator(
                        wc_and_flow_config=self.sim.model_panel.WC_AND_FLOW_CONFIGURATION,
                        manufacturing_floor_layout=self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT,
                        aimed_utilization=self.sim.model_panel.AIMED_UTILIZATION,
                        mean_process_time=self.sim.model_panel.MEAN_PROCESS_TIME_TRUE,
                        number_of_machines=self.sim.model_panel.NUMBER_OF_MACHINES)

            return

        elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "constant":
            return
        else:
            raise Exception("no valid process time distribution selected")
