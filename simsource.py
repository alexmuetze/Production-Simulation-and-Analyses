"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Generates the orders in the input of the system
- Provides several opportunities to influence the deviation and load in the input
- Supports a shiftcalender function
- Supports load-smoothing in the input to deviate from full exponential input (Markov)
"""

from flowitem import Order
import numpy as np
import pandas as pd
import random

# Modelling the source--------------------------------------------------------------------------------------------------
class Source(object):
    """
    class containing the source function, generating the load for the production system
    """
    # Initialisation of the Source Procedure----------------------------------------------------------------------------
    def __init__(self, simulation, stationary=True):

        self.sim = simulation
        self.stationary = stationary
        self.random_generator = random.Random()
        self.random_generator.seed(999999)
        self.numpy_generator= np.random.default_rng(999999)
        self.mean_time_between_arrivals = self.sim.model_panel.MEAN_TIME_BETWEEN_ARRIVAL
        self.variation_arrival = self.sim.model_panel.VARIATION_ARRIVAL

        # planned-value update in case of backlog control in order to control based on "actual" known plan
        if self.sim.policy_panel.backlog_control:
            self.update_df_planned_values()

        # in case of load-smoothing take process_time from created lookup table (full covariance)
        if self.sim.model_panel.SMOOTHING:
            self.smoothing_lookup = self.process_time_lockup()

        # enabling non_stationary control influencing interarrival_time
        if not self.stationary:
            self.non_stationary = NonStationaryControl(simulation=self.sim, source=self)

            # activate non-stationary manager
            self.sim.env.process(self.non_stationary.non_stationary_manager())

    # Function needed to compare planned values with actual values for backlog control----------------------------------
    def update_df_planned_values(self):

        if self.sim.env.now == 0:
            df = pd.DataFrame([0.0],columns=['Time_Sys_Input'])
            df["Order_Sys_Input"] = 0
            df["Load_Sys_Input"] = 0.0
            df["Time_Sys_Output"] = 0.0
            df["Order_Sys_Output"] = 0
            df["Load_Sys_Output"] = 0.0

            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                df[f"Time_WC_Input_{work_centre}"] = 0.0
                df[f"Order_WC_Input_{work_centre}"] = 0
                df[f"Load_WC_Input_{work_centre}"] = 0.0
                df[f"Time_WC_Output_{work_centre}"] = 0.0
                df[f"Order_WC_Output_{work_centre}"] = 0
                df[f"Load_WC_Output_{work_centre}"] = 0.0

            self.df_planned_values = df

        else:
            df2 = self.df_planned_values
            df = pd.DataFrame([self.sim.env.now], columns=['Time_Sys_Input'])
            df["Order_Sys_Input"] = df2.loc[:, f"Order_Sys_Input"].sum()
            df["Load_Sys_Input"] = df2.loc[:, f"Load_Sys_Input"].sum()
            df["Time_Sys_Output"] =self.sim.env.now
            df["Order_Sys_Output"] = df2.loc[:, f"Order_Sys_Output"].sum()
            df["Load_Sys_Output"] = df2.loc[:, f"Load_Sys_Output"].sum()

            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                df[f"Time_WC_Input_{work_centre}"] = self.sim.env.now
                df[f"Order_WC_Input_{work_centre}"] = df2.loc[:, f"Order_WC_Input_{work_centre}"].sum()
                df[f"Load_WC_Input_{work_centre}"] = df2.loc[:, f"Load_WC_Input_{work_centre}"].sum()
                df[f"Time_WC_Output_{work_centre}"] = self.sim.env.now
                df[f"Order_WC_Output_{work_centre}"] = df2.loc[:, f"Order_WC_Output_{work_centre}"].sum()
                df[f"Load_WC_Output_{work_centre}"] = df2.loc[:, f"Load_WC_Output_{work_centre}"].sum()

            self.df_planned_values = df

        return

    # Generation of Random Arrival--------------------------------------------------------------------------------------
    def generate_random_arrival_exp(self):
        i = 1
        inter_arrival_time = 0
        self.initial_pause: bool=True

        while True:

            # half of warm-up period is used to empty the system in order to get stable single runs
            if self.initial_pause == True:
                yield self.sim.env.timeout(1 / 2 * self.sim.model_panel.WARM_UP_PERIOD)
                self.initial_pause=False

            # create an order object and give it a name
            order = Order(simulation=self.sim)
            order.entry_time = self.sim.env.now
            order.name = ('Order%07d' % i)
            order.identifier = i
            order.interarrival_time = inter_arrival_time
            cv = self.variation_arrival

            # count input
            self.sim.data_continous_run.order_input_counter += 1
            self.sim.data_continous_run.load_input_counter += order.process_time_cumulative

            # release control
            if self.sim.policy_panel.release_control:
                # ORR release
                self.sim.release_control.order_pool(order=order)
            else:
                self.sim.process.put_in_queue_pool(order=order)

            # calculate interarrival time, cv value in order to reduce variation | use of gamma distribution
            # take non_stationary_time if not stationary
            if not self.stationary:
                a= 1 / (cv ** 2)
                inter_arrival_time=self.numpy_generator.gamma(a, self.non_stationary.current_mean_between_arrival) / a

            # if smoothing is activated get right inter_arrival time
            elif self.sim.model_panel.SMOOTHING:
                inter_arrival_time = 0
                for j, smoothing_list in enumerate (self.smoothing_lookup):
                    inter_arrival_time = smoothing_list[1]
                    if order.process_time_cumulative <= smoothing_list[0]:
                        break

            # calculation of inter_arrival time in dependence of cv
            else:
                if cv == 0:
                    inter_arrival_time = self.mean_time_between_arrivals
                else:
                    a = 1 / (cv **2)
                    inter_arrival_time = self.numpy_generator.gamma(a, self.mean_time_between_arrivals) / a

            # save interarrival_time as time after order entry
            order.interarrival_time_after = inter_arrival_time

            # pause source process for inter_arrival time using shiftcalender
            yield self.sim.env.process(self.shiftcalender(inter_arrival_time))
            i += 1
            if self.sim.env.now >= (self.sim.model_panel.WARM_UP_PERIOD + self.sim.model_panel.RUN_TIME) \
                    * self.sim.model_panel.NUMBER_OF_RUNS:
                break

    # Shiftcalender Function for Simulating Capacity Flexiblity---------------------------------------------------------
    def shiftcalender(self, inter_arrival_time):

        self.interarrivaltime = inter_arrival_time

        if not self.sim.policy_panel.capacityflexibilty:
            yield self.sim.env.timeout(self.interarrivaltime)
            return 0

        else:
            # Start auf Schichtkalender rechnen!
            while self.interarrivaltime > 0:
                if self.sim.env.now % 10 + self.interarrivaltime < self.sim.model_panel.STANDARDCAPACITY:
                    yield self.sim.env.timeout(self.interarrivaltime)
                    self.interarrivaltime = 0
                    continue

                if self.sim.env.now % 10 < self.sim.model_panel.STANDARDCAPACITY:
                    self.interarrivaltime -= (self.sim.model_panel.STANDARDCAPACITY - (self.sim.env.now % 10))
                    yield self.sim.env.timeout(10 - (self.sim.env.now % 10))
            return 0

    # Build up of a lookup-table for Load-Smoothing---------------------------------------------------------------------
    def process_time_lockup(self):

        scan_markup = 20
        iterations = self.sim.model_panel.RUN_TIME * scan_markup
        process_time_lockup = []
        inter_arrival_time_lockup = []
        cv=self.sim.model_panel.VARIATION_ARRIVAL

        for i in range(1, iterations):
            order = Order(simulation=self.sim)
            process_time_lockup.append(order.process_time_cumulative)
            del order
        process_time_lockup.sort()

        for i in range(1, iterations):
            a=1 / (cv ** 2)
            inter_arrival_time=self.numpy_generator.gamma(a, self.mean_time_between_arrivals) / a
            inter_arrival_time_lockup.append(inter_arrival_time)

        inter_arrival_time_lockup.sort()
        return list(zip(process_time_lockup, inter_arrival_time_lockup))

# NonStationaryControl - Further Development needed for adaption to shiftcalender and lookup-list-----------------------
class NonStationaryControl(object):
    """
    class containing all functions for nonstationary input control
    """
    def __init__(self, simulation, source):

        self.sim = simulation
        self.source = source
        self.random_generator = random.Random()
        self.random_generator.seed(1)

        self.plot_trajectory = False
        self.print_info = True
        self.force_run_time = True
        self.save_non_stationary_database = True

        # stationary params as input
        self.current_utilization = self.sim.model_panel.AIMED_UTILIZATION
        self.current_cv = self.sim.model_panel.VARIATION_ARRIVAL
        self.current_mean_between_arrival = self.source.mean_time_between_arrivals
        self.current_pattern = "warm_up"

        # trail patterns available
        """
        -   stationary      (stationary period)
        -   systematic      (increase of CV)
        -   stratification  (decrease of CV)
        -   combi           (monotonic increase, decrease of CV and utilization)
        -   upward_trend    (monotonic increase in utilization)
        -   downward_trend  (monotonic decrease in utilization)
        -   upward_shift    (sudden increase in utilization)
        -   downward_shift  (sudden decrease in utilization)
        """

        # initiate hard coded pattern
        pattern_sequence, total_time = self.hard_code_pattern()
        self.pattern_sequence = pattern_sequence
        self.total_time = total_time
        if self.force_run_time:
            self.sim.model_panel.RUN_TIME = self.total_time

        # print plot
        if self.plot_trajectory:
            self.plot_system(show_emperical_trajectory=False, save=True)
        # save database
        if self.save_non_stationary_database:
            self.save_non_stationary_list()

    # Non stationary pattern -------------------------------------------------------------------------------------------
    # hard coded pattern using the available functions
    def hard_code_pattern(self):
        pattern_sequence = list()
        time = 1000
        total_time = 0
        # 1: stationary period
        pattern_sequence.append(self.pattern_stationary(time=time, utilization=0.9, cv=1))
        total_time += time
        # 2: increase cv
        pattern_sequence.append(self.pattern_systematic_or_stratification(time=1000,
                                                                      utilization = 0.9,
                                                                      cv_from= 0.1,
                                                                      cv_to = 2,
                                                                      interval=100)
                                )
        total_time += 1000

        pattern_sequence.append(self.pattern_systematic_or_stratification(time=1000,
                                                                          utilization=0.9,
                                                                          cv_from=2,
                                                                          cv_to=0.2,
                                                                          interval=100)
                                )
        total_time+=1000

        # 3: shift utilization down
        pattern_sequence.append(self.pattern_upward_or_downward_shift(time=1000,
                                                                      utilization_from=0.95,
                                                                      utilization_till=0.8,
                                                                      interval=2000 / 2,
                                                                      cv=1)
                                )
        total_time += 1000
        # 2: decrease utilization period
        pattern_sequence.append(self.pattern_upward_or_downward_trend(time=1000,
                                                                      utilization_from=0.8,
                                                                      utilization_till=0.9,
                                                                      interval=100,
                                                                      cv=2)
                                )
        total_time += 1000

        # 3: stationary period
        pattern_sequence.append(self.pattern_combi(time=5000, utilization_from=0.7, utilization_till=1.2, cv_from=0.2, cv_to=10, interval=250))
        total_time += time

        return pattern_sequence, total_time

    # Non stationary control -------------------------------------------------------------------------------------------
    def non_stationary_manager(self):

        # import lists
        time_list, utilization_list, cv_list, pattern_name_list = \
            self.time_pattern_list(patterns_sequence=self.pattern_sequence)

        # import params
        number_of_patterns = len(time_list) - 1
        index = 0
        run_time = 0
        previous_time = 0
        run_number = 1

        # loop to manage the run dynamics
        while True:
            # get step params
            time = time_list[index] + run_time - previous_time
            previous_time = time_list[index] + run_time
            self.current_pattern = pattern_name_list[index]
            index += 1

            # change mean time between arrival
            if self.current_utilization != utilization_list[index]:
                self.current_utilization = utilization_list[index]
                self.current_mean_between_arrival = \
                    self.sim.general_functions.arrival_time_calculator(
                        wc_and_flow_config=self.sim.model_panel.WC_AND_FLOW_CONFIGURATION,
                        manufacturing_floor_layout=self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT,
                        aimed_utilization=self.current_utilization,
                        mean_process_time=self.sim.model_panel.MEAN_PROCESS_TIME,
                        number_of_machines=self.sim.model_panel.NUMBER_OF_MACHINES)

                if self.print_info:
                    print(f"\n\tMean time between arrival: {self.current_mean_between_arrival}\n"
                          f"\tAimed utilization {self.current_utilization}\n"
                          f"\tCurrent pattern {self.current_pattern}\n")

            if self.current_cv != cv_list[index]:
                self.current_cv = cv_list[index]
                self.source.variation_arrival = cv_list[index]

                if self.print_info:
                    print(f"\n\tVariationkoeffizient: {self.source.variation_arrival}\n"
                          f"\tAimed utilization {self.current_utilization}\n"
                          f"\tCurrent pattern {self.current_pattern}\n")


            # yield until the next change
            yield self.sim.env.timeout(time)

            # break loop if all patterns are visited
            if number_of_patterns == index:
                index = 0
                run_time += self.sim.model_panel.RUN_TIME
                previous_time = int(self.sim.env.now) - self.sim.model_panel.WARM_UP_PERIOD * run_number
                run_number += 1

                if self.print_info:
                    print(f"reset experiment trajectory {self.sim.env.now}")
        return

    # pattern list translate -------------------------------------------------------------------------------------------
    def time_pattern_list(self, patterns_sequence, cv=1):
        """
        Method that makes a list with the pattern of the non-stationary system
        """

        # initialze lists
        time_list = list()
        utilization_list = list()
        cv_list = list()
        pattern_name_list = list()

        # loop params
        time = self.sim.model_panel.WARM_UP_PERIOD
        utilization = self.current_utilization
        cv = self.source.variation_arrival

        # loop for each pattern
        """
        - 0: pattern name              <string>
        - 1: time span:                <float>
        - 2: utilization [from, to]    <list, int>
        - 3: utilization change by     <float>
        - 4: cv [from, to]             <list, int>
        - 5: cv change by              <float>
        - 6: interval                  <int>                 
        """

        for i, pattern_list in enumerate(patterns_sequence):
            # check if utilization will change
            if pattern_list[0] == 'combi':
                number_of_changes=int(pattern_list[1] / pattern_list[6])
                utilization=pattern_list[2][0]
                cv=pattern_list[4][0]
                for j in range(0, number_of_changes):
                    time_list.append(time)
                    time+=pattern_list[6]
                    utilization_list.append(utilization)
                    utilization+=pattern_list[3]
                    cv_list.append(cv)
                    cv+=pattern_list[5]
                    pattern_name_list.append(pattern_list[0])

            elif pattern_list[3] != 'na':
                if pattern_list[0][-5:-1] == "shif":
                    utilization = pattern_list[2][0]
                    time_shift = [time, time + pattern_list[6] - 0.000001,
                                  time + pattern_list[6],
                                  time + pattern_list[1]
                                  ]
                    time += pattern_list[1]
                    time_list.extend(time_shift)
                    utilization_shift = [utilization,
                                         utilization,
                                         (utilization - pattern_list[3]),
                                         (utilization - pattern_list[3])
                                         ]
                    utilization_list.extend(utilization_shift)
                    utilization = utilization - pattern_list[3]
                    cv_list_shift = [pattern_list[4][0]] * 4
                    cv_list.extend(cv_list_shift)
                    utilization_shift = [pattern_list[0]] * 4
                    pattern_name_list.extend(utilization_shift)

                elif pattern_list[0][-5:-1] == "tren":
                    number_of_changes = int(pattern_list[1] / pattern_list[6])
                    utilization=pattern_list[2][0]
                    for j in range(0, number_of_changes):
                        time_list.append(time)
                        time += pattern_list[6]
                        utilization_list.append(utilization)
                        utilization += pattern_list[3]
                        cv_list.append(pattern_list[4][0])
                        pattern_name_list.append(pattern_list[0])

            # check if cv will change
            elif pattern_list[5] != 'na':
                number_of_changes = int(pattern_list[1] / pattern_list[6])
                cv = pattern_list[4][0]
                for j in range(0, number_of_changes):
                    time_list.append(time)
                    time += pattern_list[6]
                    utilization_list.append(pattern_list[2][0])
                    cv_list.append(cv)
                    cv += pattern_list[5]
                    pattern_name_list.append(pattern_list[0])

            # stationary period
            else:
                time_list.append(time)
                time += pattern_list[1]
                utilization=pattern_list[2][0]
                utilization_list.append(pattern_list[2][0])
                cv = pattern_list[4][0]
                cv_list.append(pattern_list[4][0])
                pattern_name_list.append(pattern_list[0])

        return time_list, utilization_list, cv_list, pattern_name_list

    # pattern list -----------------------------------------------------------------------------------------------------
    def pattern_stationary(self, time, utilization, cv):
        """
        returns list with the pattern of the stationary events
        :param time:
        :param utilization:
        :param cv:
        :return: return_list

        Key for the list
        - 0: pattern name              <string>
        - 1: time span:                <float>
        - 2: utilization [from, to]    <list, int>
        - 3: utilization change by     <int>
        - 4: cv [from, to]             <list, int>
        - 5: cv change by              <float>
        - 6: interval                  <int>
        """
        # setup params
        return_list = list()
        # pattern name
        return_list.append("stationary")
        # time span
        return_list.append(time)
        # utilization
        util_list = [utilization, utilization]
        return_list.append(util_list)
        # utilization change by
        return_list.append('na')
        # cv list
        cv_list = [cv, cv]
        return_list.append(cv_list)
        # cv change by
        return_list.append('na')
        # interval
        return_list.append(time)
        return return_list

    def pattern_systematic_or_stratification(self, time, utilization, cv_from, cv_to, interval):
        """
        method that makes an systematic or stratification pattern
        :param time:
        :param utilization:
        :param cv_from:
        :param cv_to:
        :param interval:
        :return:
        """
        if time < interval:
            print("interval is larger than time, default assumption interval = time / 10")
            interval = time / 10
        # setup params
        return_list = list()
        # pattern name
        if cv_from > cv_to:
            return_list.append("stratification")
        else:
            return_list.append("systematic")
        # time span
        return_list.append(time)
        # utilization
        util_list = [utilization, utilization]
        return_list.append(util_list)
        # utilization change by
        return_list.append('na')
        # cv list
        cv_list = [cv_from, cv_from]
        return_list.append(cv_list)
        # cv change by
        cv_delta = (cv_to - cv_from) / (time / interval)
        return_list.append(cv_delta)
        # interval
        return_list.append(interval)
        return return_list

    def pattern_combi(self, time, utilization_from, utilization_till, cv_from, cv_to, interval):
        """
        method that makes an cyclic pattern
        :param time:
        :param utilization_from:
        :param utilization_till:
        :param cv_min:
        :param cv_max:
        :param interval:
        :return:
        """
        if time < interval:
            print("interval is larger than time, default assumption interval = time / 10")
            interval = time / 10
        # setup params
        return_list = list()
        # pattern name
        return_list.append("combi")
        # time span
        return_list.append(time)
        # utilization
        util_list=[utilization_from, utilization_till]
        return_list.append(util_list)
        # utilization change by
        utilization_delta=(utilization_till - utilization_from) / (time / interval)
        return_list.append(utilization_delta)
        # cv list
        cv_list=[cv_from, cv_from]
        return_list.append(cv_list)
        # cv change by
        cv_delta=(cv_to - cv_from) / (time / interval)
        return_list.append(cv_delta)
        # interval
        return_list.append(interval)
        return return_list

    def pattern_upward_or_downward_trend(self, time, utilization_from, utilization_till, cv, interval):
        """
        method that makes an upward or downward trend pattern
        :param time:
        :param utilization_from:
        :param utilization_till:
        :param cv:
        :param interval:
        :return:
        """
        if time < interval:
            print("interval is larger than time, default assumption interval = time / 10")
            interval = time / 10
        # setup params
        return_list = list()
        if utilization_from > utilization_till:
            return_list.append("downward_trend")
        else:
            return_list.append("upward_trend")
        # time span
        return_list.append(time)
        # utilization
        util_list = [utilization_from, utilization_till]
        return_list.append(util_list)
        # utilization change by
        utilization_delta = (utilization_till - utilization_from) / (time / interval)
        return_list.append(utilization_delta)
        # cv list
        cv_list = [cv, cv]
        return_list.append(cv_list)
        # cv change by
        return_list.append('na')
        # interval
        return_list.append(interval)
        return return_list

    def pattern_upward_or_downward_shift(self, time, utilization_from, utilization_till, cv, interval):
        """
        method that makes an upward or downward shift pattern
        :param time:
        :param utilization_from:
        :param utilization_till:
        :param cv:
        :param interval:
        :return:
        """
        if time < interval:
            print("interval is larger than time, default assumption interval = time / 10")
            interval = time / 10
        # setup params
        return_list = list()
        if utilization_from > utilization_till:
            return_list.append("downward_shift")
        else:
            return_list.append("upward_shift")
        # time span
        return_list.append(time)
        # utilization
        util_list = [utilization_from, utilization_till]
        return_list.append(util_list)
        # utilization change by
        utilization_delta = (utilization_from - utilization_till)
        return_list.append(utilization_delta)
        # cv list
        cv_list = [cv, cv]
        return_list.append(cv_list)
        # cv change by
        return_list.append('na')
        # interval
        interval = 0.5 * time
        return_list.append(interval)
        return return_list

    # utilities --------------------------------------------------------------------------------------------------------
    def plot_system(self, show_emperical_trajectory=True, save=False):
        """
        method that plots the non-stationary patern
        :param show_emperical_trajectory:
        :param save:
        :return:
        """
        import matplotlib.pyplot as plt

        # get data
        time_list, utilization_list, cv_list, pattern_name_list = self.time_pattern_list(
            patterns_sequence=self.pattern_sequence)
        # put data in dataframe
        if show_emperical_trajectory:
            # get empirical values
            empirical_list, new_time_list, new_utilization_list = \
                self.pseudo_random_generator(time_list=time_list, utilization_list=utilization_list)

            # moving average of utilization
            empirical_list = self.moving_average(mva_list=empirical_list, n=500)

            df = pd.DataFrame({'x': new_time_list,
                               'y_1': new_utilization_list,
                               "empirical": empirical_list})
        else:
            df = pd.DataFrame({'x': time_list, 'y_1': utilization_list})

        if show_emperical_trajectory:
            # add to the plot
            plt.plot('x', 'empirical', data=df, linestyle='-', color="blue", linewidth=1, alpha=0.4)
        # finnish the plot
        # Make a plot to visualize the results
        plt.plot('x', 'y_1', data=df, linestyle='-', color="black", linewidth=2)
        plt.title("Non Stationary Trajectory")
        plt.xlabel("time")
        plt.ylabel("Utilization")
        if save:
            plt.savefig('non_stationary_trajectory.png', dpi=96)
        # manipulate
        plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
        plt.show()
        return

    def pseudo_random_generator(self, time_list, utilization_list, start_time=0):
        """

        :param time_list:
        :param utilization_list:
        :param start_time:
        :return:
        """
        # setup params
        index = 0
        loop_time = start_time
        current_mean_between_arrival = self.current_mean_between_arrival

        # looping lists
        emperical_list = list()
        new_time_list = list()
        new_utilization_list = list()

        # loop
        while True:
            inter_arrival_time = self.random_generator.expovariate(1 / current_mean_between_arrival)
            utilization = 1 - (inter_arrival_time / self.sim.model_panel.MEAN_PROCESS_TIME)
            # utilization += 1

            if loop_time >= time_list[index]:
                # control if loop needs to be broken
                if loop_time > time_list[len(time_list) - 1] + time_list[0]:
                    break
                elif loop_time > time_list[len(time_list) - 1]:
                    current_utilization = utilization_list[index - 1]
                else:
                    # correct index
                    index += 1
                    # change mean time between arrival
                    current_utilization = utilization_list[index]

                current_mean_between_arrival = \
                    self.sim.general_functions.arrival_time_calculator(
                        wc_and_flow_config=self.sim.model_panel.WC_AND_FLOW_CONFIGURATION,
                        manufacturing_floor_layout=self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT,
                        aimed_utilization=current_utilization,
                        mean_process_time=self.sim.model_panel.MEAN_PROCESS_TIME,
                        number_of_machines=self.sim.model_panel.NUMBER_OF_MACHINES)

                # update the time
                loop_time += inter_arrival_time
            else:
                # update the time
                loop_time += inter_arrival_time

            # update lists
            emperical_list.append(utilization)
            new_time_list.append(loop_time)
            new_utilization_list.append(utilization_list[index])

        return emperical_list, new_time_list, new_utilization_list

    def moving_average(self, mva_list, n):
        """

        :param mva_list:
        :param n:
        :return:
        """
        cumsum, moving_aves = [0], []

        for i, x in enumerate(mva_list, 1):
            cumsum.append(cumsum[i - 1] + x)
            if i >= n:
                moving_ave = (cumsum[i] - cumsum[i - n]) / n
                # can do stuff with moving_ave here
                moving_aves.append(moving_ave)
            else:
                moving_aves.append(np.nan)

        return moving_aves

    def save_non_stationary_list(self, file_pattern=".csv"):
        """
        save the pattern list of the non-stationary events
        :param file_pattern:
        :return:
        """
        # import libraries
        import exp_manager as exp_manager
        # import lists
        time_list, utilization_list, cv_list, pattern_name_list = self.time_pattern_list(
            patterns_sequence=self.pattern_sequence)
        # put into a dataframe
        df = pd.DataFrame({'time': time_list,
                           'utilization': utilization_list,
                           'variationkoeffizient': cv_list,
                           "pattern name": pattern_name_list})
        # get path
        path = exp_manager.Experiment_Manager.get_directory("spam")

        # make file name
        path = path + "non_stationary_list" + file_pattern

        # save database
        exp_manager.Experiment_Manager.save_database_csv(self="spam", file=path, database=df)
        # print info
        print("#### non stationary control database saved ####")
        return