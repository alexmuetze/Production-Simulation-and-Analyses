"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Defines all necessary functions needed for order release
- Manages the pool and the sequence of release according to pool sequencing decision
- Comprises the starvation avoidance logic for release rules where applicable
- Manages the caculation of finished load and the load accounts
"""

from operator import itemgetter
import random
import math

# Main class of order release control-----------------------------------------------------------------------------------
class ReleaseControl(object):
    """
    class containing all needed functions to control the release of orders into production
    """

    # Initialisation of Release-Procedure-------------------------------------------------------------------------------
    def __init__(self, simulation):
        self.sim = simulation
        self.pool = self.sim.model_panel.ORDER_POOL
        self.default_wc = self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT[0]  # Variable for Central Procedures
        self.random_generator = random.Random()
        self.random_generator.seed(999997)

    # Function for the Order Pool in front of the release decision------------------------------------------------------
    def order_pool(self, order):
        """
        the the pool with flow items before the process
        """

        # set the priority for each job---------------------------------------------------------------------------------
        """
        Release sequencing rules 
            - FCFS
            - PRD
            - SPT_Total
            - SPT
            - PRD
            - FRO
            - MODCS
            - EDD
            - SLACK
            - "Once_Random", "True_Random"
        """

        seq_priority = None
        if self.sim.policy_panel.sequencing_rule == "FCFS":
            seq_priority = order.entry_time
        elif self.sim.policy_panel.sequencing_rule == "SPT":
            seq_priority = list(order.process_time.values())[0]
        elif self.sim.policy_panel.sequencing_rule == "SPT_Total":
            seq_priority = order.process_time_cumulative
        elif self.sim.policy_panel.sequencing_rule == "PRD":
            seq_priority = order.planned_release_date
        elif self.sim.policy_panel.sequencing_rule == "FRO":
            seq_priority = len(order.routing_sequence)
        elif self.sim.policy_panel.sequencing_rule == "MODCS":
            seq_priority = order.pool_priority
        elif self.sim.policy_panel.sequencing_rule == "EDD":
            seq_priority = order.due_date
        elif self.sim.policy_panel.sequencing_rule == "SLACK":
            seq_priority = order.due_date - order.process_time_cumulative
        elif self.sim.policy_panel.sequencing_rule in ["Once_Random", "True_Random"]:
            seq_priority = self.random_generator.random()
        else:
            raise Exception('No valid pool sequencing rule selected')

        # put each job in the pool
        order_list = [order, seq_priority, 1]
        self.pool.put(order_list)

        # Release mechanisms--------------------------------------------------------------------------------------------
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

        if self.sim.policy_panel.release_control_method == "LUMS_COR":
            work_center = order.routing_sequence[0]
            self.continuous_trigger_activation(work_center=work_center)
        elif self.sim.policy_panel.release_control_method in ["LOOR", "WLC_LD"]:
            # is an own function, therefore no call at this stage
            return
        elif self.sim.policy_panel.release_control_method == "DOOR":
            self.sim.env.process(self.DOOR(order=order))
        elif self.sim.policy_panel.release_control_method in ["WLC_AL", "WLC_CAL"]:
            print(self.sim.policy_panel.release_control_method)
        elif self.sim.policy_panel.release_control_method in ["CONWIP", "CONLOAD"]:
            self.CONWIP_or_CONLOAD()
        elif self.sim.policy_panel.release_control_method in ["UBR", "UBRLB", "UBR_Trigger", "UBRLB_Trigger",
                                                              "Immediate"]:
            self.continuous_release()
            if self.sim.policy_panel.release_control_method in ["UBR_Trigger", "UBRLB_Trigger"]:
                work_center = order.routing_sequence[0]
                self.continuous_trigger_activation(work_center=work_center)
        else:
            raise Exception("no valid release method selected")
        return

    # Input for Starvation Avoidance --> How many Orders are at the workcenter or in processing at the moment?----------
    def control_queue_empty(self, work_center):
        """
        controls if the queue is empty
        """

        in_system = len(self.sim.model_panel.ORDER_QUEUES[work_center].items) + \
                    len(self.sim.model_panel.MANUFACTURING_FLOOR[work_center].users)
        return in_system <= self.sim.policy_panel.continuous_trigger

    # Remove Order from Pool in case of release ------------------------------------------------------------------------
    def remove_from_pool(self, release_now):
        """
        remove flow item from the pool
        """

        # create a variable that is equal to a item that is removed from the pool
        release_now[2] = 0
        # sort the queue
        self.pool.items.sort(key=itemgetter(2))

        # count release
        self.sim.data_continous_run.order_release_counter += 1
        order = release_now[0]
        self.sim.data_continous_run.load_release_counter += order.process_time_cumulative

        # remove flow item from pool
        self.pool.get()

    # Shiftcalender for Timeout of Periodic Release--------------------------------------------------------------------
    def shiftcalender(self, duration):

        self.sc_duration = duration
        if not self.sim.policy_panel.capacityflexibilty:
            yield self.sim.env.timeout(self.sc_duration)
            return 0

        else:
            self.startoffset = 0

            if self.sim.env.now % 10 >= self.sim.model_panel.STANDARDCAPACITY:
                self.startoffset = 10 - (self.sim.env.now % 10)
            yield self.sim.env.timeout(self.startoffset)

            while self.sc_duration > 0:
                if self.sim.env.now % 10 + self.sc_duration < self.sim.model_panel.STANDARDCAPACITY:
                    yield self.sim.env.timeout(self.sc_duration)
                    self.sc_duration = 0
                    continue

                if self.sim.env.now % 10 < self.sim.model_panel.STANDARDCAPACITY:
                    self.sc_duration -= (self.sim.model_panel.STANDARDCAPACITY - (self.sim.env.now % 10))
                    yield self.sim.env.timeout(10 - (self.sim.env.now % 10))
            return 0

    # Function for the periodic release of orders into production-------------------------------------------------------
    def periodic_release(self):
        """
        Workload Control Periodic release using aggregate or corrected aggregate load. See workings in Land 2004.
        """
        while True:
            # deactivate release period, ensure dispatching
            self.sim.process.starvation_dispatch()
            yield self.sim.env.process(self.shiftcalender(self.sim.policy_panel.check_period))

            # set the list of released orders
            release_now = []

            # sequence the orders currently in the pool
            if self.sim.policy_panel.sequencing_rule in ["MODCS", "True_Random"]:
                self.update_pool_sequence()
            self.pool.items.sort(key=itemgetter(1))

            # contribute the load from each item in the pool
            for i, order_list in enumerate(self.pool.items):
                order = order_list[0]

                # handling release time limit if activated
                if self.sim.policy_panel.release_time_limit == True:
                    if order.planned_release_date > self.sim.env.now + \
                            self.sim.general_functions.shiftcalender_forward(self.sim.env.now,
                                                                             self.sim.policy_panel.release_time_limit_value):
                        continue

                # contribute the load from for each workstation
                for WC in order.routing_sequence:
                    # Corrected Aggregate Load
                    if self.sim.policy_panel.release_control_method in ["LUMS_COR", "WLC_CAL"]:
                        self.sim.model_panel.RELEASED[WC] += order.process_time[WC] / (
                                    order.routing_sequence.index(WC) + 1)
                    # Aggregate Load
                    elif self.sim.policy_panel.release_control_method == "WLC_AL":
                        self.sim.model_panel.RELEASED[WC] += order.process_time[WC]
                order.release = True

                # the new load is compared to the norm
                for WC in order.routing_sequence:
                    if self.sim.model_panel.RELEASED[WC] - self.sim.model_panel.PROCESSED[WC] \
                            > self.sim.policy_panel.release_norm:
                        order.release = False

                # if a norm has been violated the job is not released and the contributed load set back
                if not order.release:
                    for WC in order.routing_sequence:
                        if self.sim.policy_panel.release_control_method in ["LUMS_COR", "WLC_CAL"]:
                            self.sim.model_panel.RELEASED[WC] -= order.process_time[WC] / (
                                        order.routing_sequence.index(WC) + 1)
                        elif self.sim.policy_panel.release_control_method == "WLC_AL":
                            self.sim.model_panel.RELEASED[WC] -= order.process_time[WC]
                        else:
                            raise Exception("No valid configuration for periodic release!")

                # the released orders are collected into a list for release
                if order.release:
                    # Orders for released are collected into a list
                    release_now.append(order_list)
                    # the orders are send to the process, but dispatch after release period
                    self.sim.process.put_in_queue_pool(order=order, dispatch=False)

            # the released orders are removed from the pool using the remove from pool method
            for _, jobs in enumerate(release_now):
                self.sim.release_control.remove_from_pool(release_now=jobs)

    # Function for the continous release of orders into Production------------------------------------------------------
    def continuous_release(self):
        """
        Workload Control: continuous release using corrected aggregate load. See workings in Fernandes et al., 2017
        """

        # reset the list of released orders
        release_now = []

        # sequence the orders currently in the pool
        if self.sim.policy_panel.sequencing_rule == ["MODCS", "True_Random"]:
            self.update_pool_sequence()
        self.pool.items.sort(key=itemgetter(1))

        # contribute the load from each item in the pool
        for i, order_list in enumerate(self.pool.items):

            order = order_list[0]
            underload = 0
            overload = 0

            if self.sim.policy_panel.release_time_limit == True:
                if order.planned_release_date > self.sim.env.now + \
                        self.sim.general_functions.shiftcalender_forward(self.sim.env.now,
                                                                         self.sim.policy_panel.release_time_limit_value):
                    continue

            # contribute the load from for each workstation
            if not self.sim.policy_panel.release_control_method == "Immediate":
                for WC in order.routing_sequence:
                    self.sim.model_panel.RELEASED[WC] += order.process_time[WC] / (order.routing_sequence.index(WC) + 1)
                    order.release = True
                # The new load is compared to the norm
                for WC in order.routing_sequence:
                    if self.sim.model_panel.RELEASED[WC] - self.sim.model_panel.PROCESSED[WC] \
                            > self.sim.policy_panel.release_norm:
                        order.release = False
            else:
                order.release = True

            # control for norm violation
            if self.sim.policy_panel.release_control_method in ["UBR", "UBR_Trigger"]:
                if not order.release:
                    for WC in order.routing_sequence:
                        self.sim.model_panel.RELEASED[WC] -= order.process_time[WC] / (
                                order.routing_sequence.index(WC) + 1)

            elif self.sim.policy_panel.release_control_method in ["UBRLB", "UBRLB_Trigger"]:
                if not order.release:
                    for WC in order.routing_sequence:
                        difference = self.sim.model_panel.RELEASED[WC] - self.sim.model_panel.PROCESSED[WC]
                        load_norm_difference = difference - self.sim.policy_panel.release_norm
                        underload += abs(min(load_norm_difference, 0))
                        overload += max(load_norm_difference, 0)
                    if overload < underload:
                        for WC in order.routing_sequence:
                            self.sim.model_panel.RELEASED[WC] -= order.process_time[WC] / (
                                    order.routing_sequence.index(WC) + 1)
                    else:
                        order.release = True

            # the released orders are collected into a list for release
            if order.release:
                # Orders for released are collected into a list
                release_now.append(order_list)
                # The orders are send to the process
                self.sim.process.put_in_queue_pool(order=order)

        # the released orders are removed from the pool using the remove from pool method
        for _, jobs in enumerate(release_now):
            self.sim.release_control.remove_from_pool(release_now=jobs)
        return

    # Trigger to avoid shortages / starvation avoidance trigger---------------------------------------------------------
    def continuous_trigger(self, work_center):
        """
        Workload Control: continuous release using aggregate load. See workings in Thürer et al, 2014.
        Part of LUMS COR
        """

        # empty the release list
        trigger = 0

        # sort orders in the pool
        if self.sim.policy_panel.sequencing_rule in ["MODCS", "True_Random"]:
            self.update_pool_sequence()
        self.pool.items.sort(key=itemgetter(1))

        # control if there is any order available for the starving work centre from all items in the pool
        for i, order_list in enumerate(self.pool.items):
            order = order_list[0]
            # stop if one orders is already released
            if trigger > 0:
                return

            if self.sim.policy_panel.starvation_avoidance_time_limit == True:
                if order.planned_release_date < self.sim.env.now - \
                        self.sim.general_functions.shiftcalender_reverse(self.sim.env.now,
                                                                         self.sim.policy_panel.starvation_avoidance_limit_value):
                    continue

            # control if can be released
            if order.routing_sequence[0] == work_center:
                trigger += 1

                # contribute the load to the workload measures
                for WC in order.routing_sequence:
                    if self.sim.policy_panel.release_control_method in ["LUMS_COR", "WLC_CAL"]:
                        self.sim.model_panel.RELEASED[WC] += order.process_time[WC] / (
                                order.routing_sequence.index(WC) + 1)
                    elif self.sim.policy_panel.release_control_method == "WLC_AL":
                        self.sim.model_panel.RELEASED[WC] += order.process_time[WC]
                order.release = True

                for WC in order.routing_sequence:
                    if self.sim.model_panel.RELEASED[WC] - self.sim.model_panel.PROCESSED[WC] \
                            > self.sim.policy_panel.release_norm:
                        order.release = False

                # if a norm has been violated the job is not released and the contributed load set back
                if not order.release:
                    for WC in order.routing_sequence:
                        if self.sim.policy_panel.release_control_method in ["LUMS_COR", "WLC_CAL"]:
                            self.sim.model_panel.RELEASED[WC] -= order.process_time[WC] / (
                                        order.routing_sequence.index(WC) + 1)
                        elif self.sim.policy_panel.release_control_method == "WLC_AL":
                            self.sim.model_panel.RELEASED[WC] -= order.process_time[WC]
                        else:
                            raise Exception("No valid configuration for periodic release!")

                # if an order turned out to be released, it is send to be removed from the pool
                if order.release:
                    order.continuous_trigger = True
                    # Send the order to the starting work centre
                    self.sim.process.put_in_queue_pool(order=order)
                    # release order from the pool
                    self.sim.release_control.remove_from_pool(release_now=order_list)

        return

    # Check whether the queue of the work system is empty---------------------------------------------------------------
    def continuous_trigger_activation(self, work_center):
        """
        feedback mechanism for continuous release
        """

        # control the if the amount of orders in or before the work centre is equal or less than one
        if self.control_queue_empty(work_center=work_center):
            self.continuous_trigger(work_center=work_center)

    # Function for ConWIP or ConLoad Release----------------------------------------------------------------------------
    def CONWIP_or_CONLOAD(self):
        """
        Constant Work In Process. Fixed amount of flow units in the system, see Spearman et al. (1998)
        Constant Workload. Fixed amount of workload in the system
        """

        # reset the list of released order
        release_now = []

        # sequence the orders currently in the pool
        if self.sim.policy_panel.sequencing_rule in ["MODCS", "True_Random"]:
            self.update_pool_sequence()

        self.pool.items.sort(key=itemgetter(1))

        # Contribute the load from each item in the pool
        for i, order_list in enumerate(self.pool.items):
            order = order_list[0]

            if self.sim.policy_panel.release_time_limit == True:
                if order.planned_release_date > self.sim.env.now + \
                        self.sim.general_functions.shiftcalender_forward(self.sim.env.now,
                                                                         self.sim.policy_panel.release_time_limit_value):
                    continue

            # contribute the load
            if self.sim.policy_panel.release_control_method == "CONWIP":
                self.sim.model_panel.RELEASED[self.default_wc] += 1
            elif self.sim.policy_panel.release_control_method == "CONLOAD":
                self.sim.model_panel.RELEASED[self.default_wc] += order.process_time_cumulative
            order.release = True

            # the new load is compared to the norm
            if self.sim.model_panel.RELEASED[self.default_wc] - self.sim.model_panel.PROCESSED[self.default_wc] > \
                    self.sim.policy_panel.release_norm:
                order.release = False

            # if a norm has been violated the job is not released and the contributed load set back
            if not order.release:
                if self.sim.policy_panel.release_control_method == "CONWIP":
                    self.sim.model_panel.RELEASED[self.default_wc] -= 1
                elif self.sim.policy_panel.release_control_method == "CONLOAD":
                    self.sim.model_panel.RELEASED[self.default_wc] -= order.process_time_cumulative

            # the released orders are collected into a list for release
            if order.release:
                # orders for released are collected into a list
                release_now.append(order_list)
                # the orders are send to the manufacturing
                self.sim.process.put_in_queue_pool(order=order)

        # the released orders are removed from the pool using the remove from pool method
        for _, jobs in enumerate(release_now):
            self.sim.release_control.remove_from_pool(release_now=jobs)
        return

    # Function for Recall of Continous OR-------------------------------------------------------------------------------
    def release_time_limit_trigger(self):

        while True:

            if self.sim.env.now % 10 > self.sim.model_panel.STANDARDCAPACITY:
                yield self.sim.env.timeout(10 - self.sim.env.now % 10)


            if self.sim.policy_panel.release_control_method in ["UBR", "UBRLB", "UBR_Trigger", "UBRLB_Trigger",
                                                                "Immediate"]:
                self.continuous_release()

            yield self.sim.env.timeout(1)

            if self.sim.env.now >= (self.sim.model_panel.WARM_UP_PERIOD + self.sim.model_panel.RUN_TIME) \
                    * self.sim.model_panel.NUMBER_OF_RUNS:
                break

    # Function for DOOR-------------------------------------------------------------------------------------------------
    def DOOR(self, order):

        order.release_slack = order.planned_release_date - self.sim.env.now
        if order.release_slack > 0:
            yield self.sim.env.timeout(order.release_slack)

        self.sim.process.put_in_queue_pool(order=order)

        self.pool.items.sort(key=itemgetter(1))
        self.pool.get()

        # count release
        self.sim.data_continous_run.order_release_counter += 1
        self.sim.data_continous_run.load_release_counter += order.process_time_cumulative

        return

    # Function for LOOR-------------------------------------------------------------------------------------------------
    def LOOR(self):
        """
        Load oriented order release, was ist mit LOOR mit Starvation Avoidance?
        """

        while True:
            # deactivate release period, ensure dispatching
            self.sim.process.starvation_dispatch()
            yield self.sim.env.process(self.shiftcalender(self.sim.policy_panel.check_period))

            # set the list of released orders
            release_now = []

            # sequence the orders currently in the pool
            if self.sim.policy_panel.sequencing_rule in ["MODCS", "True_Random"]:
                self.update_pool_sequence()
            self.pool.items.sort(key=itemgetter(1))

            # contribute the load from each item in the pool
            for i, order_list in enumerate(self.pool.items):
                order = order_list[0]

                if self.sim.policy_panel.release_time_limit == True:
                    if order.planned_release_date > self.sim.env.now + \
                            self.sim.general_functions.shiftcalender_forward(self.sim.env.now,
                                                                             self.sim.policy_panel.release_time_limit_value):
                        continue

                order.release = True

                # load account is compared to the load limit
                for WC in order.routing_sequence:
                    if self.sim.model_panel.LOAD_ACCOUNT[WC] > self.sim.policy_panel.LOOR_Load_Limit[WC]:
                        order.release = False
                        break

                if not order.release:
                    continue

                # store variable for the mathematical product for discounting the process times
                v = []

                # fill the load accounts with the process time of each order in order_routing_sequence
                for i, WC in enumerate(order.routing_sequence):
                    if i == 0:
                        self.sim.model_panel.LOAD_ACCOUNT[WC] += order.process_time[WC]

                    elif i > 0:
                        v.append(self.sim.policy_panel.LOOR_Abfa[order.routing_sequence[i]])
                        self.sim.model_panel.LOAD_ACCOUNT[WC] += order.process_time[WC] * math.prod(v)

                # the released orders are collected into a list for release
                if order.release:
                    # orders for released are collected into a list
                    release_now.append(order_list)
                    # the orders are send to the process
                    self.sim.process.put_in_queue_pool(order=order, dispatch=False)

            # the released orders are removed from the pool using the remove from pool method
            for _, jobs in enumerate(release_now):
                self.sim.release_control.remove_from_pool(release_now=jobs)

    # Function fpr WLC_LD-----------------------------------------------------------------------------------------------
    def WLC_LD(self):
        """
        Workload-Control acc. to Loedding (i.e., LOOR without statistical load approach)
        """

        while True:
            # deactivate release period, ensure dispatching
            self.sim.process.starvation_dispatch()
            yield self.sim.env.process(self.shiftcalender(self.sim.policy_panel.check_period))

            # set the list of released orders
            release_now = []

            # sequence the orders currently in the pool
            if self.sim.policy_panel.sequencing_rule in ["MODCS", "True_Random"]:
                self.update_pool_sequence()
            self.pool.items.sort(key=itemgetter(1))

            # contribute the load from each item in the pool
            for i, order_list in enumerate(self.pool.items):
                order = order_list[0]

                if self.sim.policy_panel.release_time_limit == True:
                    if order.planned_release_date > self.sim.env.now + \
                            self.sim.general_functions.shiftcalender_forward(self.sim.env.now,
                                                                             self.sim.policy_panel.release_time_limit_value):
                        continue

                order.release = True

                # load account is compared to the load limit
                for WC in order.routing_sequence:
                    if self.sim.model_panel.LOAD_ACCOUNT[WC] > self.sim.policy_panel.LOOR_Load_Limit[WC]:
                        order.release = False
                        break

                if not order.release:
                    continue

                # fill the load accounts with the process time of each order in order_routing_sequence
                for i, WC in enumerate(order.routing_sequence):
                    self.sim.model_panel.LOAD_ACCOUNT[WC] += order.process_time[WC]

                # the released orders are collected into a list for release
                if order.release:
                    # orders for released are collected into a list
                    release_now.append(order_list)
                    # the orders are send to the process
                    self.sim.process.put_in_queue_pool(order=order, dispatch=False)

            # the released orders are removed from the pool using the remove from pool method
            for _, jobs in enumerate(release_now):
                self.sim.release_control.remove_from_pool(release_now=jobs)

    # Function for the calculation of finished load---------------------------------------------------------------------
    def finished_load(self, order, work_center):
        """
        add the processed load and trigger continuous release if required.
        """

        # remove load
        if self.sim.policy_panel.release_control_method == "CONWIP" and len(order.routing_sequence) == 0:
            self.sim.model_panel.PROCESSED[self.default_wc] += 1
            self.CONWIP_or_CONLOAD()

        if self.sim.policy_panel.release_control_method == "CONLOAD" and len(order.routing_sequence) == 0:
            self.sim.model_panel.PROCESSED[self.default_wc] += order.process_time_cumulative
            self.CONWIP_or_CONLOAD()

        if not self.sim.policy_panel.release_control_method in ["CONLOAD", "CONWIP"]:
            if self.sim.policy_panel.release_control_method in ["WLC_AL", "DOOR", "LOOR"]:
                self.sim.model_panel.PROCESSED[work_center] += order.process_time[work_center]
            else:
                self.sim.model_panel.PROCESSED[work_center] += order.process_time[work_center] / (
                            order.routing_sequence_data.index(work_center) + 1)

        # continuous trigger LUMS COR
        if self.sim.policy_panel.release_control_method == "LUMS_COR":
            self.sim.release_control.continuous_trigger_activation(work_center=work_center)

        # continuous release methods
        if self.sim.policy_panel.release_control_method in ["UBR", "UBRLB", "UBR_Trigger", "UBRLB_Trigger",
                                                            "Immediate"]:
            self.continuous_release()
            if self.sim.policy_panel.release_control_method in ["UBR_Trigger", "UBRLB_Trigger"]:
                self.sim.release_control.continuous_trigger_activation(work_center=work_center)

        # remove load WLC_LD
        if self.sim.policy_panel.release_control_method == "WLC_LD":
            self.sim.model_panel.LOAD_ACCOUNT[work_center] -= order.process_time[work_center]

        # recalculation in case LOOR
        if self.sim.policy_panel.release_control_method == "LOOR":
            self.sim.model_panel.LOAD_ACCOUNT[work_center] -= order.process_time[work_center]
            d = []
            a = []

            for i, WC in enumerate(order.routing_sequence):
                if i == 0:
                    d.append(self.sim.policy_panel.LOOR_Abfa[order.routing_sequence[i]])
                    self.sim.model_panel.LOAD_ACCOUNT[WC] -= order.process_time[WC] * math.prod(d)
                    self.sim.model_panel.LOAD_ACCOUNT[WC] += order.process_time[WC]
                elif i > 0:
                    d.append(self.sim.policy_panel.LOOR_Abfa[order.routing_sequence[i]])
                    self.sim.model_panel.LOAD_ACCOUNT[WC] -= order.process_time[WC] * math.prod(d)
                    a.append(self.sim.policy_panel.LOOR_Abfa[order.routing_sequence[i]])
                    self.sim.model_panel.LOAD_ACCOUNT[WC] += order.process_time[WC] * math.prod(a)

        return

    # Function to update the pool sequence every time called------------------------------------------------------------
    def update_pool_sequence(self):
        for i, order_list in enumerate(self.pool.items):
            if self.sim.policy_panel.sequencing_rule == "MODCS":
                self.sim.general_functions.modified_capacity_slack(order=order_list[0])
                order_list[1] = order_list[0].pool_priority
            elif self.sim.policy_panel.sequencing_rule == "True_Random":
                order_list[1] = self.random_generator.random()
            else:
                raise Exception("pool sequence rule cannot be updated")
