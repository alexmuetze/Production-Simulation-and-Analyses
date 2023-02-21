"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Controls the processing at every work center
- Controls the dispatching decision at every work centers pool
- Controls the capacity at every work center (backlog, WIP-control)
- Collects processing data
"""

import math
from operator import itemgetter
import numpy as np
import random
from math import trunc

# Class for the entire processing process, handling dispatching decisions and production--------------------------------
class Process(object):
    def __init__(self, simulation):
        """
        the process with discrete capacity sources
        """

        self.sim = simulation
        self.dispatching_rule: []
        self.duration: Dict[...] = {}
        self.process_time_in_process: Dict[...] = {}
        self.previous_entry_dict: Dict[...] = {}
        self.previous_finish_dict: Dict[...] = {}
        self.previous_release = 0.0

        for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
            self.duration[work_centre] = 0.0
            self.process_time_in_process[work_centre] = 0.0
            self.previous_entry_dict[work_centre] = 0.0
            self.previous_finish_dict[work_centre] = 0.0

        self.dispatching_rule = dict(
            zip(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT, self.sim.policy_panel.dispatching_rules))

        self.random_generator = random.Random()
        self.random_generator.seed(999998)
        self.WIP_control_list: list = []

    # Function for the entire pool handling, important for system_state_dispatching approach----------------------------
    def put_in_queue_pool(self, order, dispatch=True):
        """
        controls if an order can immediately be send to a capacity source or needs to be put in the queue.
        """

        # system state dispatching
        if order.first_entry and self.sim.policy_panel.dispatching_mode == 'system_state_dispatching':
            if self.sim.policy_panel.system_state_dispatching_version == "DRACO":
                # controlled release
                order.first_entry = True
                order.release = True
                order.location = "pool"

            # release order directly if no controlled release
            elif self.sim.policy_panel.system_state_dispatching_version == "FOCUS":
                # release order directly
                order.release_time = self.sim.env.now
                order.inter_release_time = self.scd_calculator(self.sim.env.now) - self.scd_calculator(self.previous_release)
                self.previous_release = self.sim.env.now
                order.pool_time = 0
                order.first_entry = False
                order.release = True
                order.location = "queue"
            else:
                raise Exception(f"invalid system state dispatching rule")

        # dispatching with priority rules
        elif order.first_entry and not self.sim.policy_panel.dispatching_mode == 'system_state_dispatching':
            order.release_time = self.sim.env.now
            order.inter_release_time = self.scd_calculator(self.sim.env.now) - self.scd_calculator(self.previous_release)
            self.previous_release = self.sim.env.now
            order.pool_time = order.release_time - order.entry_time
            order.first_entry = False
            order.release = True
            order.location = "queue"

        # get work centre
        work_centre = order.routing_sequence[0]

        # write Data
        self.sim.data_continous_run.order_input_counter_wc[work_centre] += 1
        self.sim.data_continous_run.load_input_counter_wc[work_centre] += order.process_time[work_centre]

        # put the order in queue or pool
        if order.location == "pool":  # put in the order pool in case of seperat authorization
            queue_item = self.flow_item(order=order, work_centre=work_centre)
            self.sim.model_panel.ORDER_POOLS[work_centre].put(queue_item)

        elif order.location == "queue":  # put in the order queue if pure dispatching is applied
            queue_item = self.flow_item(order=order, work_centre=work_centre)
            self.sim.model_panel.ORDER_QUEUES[work_centre].put(queue_item)
            order.operation_queue_entry_time[work_centre] = self.sim.env.now
            order.operation_inter_arrival_time[work_centre] = self.scd_calculator(self.sim.env.now) - self.scd_calculator(self.previous_entry_dict[work_centre])
            self.previous_entry_dict[work_centre] = self.sim.env.now

        # check if dispatching is needed
        if len(self.sim.model_panel.MANUFACTURING_FLOOR[work_centre].users) == 0:
            self.dispatch_order(work_center=work_centre)
        return

    # Trigger Dispatching in Case that a work center starves------------------------------------------------------------
    def starvation_dispatch(self):
        for i, work_centre in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            if len(self.sim.model_panel.MANUFACTURING_FLOOR[work_centre].users) == 0:
                self.dispatch_order(work_center=work_centre)
        return

    # Delete order from queue-------------------------------------------------------------------------------------------
    def pull_from_queue_pool(self, work_center, location):
        """
        removes an order from the queue
        """

        if location == "pool":
            # sort the pool
            self.sim.model_panel.ORDER_POOLS[work_center].items.sort(key=itemgetter(3))
            released_used = self.sim.model_panel.ORDER_POOLS[work_center].get()
        elif location == "queue":
            # sort the queue
            self.sim.model_panel.ORDER_QUEUES[work_center].items.sort(key=itemgetter(3))
            released_used = self.sim.model_panel.ORDER_QUEUES[work_center].get()

        else:
            raise Exception(f"{location} is an unknown location")

        return

    # Dispatching rule at work system-----------------------------------------------------------------------------------
    def flow_item(self, order, work_centre):
        """
        make a list of attributes that needs ot be put into the queue

        Key for the queue_item list
        0: order object
        1: dispatching priority
        2: next step
        3: pull index
        """

        """ 
            - FISFO         -- First-In-System-First-Out --> Order Number
            - FCFS          -- Input Date
            - SPT           -- Processing Time
            - MODD          -- (Deactivated)
            - EODD          -- Earliest Operation Due Date
            - Once_Random   -- Random Number Order Entry
            - True_Random   -- True Random every time a draw is made
            - SLACK         -- Slack Rule (DueDate minus Work Content)
        """

        # select dispatching mode
        if self.sim.policy_panel.dispatching_mode == 'priority_rule':
            if self.dispatching_rule[work_centre] == "FISFO":
                order.dispatching_priority[work_centre] = order.identifier
            elif self.dispatching_rule[work_centre] == "FCFS":
                order.dispatching_priority[work_centre] = self.sim.env.now
            elif self.dispatching_rule[work_centre] == "SPT":
                order.dispatching_priority[work_centre] = order.process_time[order.routing_sequence[0]]
            elif self.dispatching_rule[work_centre] in ["MODD", "EODD"]:
                order.dispatching_priority[work_centre] = order.operation_due_date[work_centre]
            elif self.dispatching_rule[work_centre] in ["Once_Random", "True_Random"]:
                order.dispatching_priority[work_centre] = self.random_generator.random()
            elif self.dispatching_rule[work_centre] == "SLACK":
                order.dispatching_priority[work_centre] = order.due_date - order.remaining_process_time
            else:
                raise Exception("no valid dispatching rule defined")

        # get queue list and return
        return self.get_flow_item(order=order, work_centre=work_centre)

    # Method to identify a queue item-----------------------------------------------------------------------------------
    def get_flow_item(self, order, work_centre):
        queue_item = [order,  # order object
                      order.dispatching_priority[work_centre],  # order priority
                      order.routing_sequence[0],  # next step
                      1,  # release from queue integer
                      order.pool_priority  # order pool priority
                      ]

        return queue_item

    # Get the order with the highest priority from queue and start the capacity process---------------------------------
    def dispatch_order(self, work_center):
        """
        Dispatch the order with the highest priority to the capacity source
        """

        # get new order for dispatching
        order_list, break_loop = self.get_most_urgent_order(work_centre=work_center)

        # no orders in queue
        if break_loop:
            return

        # get the order object
        order = order_list[0]
        self.pull_from_queue_pool(work_center=order.routing_sequence[0], location=order.location)
        order.process = self.sim.env.process(
            self.sim.process.capacity_process(order=order, work_centre=order.routing_sequence[0]))

        return

    # Process to identify the most urgent order, dependent on dispatching method recalculation of priority values-------
    def get_most_urgent_order(self, work_centre):
        """
        Update all priorities and routing steps in the queue
        """

        # setup params
        dispatching_list = list()

        # if there are no items in the queue and pool, return
        if len(self.sim.model_panel.ORDER_QUEUES[work_centre].items) == 0 \
                and len(self.sim.model_panel.ORDER_POOLS[work_centre].items) == 0:
            return None, True

        # update queue list depending on dispatching mode
        if self.sim.policy_panel.dispatching_mode == 'system_state_dispatching':
            pool_list = []
            if self.sim.policy_panel.system_state_dispatching_version == "DRACO":
                pool_list = self.sim.model_panel.ORDER_POOLS[work_centre].items.copy()

            # get queueing orders
            queue_list = self.sim.model_panel.ORDER_QUEUES[work_centre].items.copy()
            dispatching_options = self.sim.system_state_dispatching.dispatching_mode(queue_list=queue_list,
                                                                                     pool_list=pool_list,
                                                                                     work_centre=work_centre)
        elif self.sim.policy_panel.dispatching_mode == 'priority_rule':
            queue_list = self.sim.model_panel.ORDER_QUEUES[work_centre].items

            # update priorities if required
            if self.dispatching_rule[work_centre] == "MODD":
                dispatching_options = self.sim.general_functions.MODD_load_control(queue_list=queue_list,
                                                                                   work_center=work_centre)
            elif self.dispatching_rule[work_centre] == "True_Random":
                dispatching_options = self.sim.general_functions.randomize_dispatching(queue_list=queue_list,
                                                                                       work_center=work_centre)
            else:
                dispatching_options = queue_list

        else:
            raise Exception("dispatching mode undefined")

        # find order with highest impact or priority
        for i, order_list in enumerate(dispatching_options):
            # update routing step
            if len(order_list[0].routing_sequence) <= 1:
                order_list[2] = "NA"
            else:
                order_list[2] = order_list[0].routing_sequence[1]

            dispatching_list.append(order_list)

        # select order with highest impact or priority
        order = sorted(dispatching_list, key=itemgetter(1))[0]  # sort and pick with highest priority
        # set to zero to pull out of queue or pool
        order[3] = 0
        return order, False

    # Capacity Process / Production Process ----------------------------------------------------------------------------
    def capacity_process(self, order, work_centre):
        """
        The process with capacity sources
        """

        # if this first entry, then update data
        if order.first_entry:
            order.release_time = self.sim.env.now
            order.inter_release_time = self.scd_calculator(self.sim.env.now) - self.scd_calculator(self.previous_release)
            self.previous_release = self.sim.env.now
            order.pool_time = order.release_time - order.entry_time
            order.first_entry = False
            order.release = True

        if order.first_queue_dispatching_time is None:
            order.first_queue_dispatching_time = self.sim.env.now

        # update system state
        if self.sim.policy_panel.dispatching_mode == 'system_state_dispatching' \
                and self.sim.policy_panel.system_state_dispatching_authorization:
            # update authorization
            self.sim.system_state_dispatching.prior_update_a_ju(order=order, work_centre=work_centre)

        # set dispatching params
        order.work_center_RQ = self.sim.model_panel.MANUFACTURING_FLOOR[work_centre]
        req = order.work_center_RQ.request(priority=order.dispatching_priority[work_centre])

        # start processing, update order state
        order.wc_state[work_centre] = "IN_PROCESS"

        # yield a request
        with req as req:
            yield req

            # Request is finished, order directly processed
            yield self.sim.env.process(self.operation(order, work_centre))
            # order is finished and released from the machine

        # update order state
        order.wc_state[work_centre] = "PASSED"
        order.location = "queue"

        # update system_state_dispatching method
        if self.sim.policy_panel.dispatching_mode == 'system_state_dispatching':
            if self.sim.policy_panel.system_state_dispatching_authorization:
                self.sim.system_state_dispatching.post_update_a_ju(order=order, work_centre=work_centre)

        # update the routing list to avoid re-entrance
        order.routing_sequence.remove(work_centre)

        # release control
        if self.sim.policy_panel.release_control:
            self.sim.release_control.finished_load(order=order, work_center=work_centre)

        # collect data
        self.data_collection_intermediate(order=order, work_center=work_centre)

        # next action for the order
        if len(order.routing_sequence) == 0:
            self.data_collection_final(order=order)
        else:
            # activate new release
            self.put_in_queue_pool(order=order)

        # next action for the work centre
        self.dispatch_order(work_center=work_centre)
        return

    # Modul to calculate production dates with shiftcalender------------------------------------------------------------
    def operation(self, order, work_centre):
        order.operation_start_time[work_centre] = self.sim.env.now
        if not self.sim.policy_panel.capacityflexibilty:
            yield self.sim.env.timeout(order.process_time[work_centre])
            return 0

        else:
            # calculate start date according to shiftcalender
            self.startoffset = 0
            self.duration[work_centre] = order.process_time[work_centre]
            self.process_time_in_process[work_centre] = self.duration[work_centre]

            if self.sim.env.now % 10 >= self.sim.model_panel.ACTUALCAPA[work_centre]:
                self.startoffset = 10 - (self.sim.env.now % 10)
            order.operation_start_time[work_centre] = self.sim.env.now + self.startoffset
            yield self.sim.env.timeout(self.startoffset)

            while self.duration[work_centre] > 0:
                if self.sim.env.now % 10 + self.duration[work_centre] <= self.sim.model_panel.LOWERBARRIER:
                    yield self.sim.env.timeout(self.duration[work_centre])
                    self.duration[work_centre] = 0
                    self.process_time_in_process[work_centre]= 0
                    continue

                if self.sim.env.now % 10 <= self.sim.model_panel.LOWERBARRIER:
                    self.duration[work_centre] -= (self.sim.model_panel.LOWERBARRIER - (self.sim.env.now % 10))
                    yield self.sim.env.timeout(self.sim.model_panel.LOWERBARRIER - (self.sim.env.now % 10))

                    if self.sim.model_panel.LOWERBARRIER == self.sim.model_panel.ACTUALCAPA[work_centre]:
                        yield self.sim.env.timeout(10 - self.sim.model_panel.LOWERBARRIER)
                        continue

                if self.sim.env.now % 10 + self.duration[work_centre] <= self.sim.model_panel.STANDARDCAPACITY:
                    yield self.sim.env.timeout(self.duration[work_centre])
                    self.duration[work_centre] = 0
                    self.process_time_in_process[work_centre] = 0
                    continue

                if self.sim.env.now % 10 < self.sim.model_panel.STANDARDCAPACITY:
                    self.duration[work_centre] -= (self.sim.model_panel.STANDARDCAPACITY - (self.sim.env.now % 10))
                    yield self.sim.env.timeout(self.sim.model_panel.STANDARDCAPACITY - (self.sim.env.now % 10))

                    if self.sim.model_panel.STANDARDCAPACITY == self.sim.model_panel.ACTUALCAPA[work_centre]:
                        yield self.sim.env.timeout(10 - self.sim.model_panel.STANDARDCAPACITY)
                        continue

                if self.sim.env.now % 10 + self.duration[work_centre] <= self.sim.model_panel.UPPERBARRIER:
                    yield self.sim.env.timeout(self.duration[work_centre])
                    self.duration[work_centre] = 0
                    self.process_time_in_process[work_centre] = 0
                    continue

                if self.sim.env.now % 10 < self.sim.model_panel.UPPERBARRIER:
                    self.duration[work_centre] -= (self.sim.model_panel.UPPERBARRIER - (self.sim.env.now % 10))
                    yield self.sim.env.timeout(self.sim.model_panel.UPPERBARRIER - (self.sim.env.now % 10))

                    if self.sim.model_panel.UPPERBARRIER == self.sim.model_panel.ACTUALCAPA[work_centre]:
                        yield self.sim.env.timeout(10 - self.sim.model_panel.UPPERBARRIER)
                        continue

            return

    # Data Collection for Continous DB + Other Data for Processing------------------------------------------------------
    def data_collection_intermediate(self, order, work_center):
        """
        Collect data between routing steps
        """

        # count release
        self.sim.data_continous_run.order_output_counter_wc[work_center] += 1
        self.sim.data_continous_run.load_output_counter_wc[work_center] += order.process_time[work_center]

        order.remaining_process_time -= order.process_time[work_center]
        order.operation_finish_time[work_center] = self.sim.env.now
        order.operation_inter_depature_time[work_center] = self.scd_calculator(self.sim.env.now) - self.scd_calculator(self.previous_finish_dict[work_center])
        self.previous_finish_dict[work_center] = self.sim.env.now

        order.operation_time_in_queue[work_center] = self.sim.general_functions.shiftcalender_duration(
            order.operation_queue_entry_time[work_center], order.operation_start_time[work_center])

        return

    #Detailled Data Acquisition-----------------------------------------------------------------------------------------
    def data_collection_final(self, order):
        """
        Collect data finished order
        """

        # the finishing time
        order.finishing_time = self.sim.env.now

        # General data collection
        self.sim.data_continous_run.order_output_counter += 1
        self.sim.data_continous_run.load_output_counter += order.process_time_cumulative
        self.sim.data.accumulated_process_time += order.process_time_cumulative

        # setup list
        df_list = list()

        if self.sim.model_panel.COLLECT_BASIC_DATA:
            df_list.append(order.identifier)
            df_list.append(self.sim.general_functions.shiftcalender_duration(order.entry_time, order.finishing_time))
            df_list.append(order.pool_time)
            df_list.append(self.sim.general_functions.shiftcalender_duration(order.release_time, order.finishing_time))
            df_list.append(order.finishing_time - order.due_date)
            df_list.append(max(0, (order.finishing_time - order.due_date)))
            df_list.append(max(0, self.heavenside(x=(order.finishing_time - order.due_date))))

        else:
            # put all the metrics into list
            df_list.append(order.identifier)  # Order Number
            df_list.append(self.scd_calculator(self.sim.env.now))  # Time in SCD
            df_list.append(order.continuous_trigger)  # Continous Triggered (y/n)
            df_list.append(order.process_time_cumulative)  # Cummalative Process Time

            # Basics + time measures------------------------------------------------------------------------------------
            df_list.append(self.scd_calculator(order.interarrival_time))
            df_list.append(self.scd_calculator(order.interarrival_time_after))
            df_list.append(self.scd_calculator(order.entry_time))
            df_list.append(self.scd_calculator(order.planned_entry_date))
            df_list.append(self.scd_calculator(order.release_time))
            df_list.append(order.inter_release_time)
            df_list.append(self.scd_calculator(order.planned_release_date))
            df_list.append(self.scd_calculator(order.pool_time))
            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                if work_centre in order.routing_sequence_data:
                    df_list.append(order.routing_sequence_data.index(work_centre))
                    df_list.append(order.process_time[work_centre])
                    df_list.append(order.operation_inter_arrival_time[work_centre])
                    df_list.append(self.scd_calculator(order.operation_queue_entry_time[work_centre]))
                    df_list.append(self.scd_calculator(order.operation_planned_input_date[work_centre]))
                    df_list.append(self.scd_calculator(order.operation_start_time[work_centre]))
                    df_list.append(self.scd_calculator(order.operation_finish_time[work_centre]))
                    df_list.append(order.operation_inter_depature_time[work_centre])
                    df_list.append(self.scd_calculator(order.operation_due_date[work_centre]))
                else:
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)
            df_list.append(self.scd_calculator(order.finishing_time))
            df_list.append(self.scd_calculator(order.production_due_date))
            df_list.append(self.scd_calculator(order.due_date))
            df_list.append(((self.sim.data.accumulated_process_time * 100) / len(
                self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT) / (
                                        self.sim.model_panel.RUN_TIME * self.sim.model_panel.STANDARDCAPACITY / 10)))

            # Duration Measures (Production and Operations)-------------------------------------------------------------
            df_list.append(self.scd_calculator(order.release_time) - self.scd_calculator(order.entry_time))
            df_list.append(self.scd_calculator(order.finishing_time) - self.scd_calculator(order.release_time))
            df_list.append(self.scd_calculator(order.finishing_time) - self.scd_calculator(order.entry_time))
            df_list.append(self.scd_calculator(order.planned_release_date) - self.scd_calculator(order.entry_time))
            df_list.append(self.scd_calculator(order.planned_release_date) - self.scd_calculator(order.planned_entry_date))
            df_list.append(self.scd_calculator(order.production_due_date) - self.scd_calculator(order.planned_release_date))
            df_list.append(
                self.scd_calculator(order.production_due_date) - self.scd_calculator(order.planned_entry_date))
            df_list.append(self.scd_calculator(order.production_due_date) - self.scd_calculator(order.entry_time))
            df_list.append(self.scd_calculator(order.due_date) - self.scd_calculator(order.entry_time))

            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                if work_centre in order.routing_sequence_data:
                    df_list.append(self.scd_calculator(order.operation_start_time[work_centre]) - self.scd_calculator(order.operation_queue_entry_time[work_centre]))
                    df_list.append(self.scd_calculator(order.operation_finish_time[work_centre]) - self.scd_calculator(order.operation_start_time[work_centre]))
                    df_list.append(self.scd_calculator(order.operation_finish_time[work_centre]) - self.scd_calculator(order.operation_queue_entry_time[work_centre]))
                    df_list.append(self.scd_calculator(order.operation_due_date[work_centre]) - self.scd_calculator(order.operation_planned_input_date[work_centre]))
                else:
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)

            # Deviation Measures - Production---------------------------------------------------------------------------
            df_list.append(
                self.scd_calculator(order.entry_time) - self.scd_calculator(order.planned_entry_date))
            df_list.append(
                self.scd_calculator(order.release_time) - self.scd_calculator(order.planned_release_date))
            df_list.append(
                self.scd_calculator(order.finishing_time) - self.scd_calculator(order.production_due_date))
            df_list.append(self.scd_calculator(order.finishing_time) - self.scd_calculator(order.due_date))
            df_list.append(
                (self.scd_calculator(order.finishing_time) - self.scd_calculator(order.production_due_date)) -
                (self.scd_calculator(order.release_time) - self.scd_calculator(order.planned_release_date)))
            df_list.append((self.scd_calculator(order.finishing_time) - self.scd_calculator(order.due_date)) -
                           (self.scd_calculator(order.entry_time) - self.scd_calculator(
                               order.planned_entry_date)))  # TAR_Kunde
            df_list.append(max(0, (self.scd_calculator(order.finishing_time) - self.scd_calculator(
                order.due_date))))
            df_list.append(max(0, (self.scd_calculator(order.finishing_time) - self.scd_calculator(
                order.production_due_date))))
            df_list.append(
                max(0, self.heavenside(x=(order.finishing_time - order.due_date))))
            df_list.append(
                max(0, self.heavenside(
                    x=(order.finishing_time - order.production_due_date))))

            # Deviation Measures - Operations---------------------------------------------------------------------------
            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                if work_centre in order.routing_sequence_data:
                    df_list.append(self.scd_calculator(order.operation_queue_entry_time[work_centre])
                                   - self.scd_calculator(order.operation_planned_input_date[work_centre]))
                    df_list.append(self.scd_calculator(order.operation_finish_time[work_centre])
                                   - self.scd_calculator(order.operation_due_date[work_centre]))
                    df_list.append((self.scd_calculator(order.operation_finish_time[work_centre]) -
                                    self.scd_calculator(order.operation_due_date[work_centre])) -
                                   (self.scd_calculator(order.operation_queue_entry_time[work_centre]) -
                                    self.scd_calculator(order.operation_planned_input_date[work_centre])))

                else:
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)

            # Slack Measures--------------------------------------------------------------------------------------------
            df_list.append(self.scd_calculator(order.due_date) - self.scd_calculator(order.entry_time))
            df_list.append(self.scd_calculator(order.production_due_date) - self.scd_calculator(order.entry_time))
            df_list.append(self.scd_calculator(order.production_due_date) - self.scd_calculator(order.release_time))

            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                if work_centre in order.routing_sequence_data:
                    df_list.append(
                        self.scd_calculator(order.production_due_date) - self.scd_calculator(order.operation_queue_entry_time[work_centre]))
                    df_list.append(
                        self.scd_calculator(order.production_due_date) - self.scd_calculator(order.operation_start_time[work_centre]))
                    df_list.append(
                        self.scd_calculator(order.production_due_date) - self.scd_calculator(order.operation_finish_time[work_centre]))

                else:
                    df_list.append(np.nan)
                    df_list.append(np.nan)
                    df_list.append(np.nan)

            df_list.append(self.scd_calculator(order.production_due_date) - self.scd_calculator(order.finishing_time))
            df_list.append(len(order.routing_sequence_data))
            df_list.append(order.routing_sequence_data)

        # save list if it is not empty
        if len(df_list) != 0:
            self.sim.data.append_run_list(result_list=df_list)
        return

    # Supportfunction---------------------------------------------------------------------------------------------------
    def heavenside(self, x):
        if x > 0:
            return 1
        return -1

    # Supportfunction---------------------------------------------------------------------------------------------------
    def scd_calculator(self, timevalue):

        if self.sim.policy_panel.capacityflexibilty:
            self.timevalue = timevalue
            self.scdfactor = self.sim.model_panel.STANDARDCAPACITY

            if self.timevalue >= 0:
                self.scdvalue = math.trunc(self.timevalue/10) + (self.timevalue % 10) / self.scdfactor

            else:
                self.timevalue = abs(self.timevalue)
                self.scdvalue = (math.trunc(self.timevalue/10) + (self.timevalue % 10) / self.scdfactor) * -1

        else:
            self.scdvalue = timevalue

        return self.scdvalue

    # Helpful functions, currently not in use---------------------------------------------------------------------------
    def current_workload(self):
        workload = 0
        for j, WC in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            # load being processed
            if len(self.sim.model_panel.MANUFACTURING_FLOOR[WC].users) != 0:
                # get info order directly processed
                processing_order = self.sim.model_panel.MANUFACTURING_FLOOR[WC].users[0]
                workload += processing_order.process_time[WC]
            # get queueing order info
            for i, queueing_order in enumerate(self.sim.model_panel.ORDER_QUEUES[WC].items):
                if not queueing_order[0].first_entry:
                    workload += queueing_order[0].process_time[WC]

        return workload

    def current_work_in_progress(self):
        WIP_virtual = 0
        WIP = 0
        for j, WC in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            # orders in progress
            WIP += len(self.sim.model_panel.MANUFACTURING_FLOOR[WC].users)
            WIP_virtual += len(self.sim.model_panel.MANUFACTURING_FLOOR[WC].users)
            # orders in queue
            WIP += len(self.sim.model_panel.ORDER_QUEUES[WC].items)
            WIP_virtual += len(self.sim.model_panel.ORDER_QUEUES[WC].items)
            # orders in pool
            WIP_virtual += len(self.sim.model_panel.ORDER_POOLS[WC].items)
        # orders in the singular order pool
        WIP_virtual += len(self.sim.model_panel.ORDER_POOL.items)
        return WIP, WIP_virtual

    @property

    # Backlog Control---------------------------------------------------------------------------------------------------
    def backlog_control(self):

        # create dictionary for calculation of backlog
        backlog: Dict[...] = {}

        # set backlog "0" at start
        for i, WC in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            backlog[WC] = 0

        while True:

            df = self.sim.source.df_planned_values

            # calculation at "5"
            if self.sim.env.now % 10 < 5:
                yield self.sim.env.timeout(5 - self.sim.env.now % 10)
            if self.sim.env.now % 10 > 5:
                yield self.sim.env.timeout(10 - self.sim.env.now % 10 + 5)

            # backlog calculation for each work center
            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                backlog[work_centre] = df.loc[(df[f'Time_WC_Output_{work_centre}'] <= self.sim.env.now), f"Load_WC_Output_{work_centre}"].sum() -\
                                       self.sim.data_continous_run.load_output_counter_wc[work_centre]

                # borders for capacity control
                if backlog[work_centre] > 16:
                    self.sim.model_panel.ACTUALCAPA[work_centre] = self.sim.model_panel.UPPERBARRIER
                    print("Overtime" + work_centre)
                elif backlog[work_centre] < -8:
                    self.sim.model_panel.ACTUALCAPA[work_centre] = self.sim.model_panel.LOWERBARRIER
                    print("Undertime" + work_centre)
                else:
                    self.sim.model_panel.ACTUALCAPA[work_centre] = self.sim.model_panel.STANDARDCAPACITY

            # timeout
            yield self.sim.env.timeout(10)

            if self.sim.env.now >= (self.sim.model_panel.WARM_UP_PERIOD + self.sim.model_panel.RUN_TIME) \
                    * self.sim.model_panel.NUMBER_OF_RUNS:
                break

    @property
    # WIP Control-------------------------------------------------------------------------------------------------------
    def WIP_control(self):

        # create empty lists
        liste: list = []
        workload: Dict[...]={}

        # set 0
        for i, WC in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            workload[WC]=0

        # calculation of the ideal minimum WIP as basis variable for calculation of control borders
        if self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "2_erlang":
            self.ideal_WIP = self.sim.model_panel.MEAN_PROCESS_TIME * 1.5
        elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "lognormal":
            self.ideal_WIP=self.sim.model_panel.MEAN_PROCESS_TIME * (
                    1 + (self.sim.model_panel.STD_DEV_PROCESS_TIME / self.sim.model_panel.MEAN_PROCESS_TIME) ** 2)
        elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "exponential":
            self.ideal_WIP = self.sim.model_panel.MEAN_PROCESS_TIME * 2
        elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "constant":
            self.ideal_WIP = self.sim.model_panel.MEAN_PROCESS_TIME
        elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "normal":
            self.ideal_WIP=self.sim.model_panel.MEAN_PROCESS_TIME * (
                        1 + (self.sim.model_panel.STD_DEV_PROCESS_TIME / self.sim.model_panel.MEAN_PROCESS_TIME) ** 2)
        elif self.sim.model_panel.PROCESS_TIME_DISTRIBUTION == "weibull":
            self.ideal_WIP=self.sim.model_panel.MEAN_PROCESS_TIME * (
                        1 + (self.sim.model_panel.STD_DEV_PROCESS_TIME / self.sim.model_panel.MEAN_PROCESS_TIME) ** 2)

        while True:

            # control decision at 5
            if self.sim.env.now % 10 < 5:
                yield self.sim.env.timeout(5 - self.sim.env.now % 10)
            if self.sim.env.now % 10 > 5:
                yield self.sim.env.timeout(10 - self.sim.env.now % 10 + 5)


            # calculation of control result for each work center
            for WC in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                workload[WC]=0

                if len(self.sim.model_panel.MANUFACTURING_FLOOR[WC].users) != 0:
                    # get info order directly processed
                    workload[WC]+= self.process_time_in_process[WC]/2

                # get queueing order info
                for i, queueing_order in enumerate(self.sim.model_panel.ORDER_QUEUES[WC].items):
                    if not queueing_order[0].first_entry:
                        workload[WC]+=queueing_order[0].process_time[WC]

                # definition of control borders
                if workload[WC] > 6 * self.ideal_WIP:
                    self.sim.model_panel.ACTUALCAPA[WC] = self.sim.model_panel.UPPERBARRIER
                    print("Overtime" + WC)
                elif workload[WC] < 1 * self.ideal_WIP:
                    self.sim.model_panel.ACTUALCAPA[WC] = self.sim.model_panel.LOWERBARRIER
                    print("Undertime" + WC)
                else:
                    self.sim.model_panel.ACTUALCAPA[WC] = self.sim.model_panel.STANDARDCAPACITY

            # WIP-control list
            if self.sim.warm_up == True:
                liste.append(trunc(self.scd_calculator(self.sim.env.now)))
                for WC in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                    liste.append(self.sim.model_panel.ACTUALCAPA[WC])
                self.WIP_control_list.append(liste)
                liste = []

            yield self.sim.env.timeout(10)

            if self.sim.env.now >= (self.sim.model_panel.WARM_UP_PERIOD + self.sim.model_panel.RUN_TIME) \
                    * self.sim.model_panel.NUMBER_OF_RUNS:
                break