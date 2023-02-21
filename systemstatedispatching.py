"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 0.1
"""

# Notice: Adaption necessary!!! --> Left in for further development options!

import numpy as np

class SystemStateDispatching(object):

    def __init__(self, simulation):
        # function params
        self.sim = simulation
        self.work_centre_layout = self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT

        """ system-state variables """
        # dispatching
        self.T_list = list()
        self.A_dict = dict()
        self.S_list = list()
        self.V_list = list()

        self.p_ij_min = 0
        self.p_ij_mean = 0
        self.p_ij_max = np.inf
        self.slack_min = 0
        self.slack_mean = 0
        self.slack_max = np.inf
        self.slack_opn_min = 0
        self.slack_opn_mean = 0
        self.slack_opn_max = np.inf

        # release
        self.activate_release_WIP_target = self.sim.policy_panel.system_state_dispatching_release
        if self.activate_release_WIP_target:
            self.WIP_target = self.sim.policy_panel.WIP_target
        self.current_queue_list = []
        self.current_pool_list = []
        self.number_of_orders_in_system = 0
        self.WIP = 0
        self.total_workload = 0

        # authorization
        self.activate_authorization_loop_target = self.sim.policy_panel.system_state_dispatching_authorization
        if self.activate_authorization_loop_target:
            self.loop_target_jk = self.sim.policy_panel.loop_WIP_target_jk
            self.a_ju = {}
            for j_prime in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                self.a_ju[j_prime] = {}
                for k_i_prime in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                    self.a_ju[j_prime][k_i_prime] = 0

        # control weights impact functions
        self.weights = [1] * len(self.sim.model_panel.function_names)
        self.functions_activated = dict()
        for i, name in enumerate(self.sim.model_panel.function_names):
            self.functions_activated[name] = self.weights[i]
        self.max_impact_list = [0] * len(self.weights)
        return

    def prior_update_a_ju(self, order, work_centre):
        # control if the loop is not the last routing step
        if len(order.routing_sequence) > 1:
            u = order.routing_sequence[1]
            self.a_ju[work_centre][u] += 1
        return

    def post_update_a_ju(self, order, work_centre):
        # control if the order is just arrived
        if order.routing_sequence_data.index(work_centre) != 0:
            # update a_ju cards
            j = order.routing_sequence_data[order.routing_sequence_data.index(work_centre) - 1]
            # check for errors
            if self.a_ju[j][work_centre] == 0:
                raise Exception("error, tried to remove from the overlapping loop while it cannot be there")
            # update a_ju
            self.a_ju[j][work_centre] -= 1
        return

    def dispatching_mode(self, queue_list, pool_list, work_centre):
        # remove impact values previous decision
        self.max_impact_list = [0] * len(self.weights)
        # update state variables
        self.current_queue_list = queue_list
        self.current_pool_list = pool_list
        self.update_system_state_variables(work_centre=work_centre)

        # get impact of each order in queue or pool
        queue_list.extend(pool_list)
        dispatching_options = queue_list # first queue, then order
        for i, order_item in enumerate(dispatching_options):
            projected_impact_list = self._get_weighted_impact(order_item=order_item, work_centre=work_centre)
            # attach impact to the order
            order_item[1] = sum(projected_impact_list)
            # find highest impact for this selection moment
            if sum(projected_impact_list) > sum(self.max_impact_list):
                self.max_impact_list = projected_impact_list
            # - 1 because simulation model minimizes
            order_item[1] = order_item[1] * -1
        return dispatching_options

    def _get_weighted_impact(self, order_item, work_centre):
        # params
        process_time = order_item[0].process_time[work_centre]
        process_times = order_item[0].process_time.copy()
        routing_list = order_item[0].routing_sequence.copy()
        slack = self.slack(
            d_i=order_item[0].due_date,
            t=self.sim.env.now,
            k=1,
            sum_p_ij=order_item[0].remaining_process_time
        )
        slack_opn = self.slack(
            d_i=order_item[0].due_date,
            t=self.sim.env.now,
            k=len(routing_list),
            sum_p_ij=order_item[0].remaining_process_time
        )

        # get projected impact values
        impact_list = []
        """
        release element
        """
        release_impact = []
        # workload target
        if self.activate_release_WIP_target:
            rho = 0
            # release (order in pool)
            if order_item[0].first_entry:
                # more orders in the system than cap
                if self.WIP < (2*self.WIP_target):
                    rho = 1 - (self.WIP / (2*self.WIP_target))
                else:
                    rho = 0
            # release (order in queue)
            if not order_item[0].first_entry:
                # more load in the system than target
                if self.WIP < (2*self.WIP_target):
                    rho = (self.WIP / (2*self.WIP_target))
                else:
                    rho = 1
            release_impact.append(rho)
        release_impact = sum([j * 1/len(release_impact) for j in release_impact])

        """
        production authorization 
        """
        authorization_impact = []
        if self.activate_authorization_loop_target:
            # control the next downstream step
            if len(routing_list) > 1:
                k_i = routing_list[1]
                a_jk = self.a_ju[work_centre][k_i]
                if a_jk <= self.loop_target_jk:
                    alpha = 1 - (a_jk / self.loop_target_jk)
                else:
                    alpha = 0
            # order moves to completion
            else:
                alpha = 1
            authorization_impact.append(alpha)
        authorization_impact = sum([j * 1 / len(authorization_impact) for j in authorization_impact])

        """
        dispatching element
        """
        dispatching_impact = []
        # obtain pi
        if self.functions_activated["pi"] > 0:
            pi_pij_impact = 1 - (process_time / self.p_ij_max)
            dispatching_impact.append(pi_pij_impact)
        # obtain xi
        if self.functions_activated["xi"] > 0:
            return_value = self.xi_idleness_impact(
                order=order_item[0],
                process_time=process_time,
                upstream_starvation_dict=self.A_dict
            )
            if return_value > 0:
                xi_starvation_impact = 1 - (return_value / self.p_ij_max)
            else:
                xi_starvation_impact = 0
            dispatching_impact.append(xi_starvation_impact)
        # obtain tau
        if self.functions_activated["tau"] > 0:
            if slack > 0:
                tau_slack_impact = 1 - (slack / self.slack_max)
            else:
                tau_slack_impact = 1
            dispatching_impact.append(tau_slack_impact)
        # obtain delta
        if self.functions_activated["delta"] > 0:
            if slack_opn > 0:
                delta_slack_per_operation_impact = 1 - (slack_opn / self.slack_opn_max)
            else:
                delta_slack_per_operation_impact = 1
            dispatching_impact.append(delta_slack_per_operation_impact)

        dispatching_impact = sum([j * 1/len(dispatching_impact) for j in dispatching_impact])

        # aggregate all impact functions into list
        if len([release_impact, authorization_impact]) > 1:
            projected_impact = [dispatching_impact * 1/2, release_impact * 1/4, authorization_impact * 1/4]
        else:
            projected_impact = [dispatching_impact * 1/3, release_impact * 1/3, authorization_impact * 1/3]
        return projected_impact

    def update_system_state_variables(self, work_centre):
        """ set system-state variables """
        # dispatching
        self.T_list = list()
        self.A_dict = dict()
        self.S_list = list()
        self.V_list = list()

        """ update all state variables """
        # get state information from orders currently in process
        self.WIP = 0
        for j, WC in enumerate(self.work_centre_layout):
            """ 
            release / authorization 
            """
            for i, pool_order in enumerate(self.sim.model_panel.ORDER_POOLS[WC].items):
                processing_order = pool_order[0]
                # pick remaining process time
                process_times = []
                for k, operation in enumerate(processing_order.routing_sequence):
                    process_times.append(processing_order.process_time[operation])
                # order params
                slack, slack_opn = self.get_dispatching_variables(order=processing_order)
                self.append_system_state_variables(
                    process_times=process_times,
                    slack=slack,
                    slack_opn=slack_opn
                )

            """ 
            dispatching 
            """
            # processing order
            if len(self.sim.model_panel.MANUFACTURING_FLOOR[WC].users) == 1:
                self.WIP += 1
                processing_order = self.sim.model_panel.MANUFACTURING_FLOOR[WC].users[0].self
                # pick remaining process time
                process_times = []
                for k, operation in enumerate(processing_order.routing_sequence):
                    process_times.append(processing_order.process_time[operation])

                # order params
                slack, slack_opn = self.get_dispatching_variables(order=processing_order)
                self.append_system_state_variables(
                    process_times=process_times,
                    slack=slack,
                    slack_opn=slack_opn
                )

            # get state information from orders currently in queues
            for i, queueing_order in enumerate(self.sim.model_panel.ORDER_QUEUES[WC].items):
                self.WIP += 1
                processing_order = queueing_order[0]
                # pick remaining process time
                process_times = []
                for k, operation in enumerate(processing_order.routing_sequence):
                    process_times.append(processing_order.process_time[operation])

                # order params
                slack, slack_opn = self.get_dispatching_variables(order=processing_order)
                self.append_system_state_variables(
                    process_times=process_times,
                    slack=slack,
                    slack_opn=slack_opn
                )

        # if we use an hierarchical release method to control FOCUS
        for i, order in enumerate(self.sim.model_panel.ORDER_POOL.items):
            processing_order = order[0]
            process_times = []
            for k, operation in enumerate(processing_order.routing_sequence):
                process_times.append(processing_order.process_time[operation])
                # update upstream load
            # order params
            slack, slack_opn = self.get_dispatching_variables(order=processing_order)
            self.append_system_state_variables(
                process_times=process_times,
                slack=slack,
                slack_opn=slack_opn
            )

        """ 
        update system state params 
        """
        # dispatching variables
        if not len(self.T_list) == 0:
            self.p_ij_min = min(self.T_list)
            self.p_ij_max = max(self.T_list)
            self.p_ij_mean = sum(self.T_list) / len(self.T_list)

        if not len(self.S_list) == 0:
            self.slack_min = min(self.S_list)
            self.slack_max = max(self.S_list)
            self.slack_mean = sum(self.S_list) / len(self.S_list)

        if not len(self.V_list) == 0:
            self.slack_opn_min = min(self.V_list)
            self.slack_opn_max = max(self.V_list)
            self.slack_opn_mean = sum(self.V_list) / len(self.V_list)

        # virtual WIP
        self.number_of_orders_in_system = len(self.V_list)

        # find if a work centre is starving
        if self.functions_activated["xi"] > 0:
            for j, WC in enumerate(self.work_centre_layout):
                if WC != work_centre:
                    if len(self.sim.model_panel.ORDER_QUEUES[WC].items) == 0 \
                            and len(self.sim.model_panel.ORDER_POOLS[WC].items) == 0:
                        self.A_dict[WC] = 1
                    else:
                        self.A_dict[WC] = 0
                else:
                    self.A_dict[WC] = 0
        return

    def append_system_state_variables(self, process_times, slack, slack_opn):
        self.T_list.extend(process_times)
        self.S_list.append(slack)
        self.V_list.append(slack_opn)
        return

    def get_dispatching_variables(self, order):
        # slack
        slack = self.slack(d_i=order.due_date,
                           t=self.sim.env.now,
                           k=1,
                           sum_p_ij=order.remaining_process_time)
        # slack opn
        slack_opn = self.slack(d_i=order.due_date,
                               t=self.sim.env.now,
                               k=len(order.routing_sequence),
                               sum_p_ij=order.remaining_process_time
                               )
        return slack, slack_opn

    @staticmethod
    def xi_idleness_impact(order, process_time, upstream_starvation_dict):
        if len(order.routing_sequence) >= 2:
            if upstream_starvation_dict != 0:
                for WC, indicator in upstream_starvation_dict.items():
                    if order.routing_sequence[1] == WC and indicator == 1:
                        return process_time
                return 0
            else:
                return 0
        else:
            return 0

    @staticmethod
    def slack(d_i, t, k, sum_p_ij):
        return (d_i - t - sum_p_ij) / k

    @staticmethod
    def normalization(x_min, x_max, x):
        result = 0
        if (x_max - x_min) != 0:
            result = (x_max - x) / (x_max - x_min)
        return result

    def __str__(self):
        return "FOCUS"

