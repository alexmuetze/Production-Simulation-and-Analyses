"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Collecting of Detailled Run, Continious, Planned, Rank and Aggregated Data
- Different Joining and Aggregation Methods
"""


import pandas as pd
import warnings
import numpy as np


# Store Continous Data of a run-----------------------------------------------------------------------------------------
class Data_Continous_Run(object):

    # Initialisation of a run-------------------------------------------------------------------------------------------
    def __init__(self, simulation):
        self.sim = simulation
        self.continous_database = None
        self.continous_data = list()

        self.order_input_counter: int = 0
        self.order_release_counter: int = 0
        self.order_output_counter: int = 0
        self.order_pool_counter: int = 0
        self.order_shop_counter: int = 0
        self.order_total_counter: int = 0
        self.load_input_counter: float = 0.0
        self.load_release_counter: float = 0.0
        self.load_output_counter: float = 0.0
        self.load_pool_counter: float = 0.0
        self.load_shop_counter: float = 0.0
        self.load_total_counter: float = 0.0
        self.order_input_counter_wc: Dict[...] = {}
        self.load_input_counter_wc: Dict[...] = {}
        self.order_output_counter_wc: Dict[...] = {}
        self.load_output_counter_wc: Dict[...] = {}
        self.order_wip_counter_wc: Dict[...] = {}
        self.load_wip_counter_wc: Dict[...] = {}

        for i, WC in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            self.order_input_counter_wc[WC] = 0
            self.load_input_counter_wc[WC] = 0.0
            self.order_output_counter_wc[WC] = 0
            self.load_output_counter_wc[WC] = 0.0
            self.order_wip_counter_wc[WC] = 0
            self.load_wip_counter_wc[WC] = 0.0

    # Function for the Reset of the counter for each run----------------------------------------------------------------
    def reset_counter(self):
        self.order_input_counter = 0
        self.order_release_counter = 0
        self.order_output_counter = 0
        self.order_pool_counter = 0
        self.order_shop_counter = 0
        self.order_total_counter = 0
        self.load_input_counter = 0.0
        self.load_release_counter = 0.0
        self.load_output_counter = 0.0
        self.load_pool_counter = 0.0
        self.load_shop_counter = 0.0
        self.load_total_counter = 0.0

        for i, WC in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            self.order_input_counter_wc[WC] = 0
            self.load_input_counter_wc[WC] = 0.0
            self.order_output_counter_wc[WC] = 0
            self.load_output_counter_wc[WC] = 0.0
            self.order_wip_counter_wc[WC] = 0
            self.load_wip_counter_wc[WC] = 0.0

        return

    # Method for saving the data of a run-------------------------------------------------------------------------------
    def run_update(self, warmup):
        """
        function that update's the database of the experiment for each run
        """

        if not warmup:
            # Call Method to store the data automaticaly
            self.get_final_data()
            self.store_continous_data()
            # self.reset_counter()

        # Delete list to get the start after Warmup
        self.continous_data = list()
        return

    # Get the data of the last time stamp-------------------------------------------------------------------------------
    def get_final_data(self):

        df_list=list()
        df_list.append(self.sim.process.scd_calculator(self.sim.env.now))
        df_list.append(self.order_input_counter)
        df_list.append(self.order_release_counter)
        df_list.append(self.order_output_counter)
        df_list.append(self.order_input_counter - self.order_release_counter)
        df_list.append(self.order_release_counter - self.order_output_counter)
        df_list.append(self.order_input_counter - self.order_output_counter)
        df_list.append(self.load_input_counter)
        df_list.append(self.load_release_counter)
        df_list.append(self.load_output_counter)
        df_list.append(self.load_input_counter - self.load_release_counter)
        df_list.append(self.load_release_counter - self.load_output_counter)
        df_list.append(self.load_input_counter - self.load_output_counter)

        for i, WC in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            df_list.append(self.order_input_counter_wc[WC])
            df_list.append(self.load_input_counter_wc[WC])
            df_list.append(self.order_output_counter_wc[WC])
            df_list.append(self.load_output_counter_wc[WC])
            df_list.append(self.order_input_counter_wc[WC] - self.order_output_counter_wc[WC])
            df_list.append(self.load_input_counter_wc[WC] - self.load_output_counter_wc[WC])
            df_list.append(self.sim.model_panel.LOAD_ACCOUNT[WC])


        # save list if it is not empty
        if len(df_list) != 0:
            self.continous_data.append(df_list)

        return

    # Method to get the data--------------------------------------------------------------------------------------------
    def continous_data_getter(self):

        while True:
            df_list = list()
            df_list.append(self.sim.process.scd_calculator(self.sim.env.now))
            df_list.append(self.order_input_counter)
            df_list.append(self.order_release_counter)
            df_list.append(self.order_output_counter)
            df_list.append(self.order_input_counter - self.order_release_counter)
            df_list.append(self.order_release_counter - self.order_output_counter)
            df_list.append(self.order_input_counter - self.order_output_counter)
            df_list.append(self.load_input_counter)
            df_list.append(self.load_release_counter)
            df_list.append(self.load_output_counter)
            df_list.append(self.load_input_counter - self.load_release_counter)
            df_list.append(self.load_release_counter - self.load_output_counter)
            df_list.append(self.load_input_counter - self.load_output_counter)

            for i, WC in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
                df_list.append(self.order_input_counter_wc[WC])
                df_list.append(self.load_input_counter_wc[WC])
                df_list.append(self.order_output_counter_wc[WC])
                df_list.append(self.load_output_counter_wc[WC])
                df_list.append(self.order_input_counter_wc[WC] - self.order_output_counter_wc[WC])
                df_list.append(self.load_input_counter_wc[WC] - self.load_output_counter_wc[WC])
                df_list.append(self.sim.model_panel.LOAD_ACCOUNT[WC])

            # save list if it is not empty
            if len(df_list) != 0:
                self.continous_data.append(df_list)

            yield self.sim.env.process(self.shiftcalender(1))

            if self.sim.env.now >= (self.sim.model_panel.WARM_UP_PERIOD + self.sim.model_panel.RUN_TIME) \
                    * self.sim.model_panel.NUMBER_OF_RUNS:
                break
        return

    # Method to store the continous data--------------------------------------------------------------------------------
    def store_continous_data(self):

        self.columns_names_run = []
        self.columns_names_run.extend([
            "time",
            "Order_Input",
            "Order_Release",
            "Order_Output",
            "Order_WIP_Pool",
            "Order_WIP_Shop",
            "Order_WIP_Total",
            "Load_Input",
            "Load_Release",
            "Load_Output",
            "Load_WIP_Pool",
            "Load_WIP_Shop",
            "Load_WIP_Total"
            ])

        for _, work_centre in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            self.columns_names_run.append(f"Order_Input_{work_centre}")
            self.columns_names_run.append(f"Load_Input_{work_centre}")
            self.columns_names_run.append(f"Order_Output_{work_centre}")
            self.columns_names_run.append(f"Load_Output_{work_centre}")
            self.columns_names_run.append(f"Order_WIP_{work_centre}")
            self.columns_names_run.append(f"Load_WIP_{work_centre}")
            self.columns_names_run.append(f"Load_Account_LOOR_{work_centre}")

        # update database
        df_run = pd.DataFrame(self.continous_data)
        df_run.columns = self.columns_names_run

        # data processing finished. Update database new run
        self.continous_data = list()

        self.continous_database = df_run

        return

    # Shiftcalender to ensure that every interval X of a planned day the data is saved into continuous data-------------
    def shiftcalender(self, duration):

        self.sc_duration = duration
        if not self.sim.policy_panel.capacityflexibilty:
            yield self.sim.env.timeout(self.sc_duration)
            return 0
        else:
            # Start auf Schichtkalender rechnen!
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

# Store Detailed and Aggregated Data of Experiment----------------------------------------------------------------------
class Data_Experiment(object):

    # Initialisation----------------------------------------------------------------------------------------------------
    def __init__(self, simulation):

        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        self.sim = simulation
        self.accumulated_process_time = 0
        self.experiment_database = None
        self.rank_calc_database = None
        self.calculated_ranks_database = None
        self.calculated_plan_values = None
        self.order_list = list()

        # basic name list
        self.columns_names_run = []
        if self.sim.model_panel.COLLECT_BASIC_DATA:
            self.columns_names_run.extend([
                                "identifier",
                                "throughput_time",
                                "pool_time",
                                "process_throughput_time",
                                "lateness",
                                "tardiness",
                                "tardy",
                                ])

        else:
            self.columns_names_run = ["identifier", "time"]
            self.columns_names_run.append(f"continous triggered")
            self.columns_names_run.append(f"total_process_time")
            self.columns_names_run.append(f"interarrival_time")
            self.columns_names_run.append(f"interarrival_time_after")
            self.columns_names_run.append(f"entry_time")
            self.columns_names_run.append(f"planned_entry_time")
            self.columns_names_run.append(f"release_time")
            self.columns_names_run.append(f"interrelease_time")
            self.columns_names_run.append(f"planned_release_time")
            self.columns_names_run.append(f"pool_time")
            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                self.columns_names_run.append(f"operation_number_{work_centre}")
                self.columns_names_run.append(f"process_time_{work_centre}")
                self.columns_names_run.append(f"operation_interarrival_time_{work_centre}")
                self.columns_names_run.append(f"operation_input_time_{work_centre}")
                self.columns_names_run.append(f"operation_planned_input_date_{work_centre}")
                self.columns_names_run.append(f"operation_start_time_{work_centre}")
                self.columns_names_run.append(f"operation_completion_time_{work_centre}")
                self.columns_names_run.append(f"operation_interdepature_time_{work_centre}")
                self.columns_names_run.append(f"operation_due_date_{work_centre}")
            self.columns_names_run.append(f"finishing_time")
            self.columns_names_run.append(f"due_date")
            self.columns_names_run.append(f"customer_due_date")
            self.columns_names_run.append(f"utilization")

            self.columns_names_run.append(f"pool_throughput_time")
            self.columns_names_run.append(f"shop_throughput_time")
            self.columns_names_run.append(f"throughput_time")
            self.columns_names_run.append(f"dispositive_time_plan")
            self.columns_names_run.append(f"planned_pool_throughput_time")
            self.columns_names_run.append(f"planned_shop_throughput_time")
            self.columns_names_run.append(f"planned_throughput_time")
            self.columns_names_run.append(f"time_to_prodcution_dd")
            self.columns_names_run.append(f"time_to_customer_dd")

            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                self.columns_names_run.append(f"inter_operation_time_{work_centre}")
                self.columns_names_run.append(f"operation_time_{work_centre}")
                self.columns_names_run.append(f"throughput_time_{work_centre}")
                self.columns_names_run.append(f"planned_throughput_time_{work_centre}")

            self.columns_names_run.append(f"deviation_input")
            self.columns_names_run.append(f"deviation_release")
            self.columns_names_run.append(f"lateness_prod")
            self.columns_names_run.append(f"lateness_customer")
            self.columns_names_run.append(f"deviation_relative_prod")
            self.columns_names_run.append(f"deviation_relative_customer")
            self.columns_names_run.append(f"tardiness_customer")
            self.columns_names_run.append(f"tardiness_prod")
            self.columns_names_run.append(f"tardy_customer")
            self.columns_names_run.append(f"tardy_prod")

            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                self.columns_names_run.append(f"deviation_input_{work_centre}")
                self.columns_names_run.append(f"deviation_output_{work_centre}")
                self.columns_names_run.append(f"deviation_relative_{work_centre}")

            self.columns_names_run.append(f"customer_slack")
            self.columns_names_run.append(f"entry_slack")
            self.columns_names_run.append(f"release_slack")
            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                self.columns_names_run.append(f"queue_entry_slack_{work_centre}")
                self.columns_names_run.append(f"operation_start_slack_{work_centre}")
                self.columns_names_run.append(f"operation_completion_slack_{work_centre}")
            self.columns_names_run.append(f"finish_slack")
            self.columns_names_run.append(f'length_of_routing')
            self.columns_names_run.append(f"Routing")

    # Append Data to already existing list------------------------------------------------------------------------------
    def append_run_list(self, result_list):
        """
        append the result of an order to the already existing order list
        """

        self.order_list.append(result_list)
        return

    # Update Procedure--------------------------------------------------------------------------------------------------
    def run_update(self, warmup):
        """
        function that update's the database of the experiment for each run
        """

        if not warmup:
            # update database
            self.generate_aggregated_load_data()
            self.store_data_for_rank_calculation()
            self.store_detailled_run_data()
            self.store_aggregated_run_data()

        # clean for next run
        if warmup:
            self.store_data_for_rank_calculation()

        self.order_list=list()

        # setting back
        self.accumulated_process_time = 0
        return

    # Store Data needed for rank calculation (preparation)--------------------------------------------------------------
    def store_data_for_rank_calculation(self):
        df_run = pd.DataFrame(self.order_list)
        df_run.columns = self.columns_names_run

        df = pd.DataFrame()
        df["order_number"] = df_run["identifier"]
        df["actual_entry"] = df_run["entry_time"]
        df["planned_entry"]=df_run["planned_entry_time"]
        df["total_process_time"]=df_run["total_process_time"]
        df["actual_release"]=df_run["release_time"]
        df["planned_release"]=df_run["planned_release_time"]
        df["actual_finish"]=df_run["finishing_time"]
        df["planned_finish"]=df_run["due_date"]

        for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
            df[f"process_time_{work_centre}"]=df_run[f"process_time_{work_centre}"]
            df[f"actual_input_{work_centre}"]=df_run[f"operation_input_time_{work_centre}"]
            df[f"planned_input_{work_centre}"]=df_run[f"operation_planned_input_date_{work_centre}"]
            df[f"process_time_{work_centre}"]=df_run[f"process_time_{work_centre}"]
            df[f"actual_output_{work_centre}"]=df_run[f"operation_completion_time_{work_centre}"]
            df[f"planned_output_{work_centre}"]=df_run[f"operation_due_date_{work_centre}"]


        if self.rank_calc_database is None:
            self.rank_calc_database = df
        else:
            self.rank_calc_database = pd.concat([self.rank_calc_database, df], ignore_index=True)
        return

    # Rank Calculation via before saved Database------------------------------------------------------------------------
    def calculation_of_ranks(self):

        df = self.rank_calc_database
        shape=df.shape
        df=df.sort_values(by='order_number')

        list=[]
        df2=df.iloc[:, [1, 2, 4, 5, 6, 7]]

        list=df2.columns.tolist()

        for i in list:
            if i == 'actual_release':
                df=df.sort_values(by='planned_release')

            df=df.sort_values(by=i)
            df[f"{i}_Rank"]=df[f"{i}"].rank(method='first')
            df[f"{i}_Rank_weighted"]=df["total_process_time"].cumsum()

        list=[]

        for i in range(0, self.sim.model_panel.NUMBER_OF_WORKCENTRES):
            df2=df.iloc[:, [9 + 5 * i, 10 + 5 * i, 11 + 5 * i, 12 + 5 * i]]
            list=df2.columns.tolist()

            for j in list:
                df=df.sort_values(by=j)
                df[f"{j}_Rank"]=df[f"{j}"].rank(method='first')
                variable=df.iloc[:, [7 + 5 * i]].columns.tolist()
                df[f"{j}_Rank_weighted"]=df[f"{variable[0]}"].cumsum()


        df2=df.drop(df.iloc[:,1:shape[1]], axis=1)

        self.calculated_ranks_database = df2

        return

    # Join Rank Database to Target Database-----------------------------------------------------------------------------
    def join_ranks(self, database):

        df = database
        df2 = self.calculated_ranks_database

        database = pd.merge(df,df2, how="left", left_on= 'identifier', right_on='order_number')


        return database

    # Calculation of Planned Values-------------------------------------------------------------------------------------
    def create_planned_values(self, num_export):

        lowerlimit = 0
        upperlimit = num_export * (self.sim.model_panel.RUN_TIME + self.sim.model_panel.WARM_UP_PERIOD)/10
        stepsize = 1 / self.sim.model_panel.STANDARDCAPACITY


        df2=pd.DataFrame()
        df2["Time"]=np.arange(lowerlimit, upperlimit, stepsize)

        df=self.rank_calc_database

        df_filter=pd.DataFrame()
        df_filter2=pd.DataFrame()
        df_filter3=pd.DataFrame()
        df_filter["Time"]=np.ceil(
            (df["planned_entry"] * self.sim.model_panel.STANDARDCAPACITY)) / self.sim.model_panel.STANDARDCAPACITY
        df_filter["total_process_time"]=df["total_process_time"]
        df_filter2["sum"]=df_filter.groupby(["Time"])["total_process_time"].sum()
        df_filter2['Planned_Load_Input']=df_filter2['sum'].cumsum()
        df_filter2=df_filter2.drop(df_filter2.columns[0], axis=1)
        df_filter3["sum"]=df_filter.groupby(["Time"])["Time"].count()
        df_filter3['Planned_Order_Input']=df_filter3['sum'].cumsum()
        df_filter3=df_filter3.drop(df_filter3.columns[0], axis=1)
        df2=pd.merge(df2, df_filter2, how="left", left_on='Time', right_index=True)
        df2=pd.merge(df2, df_filter3, how="left", left_on='Time', right_index=True)

        df_filter=pd.DataFrame()
        df_filter2=pd.DataFrame()
        df_filter3=pd.DataFrame()
        df_filter["Time"]=np.ceil(
            (df["planned_release"] * self.sim.model_panel.STANDARDCAPACITY)) / self.sim.model_panel.STANDARDCAPACITY
        df_filter["total_process_time"]=df["total_process_time"]
        df_filter2["sum"]=df_filter.groupby(["Time"])["total_process_time"].sum()
        df_filter2['Planned_Load_Release']=df_filter2['sum'].cumsum()
        df_filter2=df_filter2.drop(df_filter2.columns[0], axis=1)
        df_filter3["sum"]=df_filter.groupby(["Time"])["Time"].count()
        df_filter3['Planned_Order_Release']=df_filter3['sum'].cumsum()
        df_filter3=df_filter3.drop(df_filter3.columns[0], axis=1)
        df2=pd.merge(df2, df_filter2, how="left", left_on='Time', right_index=True)
        df2=pd.merge(df2, df_filter3, how="left", left_on='Time', right_index=True)

        df_filter=pd.DataFrame()
        df_filter2=pd.DataFrame()
        df_filter3=pd.DataFrame()
        df_filter["Time"]=np.ceil(
            (df["planned_finish"] * self.sim.model_panel.STANDARDCAPACITY)) / self.sim.model_panel.STANDARDCAPACITY
        df_filter["total_process_time"]=df["total_process_time"]
        df_filter2["sum"]=df_filter.groupby(["Time"])["total_process_time"].sum()
        df_filter2['Planned_Load_Output']=df_filter2['sum'].cumsum()
        df_filter2=df_filter2.drop(df_filter2.columns[0], axis=1)
        df_filter3["sum"]=df_filter.groupby(["Time"])["Time"].count()
        df_filter3['Planned_Order_Output']=df_filter3['sum'].cumsum()
        df_filter3=df_filter3.drop(df_filter3.columns[0], axis=1)
        df2=pd.merge(df2, df_filter2, how="left", left_on='Time', right_index=True)
        df2=pd.merge(df2, df_filter3, how="left", left_on='Time', right_index=True)

        for i, work_centre in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            df_filter=pd.DataFrame()
            df_filter2=pd.DataFrame()
            df_filter3=pd.DataFrame()
            df_filter["Time"]=np.ceil(
                (df[f"planned_input_{work_centre}"] * self.sim.model_panel.STANDARDCAPACITY)) / self.sim.model_panel.STANDARDCAPACITY
            df_filter[f"process_time_{work_centre}"]=df[f"process_time_{work_centre}"]
            df_filter2["sum"]=df_filter.groupby(["Time"])[f"process_time_{work_centre}"].sum()
            df_filter2[f'Planned_Load_Input_{work_centre}']=df_filter2['sum'].cumsum()
            df_filter2=df_filter2.drop(df_filter2.columns[0], axis=1)
            df_filter3["sum"]=df_filter.groupby(["Time"])["Time"].count()
            df_filter3[f'Planned_Order_Input_{work_centre}']=df_filter3['sum'].cumsum()
            df_filter3=df_filter3.drop(df_filter3.columns[0], axis=1)
            df2=pd.merge(df2, df_filter2, how="left", left_on='Time', right_index=True)
            df2=pd.merge(df2, df_filter3, how="left", left_on='Time', right_index=True)

            df_filter=pd.DataFrame()
            df_filter2=pd.DataFrame()
            df_filter3=pd.DataFrame()
            df_filter["Time"]=np.ceil(
                (df[
                     f"planned_output_{work_centre}"] * self.sim.model_panel.STANDARDCAPACITY)) / \
                              self.sim.model_panel.STANDARDCAPACITY
            df_filter[f"process_time_{work_centre}"]=df[f"process_time_{work_centre}"]
            df_filter2["sum"]=df_filter.groupby(["Time"])[f"process_time_{work_centre}"].sum()
            df_filter2[f'Planned_Load_Output_{work_centre}']=df_filter2['sum'].cumsum()
            df_filter2=df_filter2.drop(df_filter2.columns[0], axis=1)
            df_filter3["sum"]=df_filter.groupby(["Time"])["Time"].count()
            df_filter3[f'Planned_Order_Output_{work_centre}']=df_filter3['sum'].cumsum()
            df_filter3=df_filter3.drop(df_filter3.columns[0], axis=1)
            df2=pd.merge(df2, df_filter2, how="left", left_on='Time', right_index=True)
            df2=pd.merge(df2, df_filter3, how="left", left_on='Time', right_index=True)

        df2.fillna(method='ffill', inplace=True)
        df2.replace(np.nan, 0, inplace=True)

        df2["Planned_Load_WIP_Pool"]=df2["Planned_Load_Input"] - df2["Planned_Load_Release"]
        df2["Planned_Order_WIP_Pool"]=df2["Planned_Order_Input"] - df2["Planned_Order_Release"]
        df2["Planned_Load_WIP_Shop"]= df2["Planned_Load_Release"] - df2["Planned_Load_Output"]
        df2["Planned_Order_WIP_Shop"]= df2["Planned_Order_Release"] - df2["Planned_Order_Output"]
        df2["Planned_Load_WIP_System"]=df2["Planned_Load_Input"] - df2["Planned_Load_Output"]
        df2["Planned_Order_WIP_System"]=df2["Planned_Order_Input"] - df2["Planned_Order_Output"]

        for i, work_centre in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            df2[f"Planned_Load_WIP_{work_centre}"]= df2[f'Planned_Load_Input_{work_centre}'] - df2[f'Planned_Load_Output_{work_centre}']
            df2[f"Planned_Order_WIP_{work_centre}"]= df2[f'Planned_Order_Input_{work_centre}'] - df2[f'Planned_Order_Output_{work_centre}']

        self.calculated_plan_values = df2

        return

    # Join the Planned Values to target Database------------------------------------------------------------------------
    def join_planned_values(self, database):

        df=database
        df2= self.calculated_plan_values

        database=pd.merge(df, df2, how="left", left_on='time', right_on='Time')

        return database

    # Store of Detailled Run Data---------------------------------------------------------------------------------------
    def store_detailled_run_data(self):
        df_run = pd.DataFrame(self.order_list)
        df_run.columns = self.columns_names_run

        self.run_database = df_run

    # Generating the aggregated load_data on the basis of the continous data--------------------------------------------
    def generate_aggregated_load_data(self):

        df_run = self.sim.data_continous_run.continous_database
        self.order_input = df_run.loc[:, "Order_Input"].max() - df_run.loc[:, "Order_Input"].min()
        self.order_release = df_run.loc[:, "Order_Release"].max() - df_run.loc[:, "Order_Release"].min()
        self.order_output = df_run.loc[:, "Order_Output"].max() - df_run.loc[:, "Order_Output"].min()
        self.load_input = df_run.loc[:, "Load_Input"].max() - df_run.loc[:, "Load_Input"].min()
        self.load_release = df_run.loc[:, "Load_Release"].max() - df_run.loc[:, "Load_Release"].min()
        self.load_output = df_run.loc[:, "Load_Output"].max() - df_run.loc[:, "Load_Output"].min()
        self.order_at_start = df_run.loc[:, "Order_Input"].min() - df_run.loc[:, "Order_Output"].min()
        self.order_at_end = df_run.loc[:, "Order_Input"].max() - df_run.loc[:, "Order_Output"].max()
        self.load_at_start = df_run.loc[:, "Load_Input"].min() - df_run.loc[:, "Load_Output"].min()
        self.load_at_end = df_run.loc[:, "Load_Input"].max() - df_run.loc[:, "Load_Output"].max()
        self.mean_order_pool = df_run.loc[:, "Order_WIP_Pool"].mean()
        self.mean_order_shop = df_run.loc[:, "Order_WIP_Shop"].mean()
        self.mean_order_total = df_run.loc[:, "Order_WIP_Total"].mean()
        self.std_order_pool=df_run.loc[:, "Order_WIP_Pool"].std()
        self.std_order_shop=df_run.loc[:, "Order_WIP_Shop"].std()
        self.std_order_total=df_run.loc[:, "Order_WIP_Total"].std()

        self.mean_load_pool=df_run.loc[:, "Load_WIP_Pool"].mean()
        self.mean_load_shop=df_run.loc[:, "Load_WIP_Shop"].mean()
        self.mean_load_total=df_run.loc[:, "Load_WIP_Total"].mean()
        self.std_load_pool = df_run.loc[:, "Load_WIP_Pool"].std()
        self.std_load_shop = df_run.loc[:, "Load_WIP_Shop"].std()
        self.std_load_total = df_run.loc[:, "Load_WIP_Total"].std()

        self.order_input_wc: Dict[...] = {}
        self.load_input_wc: Dict[...] = {}
        self.order_output_wc: Dict[...] = {}
        self.load_output_wc: Dict[...] = {}
        self.mean_order_WIP_wc: Dict[...] = {}
        self.mean_load_WIP_wc: Dict[...] = {}
        self.std_order_WIP_wc: Dict[...]={}
        self.std_load_WIP_wc: Dict[...]={}
        self.order_at_start_wc: Dict[...] = {}
        self.order_at_end_wc: Dict[...] = {}
        self.load_at_start_wc: Dict[...] = {}
        self.load_at_end_wc: Dict[...] = {}
        self.utilization_wc: Dict[...] = {}

        for i, WC in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
            self.order_input_wc[WC] = df_run.loc[:, f"Order_Input_{WC}"].max() - df_run.loc[:, f"Order_Input_{WC}"].min()
            self.load_input_wc[WC] = df_run.loc[:, f"Load_Input_{WC}"].max() - df_run.loc[:, f"Load_Input_{WC}"].min()
            self.order_output_wc[WC] = df_run.loc[:, f"Order_Output_{WC}"].max() - df_run.loc[:, f"Order_Output_{WC}"].min()
            self.load_output_wc[WC] = df_run.loc[:, f"Load_Output_{WC}"].max() - df_run.loc[:, f"Load_Output_{WC}"].min()
            self.order_at_start_wc[WC] = df_run.loc[:, f"Order_Input_{WC}"].min() - df_run.loc[:, f"Order_Output_{WC}"].min()
            self.order_at_end_wc[WC] = df_run.loc[:, f"Order_Input_{WC}"].max() - df_run.loc[:, f"Order_Output_{WC}"].max()
            self.load_at_start_wc[WC] = df_run.loc[:, f"Load_Input_{WC}"].min() - df_run.loc[:, f"Load_Output_{WC}"].min()
            self.load_at_end_wc[WC] = df_run.loc[:, f"Load_Input_{WC}"].max() - df_run.loc[:, f"Load_Output_{WC}"].max()
            self.utilization_wc[WC] = self.load_output_wc[WC] / self.sim.model_panel.RUN_TIME / self.sim.model_panel.STANDARDCAPACITY * 10
            self.mean_order_WIP_wc[WC] = df_run.loc[:, f"Order_WIP_{WC}"].mean()
            self.mean_load_WIP_wc[WC] = df_run.loc[:, f"Load_WIP_{WC}"].mean()
            self.std_order_WIP_wc[WC]=df_run.loc[:, f"Order_WIP_{WC}"].std()
            self.std_load_WIP_wc[WC]=df_run.loc[:, f"Load_WIP_{WC}"].std()

        return

    # Store the aggregated run data-------------------------------------------------------------------------------------
    def store_aggregated_run_data(self):
        # put all data into dataframe
        df_run = self.run_database

        # dataframe for each run
        df = pd.DataFrame(
            [(int(self.sim.env.now / (self.sim.model_panel.WARM_UP_PERIOD + self.sim.model_panel.RUN_TIME)))],
            columns=['run'])
        df["Exp_Name"] = self.sim.model_panel.experiment_name
        df["Planned_Utilization"] = self.sim.model_panel.AIMED_UTILIZATION
        df["Release"] = self.sim.policy_panel.release_control_method
        df["Dispatching Rule"] = self.sim.policy_panel.dispatching_rules_print
        df["total_time"] = self.sim.model_panel.RUN_TIME
        df["nr_flow_items"] = df_run.shape[0]
        df["percentage_continous triggered"] = df_run.loc[:, "continous triggered"].sum() / df_run.shape[0]
        df["utilization"] = df_run.loc[:, "utilization"].max()
        df["mean_interarrival_time"] = df_run.loc[:, "interarrival_time"].mean()
        df["std_interarrival_time"] = df_run.loc[:, "interarrival_time"].std()
        df["mean_interrelease_time"] = df_run.loc[:, "interrelease_time"].mean()
        df["std_interrelease_time"] = df_run.loc[:, "interrelease_time"].std()

        df["mean_total_process_time"] = df_run.loc[:, "total_process_time"].mean()
        df["std_total_process_time"] = df_run.loc[:, "total_process_time"].std()
        df["Order_Input_Total"] = self.order_input
        df["Order_Release_Total"] = self.order_release
        df["Order_Output_Total"] = self.order_output
        df["Order_at_Start"] = self.order_at_start
        df["Order_at_End"] = self.order_at_end
        df["Load_Input_Total"] = self.load_input
        df["Load_Release_Total"] = self.load_release
        df["Load_Output_Total"] = self.load_output
        df["Load_at_Start"] = self.load_at_start
        df["Load_at_End"] = self.load_at_end

        df["mean_WIP_Pool"] = self.mean_order_pool
        df["mean_WIP_Shop"] = self.mean_order_shop
        df["mean_WIP_System"] = self.mean_order_total
        df["std_WIP_Pool"]=self.std_order_pool
        df["std_WIP_Shop"]=self.std_order_shop
        df["std_WIP_System"]=self.std_order_total
        df["mean_Load_Pool"] = self.mean_load_pool
        df["mean_Load_Shop"] = self.mean_load_shop
        df["mean_Load_System"] = self.mean_load_total
        df["std_Load_Pool"]=self.std_load_pool
        df["std_Load_Shop"]=self.std_load_shop
        df["std_Load_System"]=self.std_load_total

        df["mean_pool_throughput_time"] = df_run.loc[:, "pool_throughput_time"].mean()
        df["std_pool_throughput_time"] = df_run.loc[:, "pool_throughput_time"].std()
        df["mean_shop_throughput_time"] = df_run.loc[:, "shop_throughput_time"].mean()
        df["std_shop_throughput_time"] = df_run.loc[:, "shop_throughput_time"].std()
        df["mean_throughput_time"] = df_run.loc[:, "throughput_time"].mean()
        df["std_throughput_time"] = df_run.loc[:, "throughput_time"].std()
        df["mean_dispositive_time"] = df_run.loc[:, "dispositive_time_plan"].mean()
        df["std_dispositive_time"] = df_run.loc[:, "dispositive_time_plan"].std()
        df["mean_planned_pool_throughput_time"]=df_run.loc[:, "planned_pool_throughput_time"].mean()
        df["std_planned_pool_throughput_time"]=df_run.loc[:, "planned_pool_throughput_time"].std()
        df["mean_planned_shop_throughput_time"] = df_run.loc[:, "planned_shop_throughput_time"].mean()
        df["std_planned_shop_throughput_time"] = df_run.loc[:, "planned_shop_throughput_time"].std()
        df["mean_planned_throughput_time"]=df_run.loc[:, "planned_throughput_time"].mean()
        df["std_planned_throughput_time"]=df_run.loc[:, "planned_throughput_time"].std()
        df["mean_time_to_prodcution_dd"] = df_run.loc[:, "time_to_prodcution_dd"].mean()
        df["std_time_to_prodcution_dd"] = df_run.loc[:, "time_to_prodcution_dd"].std()
        df["mean_time_to_customer_dd"] = df_run.loc[:, "time_to_customer_dd"].mean()
        df["std_time_to_customer_dd"] = df_run.loc[:, "time_to_customer_dd"].std()

        df["mean_deviation_input"]=df_run.loc[:, "deviation_input"].mean()
        df["std_deviation_input"]=df_run.loc[:, "deviation_input"].std()
        df["mean_deviation_release"] = df_run.loc[:, "deviation_release"].mean()
        df["std_deviation_release"] = df_run.loc[:, "deviation_release"].std()
        df["mean_lateness_prod"] = df_run.loc[:, "lateness_prod"].mean()
        df["std_lateness_prod"] = df_run.loc[:, "lateness_prod"].std()
        df["mean_lateness_customer"] = df_run.loc[:, "lateness_customer"].mean()
        df["std_lateness_customer"] = df_run.loc[:, "lateness_customer"].std()
        df["mean_deviation_relative_prod"] = df_run.loc[:, "deviation_relative_prod"].mean()
        df["std_deviation_relative_prod"] = df_run.loc[:, "deviation_relative_prod"].std()
        df["mean_deviation_relative_customer"] = df_run.loc[:, "deviation_relative_customer"].mean()
        df["std_deviation_relative_customer"] = df_run.loc[:, "deviation_relative_customer"].std()
        df["percentage_tardy_prod"] = df_run.loc[:, "tardy_prod"].sum() / df_run.shape[0]
        df["percentage_tardy_customer"] = df_run.loc[:, "tardy_customer"].sum() / df_run.shape[0]

        for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
            df[f"mean_process_time_{work_centre}"] = df_run.loc[:, f"process_time_{work_centre}"].mean()
            df[f"std_process_time_{work_centre}"] = df_run.loc[:, f"process_time_{work_centre}"].std()
            df[f"ideal_minimum_wip_{work_centre}"] = df_run.loc[:, f"process_time_{work_centre}"].mean() + \
                                                     df_run.loc[:, f"process_time_{work_centre}"].std() * \
                                                     df_run.loc[:, f"process_time_{work_centre}"].std() / \
                                                     df_run.loc[:, f"process_time_{work_centre}"].mean()
            df[f"total_input_order_{work_centre}"] = self.order_input_wc[work_centre]
            df[f"total_input_load_{work_centre}"] = self.load_input_wc[work_centre]
            df[f"total_output_order_{work_centre}"] = self.order_output_wc[work_centre]
            df[f"total_output_load_{work_centre}"] = self.load_output_wc[work_centre]
            df[f"total_order_at_start_{work_centre}"] = self.order_at_start_wc[work_centre]
            df[f"total_order_at_end_{work_centre}"] = self.order_at_end_wc[work_centre]
            df[f"total_load_at_start_{work_centre}"] = self.load_at_start_wc[work_centre]
            df[f"total_load_at_end_{work_centre}"] = self.load_at_end_wc[work_centre]
            df[f"utilization_{work_centre}"] = self.utilization_wc[work_centre]
            df[f"mean_WIP_{work_centre}"] = self.mean_order_WIP_wc[work_centre]
            df[f"mean_load_{work_centre}"] = self.mean_load_WIP_wc[work_centre]
            df[f"std_WIP_{work_centre}"]=self.std_order_WIP_wc[work_centre]
            df[f"std_load_{work_centre}"]=self.std_load_WIP_wc[work_centre]
            df[f"mean_interarrival_time_{work_centre}"] = df_run.loc[:, f"operation_interarrival_time_{work_centre}"].mean()
            df[f"std_interarrival_time_{work_centre}"] = df_run.loc[:, f"operation_interarrival_time_{work_centre}"].std()
            df[f"mean_interdepature_time_{work_centre}"] = df_run.loc[:,
                                                          f"operation_interdepature_time_{work_centre}"].mean()
            df[f"std_interdepature_time_{work_centre}"] = df_run.loc[:,
                                                         f"operation_interdepature_time_{work_centre}"].std()
            df[f"mean_inter_operation_time_{work_centre}"] = df_run.loc[:, f"inter_operation_time_{work_centre}"].mean()
            df[f"mean_inter_operation_time_{work_centre}"] = df_run.loc[:, f"inter_operation_time_{work_centre}"].mean()
            df[f"std_inter_operation_time_{work_centre}"] = df_run.loc[:, f"inter_operation_time_{work_centre}"].std()
            df[f"mean_operation_time_{work_centre}"] = df_run.loc[:, f"operation_time_{work_centre}"].mean()
            df[f"std_operation_time_{work_centre}"] = df_run.loc[:, f"operation_time_{work_centre}"].std()
            df[f"mean_throughput_time_{work_centre}"] = df_run.loc[:, f"throughput_time_{work_centre}"].mean()
            df[f"std_throughput_time_{work_centre}"] = df_run.loc[:, f"throughput_time_{work_centre}"].std()
            df[f"mean_planned_throughput_time_{work_centre}"] = df_run.loc[:, f"planned_throughput_time_{work_centre}"].mean()
            df[f"std_planned_throughput_time_{work_centre}"] = df_run.loc[:, f"planned_throughput_time_{work_centre}"].std()
            df[f"mean_deviation_input_{work_centre}"] = df_run.loc[:, f"deviation_input_{work_centre}"].mean()
            df[f"std_deviation_input_{work_centre}"] = df_run.loc[:, f"deviation_input_{work_centre}"].std()
            df[f"mean_deviation_output_{work_centre}"] = df_run.loc[:, f"deviation_output_{work_centre}"].mean()
            df[f"std_deviation_output_{work_centre}"] = df_run.loc[:, f"deviation_output_{work_centre}"].std()
            df[f"mean_deviation_relative_{work_centre}"] = df_run.loc[:,
                                                                f"deviation_relative_{work_centre}"].mean()
            df[f"std_deviation_relative_{work_centre}"] = df_run.loc[:,
                                                               f"deviation_relative_{work_centre}"].std()

        for i in self.sim.model_panel.params_dict:
            df[f"{i}"] = str(self.sim.model_panel.params_dict[i])

        if self.sim.policy_panel.WIP_control == True:
            self.WIP_names_run=[]
            self.WIP_names_run.extend(["time"])
            for _, work_centre in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
                self.WIP_names_run.append(f"Capacity_{work_centre}")
            df_run=pd.DataFrame(self.sim.process.WIP_control_list)
            df_run.columns=self.WIP_names_run
            self.sim.process.WIP_control_list = []

            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                df[f"mean_capacity_{work_centre}"]=df_run[f"Capacity_{work_centre}"].mean()


        # save data from the run
        if self.experiment_database is None:
            self.experiment_database = df
        else:
            self.experiment_database = pd.concat([self.experiment_database, df], ignore_index=True)
        return

    # Append Planned and Ranked Data to target Database-----------------------------------------------------------------
    def append_aggregated_plan_and_rank_data(self):

        for j in range(1, self.sim.model_panel.NUMBER_OF_RUNS+1):
            df = self.sim.data.experiment_database
            df2 = self.sim.continous_run_db[j - 1]
            df3 = self.sim.detailled_run_db[j - 1]

            df.loc[df['run'] == j, "planned_mean_WIP_Pool"] = df2["Planned_Load_WIP_Pool"].mean()
            df.loc[df['run'] == j, "planned_mean_WIP_Shop"]= df2["Planned_Load_WIP_Pool"].mean()
            df.loc[df['run'] == j, "planned_mean_WIP_System"]= df2["Planned_Load_WIP_Pool"].mean()
            df.loc[df['run'] == j, "planned_std_WIP_Pool"]= df2["Planned_Load_WIP_Pool"].std()
            df.loc[df['run'] == j, "planned_std_WIP_Shop"]= df2["Planned_Load_WIP_Pool"].std()
            df.loc[df['run'] == j, "planned_std_WIP_System"]= df2["Planned_Load_WIP_Pool"].std()
            df.loc[df['run'] == j, "planned_mean_Load_Pool"]= df2["Planned_Load_WIP_Pool"].mean()
            df.loc[df['run'] == j, "planned_mean_Load_Shop"]= df2["Planned_Load_WIP_Pool"].mean()
            df.loc[df['run'] == j, "planned_mean_Load_System"]= df2["Planned_Load_WIP_Pool"].mean()
            df.loc[df['run'] == j, "planned_std_Load_Pool"]= df2["Planned_Load_WIP_Pool"].std()
            df.loc[df['run'] == j, "planned_std_Load_Shop"]= df2["Planned_Load_WIP_Pool"].std()
            df.loc[df['run'] == j, "planned_std_Load_System"]= df2["Planned_Load_WIP_Pool"].std()
            df.loc[df['run'] == j, "planned_Order_Input_Total"]= df2["Planned_Order_Input"].max() - df2["Planned_Order_Input"].min()
            df.loc[df['run'] == j, "planned_Order_Release_Total"]= df2["Planned_Order_Release"].max() - df2["Planned_Order_Release"].min()
            df.loc[df['run'] == j, "planned_Order_Output_Total"]= df2["Planned_Order_Output"].max() - df2["Planned_Order_Output"].min()
            df.loc[df['run'] == j, "planned_Load_Input_Total"]= df2["Planned_Load_Input"].max() - df2["Planned_Load_Input"].min()
            df.loc[df['run'] == j, "planned_Load_Release_Total"]= df2["Planned_Load_Release"].max() - df2["Planned_Load_Release"].min()
            df.loc[df['run'] == j, "planned_Load_Output_Total"]= df2["Planned_Load_Output"].max() - df2["Planned_Load_Output"].min()

            for i, work_centre in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
                df.loc[df['run'] == j, f"planned_mean_WIP_{work_centre}"] = df2[f"Planned_Order_WIP_{work_centre}"].mean()
                df.loc[df['run'] == j, f"planned_mean_load_{work_centre}"]= df2[f"Planned_Load_WIP_{work_centre}"].mean()
                df.loc[df['run'] == j, f"planned_std_WIP_{work_centre}"]=df2[f"Planned_Order_WIP_{work_centre}"].std()
                df.loc[df['run'] == j, f"planned_std_load_{work_centre}"]=df2[f"Planned_Load_WIP_{work_centre}"].std()
                df.loc[df['run'] == j, f"planned_total_input_order_{work_centre}"]= df2[f"Planned_Order_Input_{work_centre}"].max() - df2[f"Planned_Order_Input_{work_centre}"].min()
                df.loc[df['run'] == j, f"planned_total_input_load_{work_centre}"]= df2[f"Planned_Load_Input_{work_centre}"].max() - df2[f"Planned_Load_Input_{work_centre}"].min()
                df.loc[df['run'] == j, f"planned_total_output_order_{work_centre}"]= df2[f"Planned_Order_Output_{work_centre}"].max() - df2[f"Planned_Order_Output_{work_centre}"].min()
                df.loc[df['run'] == j, f"planned_total_output_load_{work_centre}"]= df2[f"Planned_Load_Output_{work_centre}"].max() - df2[f"Planned_Load_Output_{work_centre}"].min()

            df.loc[df['run'] == j, "mean_rank_deviation_input"]=(df3["actual_entry_Rank"] - df3["planned_entry_Rank"]).mean()
            df.loc[df['run'] == j, "mean_weighted_rank_deviation_input"]=(df3["actual_entry_Rank_weighted"] - df3["planned_entry_Rank_weighted"]).mean()
            df.loc[df['run'] == j, "mean_rank_deviation_release"]=(df3["actual_release_Rank"] - df3["planned_release_Rank"]).mean()
            df.loc[df['run'] == j, "mean_weighted_rank_deviation_release"]=(df3["actual_release_Rank_weighted"] - df3["planned_release_Rank_weighted"]).mean()
            df.loc[df['run'] == j, "mean_rank_deviation_output"]=(df3["actual_finish_Rank"] - df3["planned_finish_Rank"]).mean()
            df.loc[df['run'] == j, "mean_weighted_rank_deviation_output"]=(df3["actual_finish_Rank_weighted"] - df3["planned_finish_Rank_weighted"]).mean()

            df.loc[df['run'] == j, "std_rank_deviation_input"]=(
                        df3["actual_entry_Rank"] - df3["planned_entry_Rank"]).std()
            df.loc[df['run'] == j, "std_weighted_rank_deviation_input"]=(
                        df3["actual_entry_Rank_weighted"] - df3["planned_entry_Rank_weighted"]).std()
            df.loc[df['run'] == j, "std_rank_deviation_release"]=(
                        df3["actual_release_Rank"] - df3["planned_release_Rank"]).std()
            df.loc[df['run'] == j, "std_weighted_rank_deviation_release"]=(
                        df3["actual_release_Rank_weighted"] - df3["planned_release_Rank_weighted"]).std()
            df.loc[df['run'] == j, "std_rank_deviation_output"]=(
                        df3["actual_finish_Rank"] - df3["planned_finish_Rank"]).std()
            df.loc[df['run'] == j, "std_weighted_rank_deviation_output"]=(
                        df3["actual_finish_Rank_weighted"] - df3["planned_finish_Rank_weighted"]).std()

            for i, work_centre in enumerate(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT):
                df.loc[df['run'] == j, f"std_rank_deviation_input_{work_centre}"] = (df3[f"actual_input_{work_centre}_Rank"] - df3[f"planned_input_{work_centre}_Rank"]).std()
                df.loc[df['run'] == j, f"std_weighted_rank_deviation_input_{work_centre}"]= (df3[f"actual_input_{work_centre}_Rank_weighted"] - df3[f"planned_input_{work_centre}_Rank_weighted"]).std()
                df.loc[df['run'] == j, f"std_rank_deviation_output_{work_centre}"]=(
                            df3[f"actual_output_{work_centre}_Rank"] - df3[f"planned_output_{work_centre}_Rank"]).std()
                df.loc[df['run'] == j, f"std_weighted_rank_deviation_output_{work_centre}"]=(
                            df3[f"actual_output_{work_centre}_Rank_weighted"] - df3[
                        f"planned_output_{work_centre}_Rank_weighted"]).std()

        return

    # Calculation and Appending of Load / WIP data via Continous (hourly) data and Aggregation--------------------------
    def append_BELA_data(self):

        for j in range(1, self.sim.model_panel.NUMBER_OF_RUNS+1):
            df = self.sim.data.experiment_database
            df2 = self.sim.continous_run_db[j - 1]

            calculate_system: bool = True

            if calculate_system == True:
                minZeit=int(round(df2["time"].min()))
                maxZeit=int(round(df2["time"].max()))

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Load_Input"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Load_Input"] - df_aggregat[
                    f"Load_Input"].shift(1)
                meanBela_Load=round(df_aggregat['Difference'].mean(), 4)
                stdBela_Load=round(df_aggregat['Difference'].std(), 4)
                minBela_Load=round(df_aggregat['Difference'].min(), 4)
                maxBela_Load=round(df_aggregat['Difference'].max(), 4)
                medianBela_Load=round(df_aggregat['Difference'].median(), 4)
                df.loc[df['run'] == j, f"BL_Bela_System_mean"]=meanBela_Load
                df.loc[df['run'] == j, f"BL_Bela_System_std"]=stdBela_Load
                df.loc[df['run'] == j, f"BL_Bela_System_min"]=minBela_Load
                df.loc[df['run'] == j, f"BL_Bela_System_max"]=maxBela_Load
                df.loc[df['run'] == j, f"BL_Bela_System_median"]=medianBela_Load

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Load_Output"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Load_Output"] - df_aggregat[
                    f"Load_Output"].shift(1)
                meanLeistung_Load=round(df_aggregat['Difference'].mean(), 4)
                stdLeistung_Load=round(df_aggregat['Difference'].std(), 4)
                minLeistung_Load=round(df_aggregat['Difference'].min(), 4)
                maxLeistung_Load=round(df_aggregat['Difference'].max(), 4)
                medianLeistung_Load=round(df_aggregat['Difference'].median(), 2)
                df.loc[df['run'] == j, f"BL_Leistung_System_mean"]=meanLeistung_Load
                df.loc[df['run'] == j, f"BL_Leistung_System_std"]=stdLeistung_Load
                df.loc[df['run'] == j, f"BL_Leistung_System_min"]=minLeistung_Load
                df.loc[df['run'] == j, f"BL_Leistung_System_max"]=maxLeistung_Load
                df.loc[df['run'] == j, f"BL_Leistung_System_median"]=medianLeistung_Load

                meanBestand_Load=round(df2[f"Load_WIP_Total"].mean(), 4)
                stdBestand_Load=round(df2[f"Load_WIP_Total"].std(), 4)
                minBestand_Load=round(df2[f"Load_WIP_Total"].min(), 4)
                maxBestand_Load=round(df2[f"Load_WIP_Total"].max(), 4)
                medianBestand_Load=round(df2[f"Load_WIP_Total"].median(), 4)
                df.loc[df['run'] == j, f"BL_Bestand_System_mean"]=meanBestand_Load
                df.loc[df['run'] == j, f"BL_Bestand_System_std"]=stdBestand_Load
                df.loc[df['run'] == j, f"BL_Bestand_System_min"]=minBestand_Load
                df.loc[df['run'] == j, f"BL_Bestand_System_max"]=maxBestand_Load
                df.loc[df['run'] == j, f"BL_Bestand_System_median"]=medianBestand_Load

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Order_Input"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Order_Input"] - df_aggregat[
                    f"Order_Input"].shift(1)
                meanBela_Order=round(df_aggregat['Difference'].mean(), 4)
                stdBela_Order=round(df_aggregat['Difference'].std(), 4)
                minBela_Order=round(df_aggregat['Difference'].min(), 4)
                maxBela_Order=round(df_aggregat['Difference'].max(), 4)
                medianBela_Order=round(df_aggregat['Difference'].max(), 4)
                df.loc[df['run'] == j, f"BA_Bela_System_mean"]=meanBela_Order
                df.loc[df['run'] == j, f"BA_Bela_System_std"]=stdBela_Order
                df.loc[df['run'] == j, f"BA_Bela_System_min"]=minBela_Order
                df.loc[df['run'] == j, f"BA_Bela_System_max"]=maxBela_Order
                df.loc[df['run'] == j, f"BA_Bela_System_median"]=medianBela_Order

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Order_Output"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Order_Output"] - df_aggregat[
                    f"Order_Output"].shift(1)
                meanLeistung_Order=round(df_aggregat['Difference'].mean(), 4)
                stdLeistung_Order=round(df_aggregat['Difference'].std(), 4)
                minLeistung_Order=round(df_aggregat['Difference'].min(), 4)
                maxLeistung_Order=round(df_aggregat['Difference'].max(), 4)
                medianLeistung_Order=round(df_aggregat['Difference'].median(), 4)
                df.loc[df['run'] == j, f"BA_Leistung_System_mean"]=meanLeistung_Order
                df.loc[df['run'] == j, f"BA_Leistung_System_std"]=stdLeistung_Order
                df.loc[df['run'] == j, f"BA_Leistung_System_min"]=minLeistung_Order
                df.loc[df['run'] == j, f"BA_Leistung_System_max"]=maxLeistung_Order
                df.loc[df['run'] == j, f"BA_Leistung_System_median"]=medianLeistung_Order

                meanBestand_Order=round(df2[f"Order_WIP_Total"].mean(), 4)
                stdBestand_Order=round(df2[f"Order_WIP_Total"].std(), 4)
                minBestand_Order=round(df2[f"Order_WIP_Total"].min(), 4)
                maxBestand_Order=round(df2[f"Order_WIP_Total"].max(), 4)
                medianBestand_Order=round(df2[f"Order_WIP_Total"].median(), 4)
                df.loc[df['run'] == j, f"BA_Bestand_System_mean"]=meanBestand_Order
                df.loc[df['run'] == j, f"BA_Bestand_System_std"]=stdBestand_Order
                df.loc[df['run'] == j, f"BA_Bestand_System_min"]=minBestand_Order
                df.loc[df['run'] == j, f"BA_Bestand_System_max"]=maxBestand_Order
                df.loc[df['run'] == j, f"BA_Bestand_System_median"]=medianBestand_Order

            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                minZeit=int(round(df2["time"].min()))
                maxZeit=int(round(df2["time"].max()))

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Load_Input_{work_centre}"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Load_Input_{work_centre}"] - df_aggregat[
                    f"Load_Input_{work_centre}"].shift(1)
                meanBela_Load=round(df_aggregat['Difference'].mean(), 4)
                stdBela_Load=round(df_aggregat['Difference'].std(), 4)
                minBela_Load=round(df_aggregat['Difference'].min(), 4)
                maxBela_Load=round(df_aggregat['Difference'].max(), 4)
                medianBela_Load=round(df_aggregat['Difference'].max(), 4)
                df.loc[df['run'] == j, f"BL_Bela_{work_centre}_mean"]=meanBela_Load
                df.loc[df['run'] == j, f"BL_Bela_{work_centre}_std"]=stdBela_Load
                df.loc[df['run'] == j, f"BL_Bela_{work_centre}_min"]=minBela_Load
                df.loc[df['run'] == j, f"BL_Bela_{work_centre}_max"]=maxBela_Load
                df.loc[df['run'] == j, f"BL_Bela_{work_centre}_median"]=medianBela_Load

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Load_Output_{work_centre}"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Load_Output_{work_centre}"] - df_aggregat[
                    f"Load_Output_{work_centre}"].shift(1)
                meanLeistung_Load=round(df_aggregat['Difference'].mean(), 4)
                stdLeistung_Load=round(df_aggregat['Difference'].std(), 4)
                minLeistung_Load=round(df_aggregat['Difference'].min(), 4)
                maxLeistung_Load=round(df_aggregat['Difference'].max(), 4)
                medianLeistung_Load=round(df_aggregat['Difference'].median(), 4)
                df.loc[df['run'] == j, f"BL_Leistung_{work_centre}_mean"]=meanLeistung_Load
                df.loc[df['run'] == j, f"BL_Leistung_{work_centre}_std"]=stdLeistung_Load
                df.loc[df['run'] == j, f"BL_Leistung_{work_centre}_min"]=minLeistung_Load
                df.loc[df['run'] == j, f"BL_Leistung_{work_centre}_max"]=maxLeistung_Load
                df.loc[df['run'] == j, f"BL_Leistung_{work_centre}_median"]=medianLeistung_Load

                meanBestand_Load=round(df2[f"Load_WIP_{work_centre}"].mean(), 4)
                stdBestand_Load=round(df2[f"Load_WIP_{work_centre}"].std(), 4)
                minBestand_Load=round(df2[f"Load_WIP_{work_centre}"].min(), 4)
                maxBestand_Load=round(df2[f"Load_WIP_{work_centre}"].max(), 4)
                medianBestand_Load=round(df2[f"Load_WIP_{work_centre}"].median(), 4)
                df.loc[df['run'] == j, f"BL_Bestand_{work_centre}_mean"]=meanBestand_Load
                df.loc[df['run'] == j, f"BL_Bestand_{work_centre}_std"]=stdBestand_Load
                df.loc[df['run'] == j, f"BL_Bestand_{work_centre}_min"]=minBestand_Load
                df.loc[df['run'] == j, f"BL_Bestand_{work_centre}_max"]=maxBestand_Load
                df.loc[df['run'] == j, f"BL_Bestand_{work_centre}_median"]=medianBestand_Load

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Order_Input_{work_centre}"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Order_Input_{work_centre}"] - df_aggregat[
                    f"Order_Input_{work_centre}"].shift(1)
                meanBela_Order=round(df_aggregat['Difference'].mean(), 4)
                stdBela_Order=round(df_aggregat['Difference'].std(), 4)
                minBela_Order=round(df_aggregat['Difference'].min(), 4)
                maxBela_Order=round(df_aggregat['Difference'].max(), 4)
                medianBela_Order=round(df_aggregat['Difference'].max(), 4)
                df.loc[df['run'] == j, f"BA_Bela_{work_centre}_mean"]=meanBela_Order
                df.loc[df['run'] == j, f"BA_Bela_{work_centre}_std"]=stdBela_Order
                df.loc[df['run'] == j, f"BA_Bela_{work_centre}_min"]=minBela_Order
                df.loc[df['run'] == j, f"BA_Bela_{work_centre}_max"]=maxBela_Order
                df.loc[df['run'] == j, f"BA_Bela_{work_centre}_median"]=medianBela_Order

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Order_Output_{work_centre}"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Order_Output_{work_centre}"] - df_aggregat[
                    f"Order_Output_{work_centre}"].shift(1)
                meanLeistung_Order=round(df_aggregat['Difference'].mean(), 4)
                stdLeistung_Order=round(df_aggregat['Difference'].std(), 4)
                minLeistung_Order=round(df_aggregat['Difference'].min(), 4)
                maxLeistung_Order=round(df_aggregat['Difference'].max(), 4)
                medianLeistung_Order=round(df_aggregat['Difference'].median(), 4)
                df.loc[df['run'] == j, f"BA_Leistung_{work_centre}_mean"]=meanLeistung_Order
                df.loc[df['run'] == j, f"BA_Leistung_{work_centre}_std"]=stdLeistung_Order
                df.loc[df['run'] == j, f"BA_Leistung_{work_centre}_min"]=minLeistung_Order
                df.loc[df['run'] == j, f"BA_Leistung_{work_centre}_max"]=maxLeistung_Order
                df.loc[df['run'] == j, f"BA_Leistung_{work_centre}_median"]=medianLeistung_Order

                meanBestand_Order=round(df2[f"Order_WIP_{work_centre}"].mean(), 4)
                stdBestand_Order=round(df2[f"Order_WIP_{work_centre}"].std(), 4)
                minBestand_Order=round(df2[f"Order_WIP_{work_centre}"].min(), 4)
                maxBestand_Order=round(df2[f"Order_WIP_{work_centre}"].max(), 4)
                medianBestand_Order=round(df2[f"Order_WIP_{work_centre}"].median(), 4)
                df.loc[df['run'] == j, f"BA_Bestand_{work_centre}_mean"]=meanBestand_Order
                df.loc[df['run'] == j, f"BA_Bestand_{work_centre}_std"]=stdBestand_Order
                df.loc[df['run'] == j, f"BA_Bestand_{work_centre}_min"]=minBestand_Order
                df.loc[df['run'] == j, f"BA_Bestand_{work_centre}_max"]=maxBestand_Order
                df.loc[df['run'] == j, f"BA_Bestand_{work_centre}_median"]=medianBestand_Order

        return