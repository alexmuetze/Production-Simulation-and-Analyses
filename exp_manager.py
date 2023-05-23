"""
Project: Thesis_Alexander_Muetze_Final
Based on: Arno Kasper (https://github.com/ArnoKasper/ProcessSim)
Edited and Programmed by: Alexander Muetze
Version: 1.0
"""

"""
- Execution of an experiment
- Called by batch manager
- Save of experiment data
- Creation of Data Analysis (Operating Curves, Scatter Diagrams, etc.)
- Trigger of further processing
- Variable Names and Analyses are in German
"""

import socket
import random
import os
import simulationmodel as sim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

class Experiment_Manager(object):

    # Create a batch of experiments with an upper an lower limit--------------------------------------------------------
    def __init__(self, lower, upper):
        """
        initialize experiments integers
        """
        self.lower = lower
        self.upper = upper

        if self.lower > self.upper:
            raise Exception("lower exp number higher than upper exp number")

        self.total_name = "Exp_Series"
        self.pkl_name = "Exp_PKL"
        self.pkl_name_mean = "Exp_PKL_Mean"
        self.total_experiment_database = None
        self.pkl_database = None
        self.pkl_database_mean =None
        self.count_experiment = 0
        self.exp_manager()

    # Create a batch of experiments with an upper an lower limit---------------------------------------------------------
    def exp_manager(self):
        """
        define the experiment manager who controls the simulation model
        """

        # use a loop to illiterate multiple experiments from the exp_dat list
        for i in range(self.lower, (self.upper + 1)):
            # import simulation model and run
            self.sim = sim.SimulationModel(exp_number=i)
            self.create_subfolder(self.sim.model_panel.new_folder)
            self.sim.sim_function()
            self.num_export = 0

            self.name = "Exp_Rank_" + str(i)
            self.sim.data.calculation_of_ranks()

            # export the ranked production data if needed
            #self.saving_exp(self.sim.data.calculated_ranks_database, self.name)

            # calculation planned_values
            self.sim.data.create_planned_values(self.sim.model_panel.NUMBER_OF_RUNS)

            # join of planned values and ranked values to run database
            for j in range(1, self.sim.model_panel.NUMBER_OF_RUNS+1):
                self.sim.continous_run_db[j - 1] =self.sim.data.join_planned_values(self.sim.continous_run_db[j - 1])
                self.sim.detailled_run_db[j - 1] =self.sim.data.join_ranks(self.sim.detailled_run_db[j - 1])

            # aggregate planned and ranked values for run overview
            self.sim.data.append_aggregated_plan_and_rank_data()

            # calculation of load data (input, output, WIP, load) over time (mean, std, min, max)
            self.sim.data.append_BELA_data()

            self.name="Exp_" + str(i)

            # export entire experiment (all runs) database if needed. Attention! Big file!
            #self.saving_exp(self.sim.data.experiment_database, self.name)


            # definition of number of runs to be analysed in depth
            if self.sim.model_panel.NUMBER_OF_RUNS > 5:
                self.num_export = 1
            else:
                self.num_export = self.sim.model_panel.NUMBER_OF_RUNS


            # export of detailled experiment database (all orders) and continous_run_database (hourly data)
            for j in range(1, self.num_export+1):
                self.name = "Exp_" + str(i) + "_Run_" + str(j) + "_Continuous"
                self.saving_exp(self.sim.continous_run_db[j-1], self.name)

            for j in range(1, self.num_export+1):
                self.name = "Exp_" + str(i) + "_Run_" + str(j) + "_Detailled"
                self.saving_exp(self.sim.detailled_run_db[j-1], self.name)


            # export of Throughput Diagram, Load Analysation and DueDate-Deviation if needed
            # ATTENTION! Needed to be specified for an experiment (here exp 15 of a batch of 20)
            if i == self.lower + 6:
                print("Creating: DUDI")
                self.DUDI(self.sim.detailled_run_db[0], self.sim.continous_run_db[0],
                                     self.total_experiment_database)
                print("Creating: DUDI_ung")
                self.DUDI_ung(self.sim.detailled_run_db[0], self.sim.continous_run_db[0],
                                     self.total_experiment_database)
                print("Creating: BELA")
                self.BELA(self.sim.detailled_run_db[0], self.sim.continous_run_db[0])
                print("Creating: TAX_System")
                self.TAX_Gesamtsystem(self.sim.detailled_run_db[0])
                print("Creating: TAX")
                self.TAX_ASys(self.sim.detailled_run_db[0])


            # define and create entire experiment database (all experiment of batch)
            if self.total_experiment_database is None:
                self.total_experiment_database = self.sim.data.experiment_database
            else:
                self.total_experiment_database = pd.concat([self.total_experiment_database, self.sim.data.experiment_database], ignore_index=True)

            # analysations of entire batch (e.g. operating curves) Attention! Needs several different mean WIP-levels
            # e.g. through different utilisation levels in the experiments of a batch
            if i == self.upper:
                self.saving_exp(self.total_experiment_database, self.total_name)

                print("Creating: PKL_mean")
                self.PKL_mean(self.total_experiment_database)
                print("Creating: PKL")
                self.PKL(self.total_experiment_database)

                self.saving_exp(self.pkl_database, self.pkl_name)
                self.saving_exp(self.pkl_database_mean, self.pkl_name_mean)

                print("Creating: Routing_Matrix")
                self.MaterialflussMatrix(self.sim.detailled_run_db, self.sim.continous_run_db, self.total_experiment_database)

    # Saving Experiment Function----------------------------------------------------------------------------------------
    def saving_exp(self, database, name):
        """
        save all the experiment data versions
        :return: void
        """
        # initialize params
        df = database
        file_version = ".csv"  # ".xlsx"#".csv"#

        # get file directory
        path = self.get_directory()

        # create the experimental name
        exp_name = self.sim.model_panel.experiment_name + name

        # save file
        file = path + exp_name + file_version
        try:
            # save as csv file
            if file_version == ".csv":
                self.save_database_csv(file=file, database=df)

            # save as excel file
            elif file_version == ".xlsx":
                self.save_database_xlsx(file=file, database=df)

        except PermissionError:
            # failed to save, make a random addition to the name to save anyway
            from string import ascii_lowercase, digits
            random_genetator = random.Random()
            random_name = "random_"
            strings = []
            strings[:0] = ascii_lowercase + digits
            name_lenght = random_genetator.randint(1, len(strings) + 1)

            # build the name
            for j in range(0, name_lenght):
                random_genetator.shuffle(strings)
                random_name += strings[j]

            # change original name
            file = path + random_name + exp_name + file_version

            # save as csv file
            if file_version == ".csv":
                self.save_database_csv(file=file, database=df)

            # save as excel file
            elif file_version == ".xlsx":
                self.save_database_xlsx(file=file, database=df)

            # notify the user
            warnings.warn(f"Permission Error, saved with name {random_name + exp_name}", Warning)

        # add the experiment number for the next experiment
        self.count_experiment += 1

        if self.sim.model_panel.print_results:
            """
            try:
                print(f"\nresults of experiment {exp_name}:")
                print(df.iloc[:, [0, 2, 3, 4, 7, 9, 10, 11, 13, 14,15, 16]].describe().loc[['mean']].to_string(index=False))
                print("\n")
            except (KeyError, IndexError):
                print("could not print simulation results")
            """

        print(f"simulation data saved with name:    {exp_name}")

        # counter if needed
        # if self.sim.print_info:
        #    print(f"\tinput this experiment:      {self.sim.data_continous_run.order_input_counter}")
        #    print(f"\toutput this experiment:     {self.sim.data_continous_run.order_output_counter}")

    # Saving Procedures-------------------------------------------------------------------------------------------------
    def save_database_csv(self, file, database):
        database.to_csv(file, index=False)

    def save_database_xlsx(self, file, database):
        writer = pd.ExcelWriter(file, engine='xlsxwriter')
        database.to_excel(writer, sheet_name='name', index=False)
        writer.save()

    # Options for Saving Data-------------------------------------------------------------------------------------------
    def get_directory(self):
        # define different path options
        machine_name = socket.gethostname()
        LUIS = False
        path = ""

        # find path for specific machine
        if machine_name == "IFA-MUE-U747":
            path = "C:/Users/mue/Desktop/Exp_Data/" + self.sim.model_panel.new_folder
        elif machine_name == "IFA-HYPERV-LF-2":
            path = "C:/Users/mue/Desktop/Exp_Data/" + self.sim.model_panel.new_folder
        elif machine_name in ["WKS033389", "WKS052605"]:
            path = "C:/Users/P288125/Dropbox/Professioneel/Research/Results/test/" + self.sim.model_panel.new_folder
        elif LUIS == True:
            path = "/bigwork/nhk2mue1/SimulationData/PRD/" + self.sim.model_panel.new_folder
        else:
            warnings.warn(f"{machine_name} is an unknown machine name ", Warning)
            path = os.path.abspath(os.getcwd()) + "\\"
            print(f"files are saved in {path}")
        return path

    # Calculation and printing of Operating Curves for all work centers-------------------------------------------------
    def PKL(self, aggregated):
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.axes_style("darkgrid")
        sns.color_palette("tab10")
        # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Read File in Case of seperate use
        df= aggregated

        # If Filtering for specific run is neccessary or wanted
        df2=df.loc[df['run'] == 1]
        df=df2


        df["Variationskoeffizient_Freigabe_Quadrat"]=(df["std_interrelease_time"] / df["mean_interrelease_time"]) ** 2

        for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
            plt.clf()
            plt.cla()
            L_max= self.sim.model_panel.STANDARDCAPACITY
            # Release = df["Release"].iloc[0]
            Release= self.sim.policy_panel.release_control_method
            BImin_mean=df[f"ideal_minimum_wip_{work_centre}"].mean()
            BImin_std=df[f"ideal_minimum_wip_{work_centre}"].std()
            ZDF_mean=df[f"mean_operation_time_{work_centre}"].mean()
            ZDF_std=df[f"std_operation_time_{work_centre}"].mean()
            df[f"Variationskoeffizient_Zugang_Quadrat_{work_centre}"]=(df[f"std_interarrival_time_{work_centre}"] / df[
                f"mean_interarrival_time_{work_centre}"]) ** 2
            df[f"Variationskoeffizient_Prozesszeiten_Quadrat_{work_centre}"]=(df[f"std_process_time_{work_centre}"] /
                                                                              df[
                                                                                  f"mean_process_time_"
                                                                                  f"{work_centre}"]) ** 2

            VKF_mean=round(df["Variationskoeffizient_Freigabe_Quadrat"].mean(), 2)
            VKF_std=round(df["Variationskoeffizient_Freigabe_Quadrat"].std(), 2)
            VKA_mean=round(df[f"Variationskoeffizient_Zugang_Quadrat_{work_centre}"].mean(), 2)
            VKA_std=round(df[f"Variationskoeffizient_Zugang_Quadrat_{work_centre}"].std(), 2)
            VKP_mean=round(df[f"Variationskoeffizient_Prozesszeiten_Quadrat_{work_centre}"].mean(), 2)
            VKP_std=round(df[f"Variationskoeffizient_Prozesszeiten_Quadrat_{work_centre}"].std(), 2)

            list_avg_wip=[]
            list_avg_output=[]
            list_avg_reach=[]
            list_avg_ttp=[]

            list_avg_wip=df.groupby('Planned_Utilization')[f"mean_load_{work_centre}"].mean()
            list_avg_output=df.groupby('Planned_Utilization')[f"utilization_{work_centre}"].mean() * L_max
            list_avg_reach=list_avg_wip / list_avg_output
            list_avg_ttp=list_avg_reach - df.groupby('Planned_Utilization')[
                f"mean_operation_time_{work_centre}"].mean() * \
                         df.groupby('Planned_Utilization')[f"std_operation_time_{work_centre}"].mean() * \
                         df.groupby('Planned_Utilization')[f"std_operation_time_{work_centre}"].mean() / \
                         df.groupby('Planned_Utilization')[f"mean_operation_time_{work_centre}"].mean() / \
                         df.groupby('Planned_Utilization')[f"mean_operation_time_{work_centre}"].mean()

            zipped=list(zip(list_avg_wip, list_avg_output, list_avg_reach, list_avg_ttp))
            result=pd.DataFrame(zipped, columns=['Bestand', 'Leistung', 'Reichweite', 'FIFO-Durchlaufzeit'])
            print(result)

            Schritt=0
            calphaEbene=[]

            c=0.25
            c=round(c, 2)
            Fehlerspeicher_groß=10000000000000
            for alpha in range(1, 200, 1):
                df2=result.copy()
                Fehler=0.0
                Schritt=Schritt + 1

                for i in range(len(df2)):
                    b=df2.loc[i, "Bestand"]
                    L=df2.loc[i, "Leistung"]

                    func=lambda x: abs(
                        BImin_mean * x / self.sim.model_panel.STANDARDCAPACITY + BImin_mean * alpha * (np.abs(1 - (np.abs(1 - x / self.sim.model_panel.STANDARDCAPACITY) ** c)) ** (1 / c)) - b)

                    x_guess=8
                    bnds=[(1, 8)]
                    x_solution= scipy.optimize.minimize(func, x_guess, bounds=bnds)

                    df2.loc[i, "Fehler"]=abs(x_solution.x - L)

                Fehler=df2['Fehler'].sum()
                calphaEbene.insert(Schritt, [c, alpha, Fehler])
                if Fehler <= Fehlerspeicher_groß:
                    Fehlerspeicher_groß=Fehler
                else:
                    break

            calpha=pd.DataFrame(calphaEbene)

            calpha2=calpha[2].idxmin()

            c25=calpha._get_value(calpha2, 0)
            alpha25=calpha._get_value(calpha2, 1)
            Fehlermin25=calpha._get_value(calpha2, 2)

            copt=c25
            alphaopt=alpha25

            for c in np.arange(0.26, 0.6, 0.01):
                Fehlerspeicher=100000000000
                c=round(c, 2)
                for alpha in range(1, alphaopt + 1, 1):
                    df2=result.copy()
                    Fehler=0.0
                    Schritt=Schritt + 1

                    for i in range(len(df2)):
                        b=df2.loc[i, "Bestand"]
                        L=df2.loc[i, "Leistung"]

                        func=lambda x: abs(
                            BImin_mean * x / self.sim.model_panel.STANDARDCAPACITY + BImin_mean * alpha * (
                                        np.abs(1 - (np.abs(1 - x / self.sim.model_panel.STANDARDCAPACITY) ** c)) ** (1 / c)) - b)

                        x_guess=8
                        bnds=[(1, 8)]
                        x_solution=scipy.optimize.minimize(func, x_guess, bounds=bnds)

                        df2.loc[i, "Fehler"]=abs(x_solution.x - L)

                    Fehler=df2['Fehler'].sum()
                    calphaEbene.insert(Schritt, [c, alpha, Fehler])
                    if Fehler <= Fehlerspeicher:
                        Fehlerspeicher=Fehler
                    else:
                        break

                calpha=pd.DataFrame(calphaEbene)
                calpha2=calpha[2].idxmin()

                Fehlermin=calpha._get_value(calpha2, 2)
                if Fehlermin <= Fehlerspeicher_groß:
                    Fehlerspeicher_groß=Fehlermin
                    copt=calpha._get_value(calpha2, 0)
                    alphaopt=calpha._get_value(calpha2, 1)
                else:
                    break

            if copt == 0.25:
                for c in np.arange(0.24, 0.1, -0.01):
                    Fehlerspeicher=100000000000
                    c=round(c, 2)
                    for alpha in range(alphaopt, 200, 1):
                        df2=result.copy()
                        Fehler=0.0
                        Schritt=Schritt + 1

                        for i in range(len(df2)):
                            b=df2.loc[i, "Bestand"]
                            L=df2.loc[i, "Leistung"]

                            func=lambda x: abs(
                                BImin_mean * x / self.sim.model_panel.STANDARDCAPACITY + BImin_mean * alpha * (
                                            np.abs(1 - (np.abs(1 - x / self.sim.model_panel.STANDARDCAPACITY) ** c)) ** (1 / c)) - b)

                            x_guess=8
                            bnds=[(1, 8)]
                            x_solution=scipy.optimize.minimize(func, x_guess, bounds=bnds)

                            df2.loc[i, "Fehler"]=abs(x_solution.x - L)

                        Fehler=df2['Fehler'].sum()
                        calphaEbene.insert(Schritt, [c, alpha, Fehler])
                        if Fehler <= Fehlerspeicher:
                            Fehlerspeicher=Fehler
                        else:
                            break

                    calpha=pd.DataFrame(calphaEbene)
                    calpha2=calpha[2].idxmin()

                    Fehlermin=calpha._get_value(calpha2, 2)
                    if Fehlermin <= Fehlerspeicher_groß:
                        Fehlerspeicher_groß=Fehlermin
                        copt=calpha._get_value(calpha2, 0)
                        alphaopt=calpha._get_value(calpha2, 1)
                    else:
                        break

            mean_Fehler_25=round(Fehlermin25 / len(result), 3)
            mean_Fehler=round(Fehlermin / len(result), 3)

            mylist=np.arange(0, 1, 0.00001).tolist()
            df3=pd.DataFrame(mylist)
            df3.columns=['tWert']
            df3["Leistung"]=L_max * (1 - (1 - df3.tWert ** copt) ** (1 / copt))
            df3["Bestand"]=BImin_mean * (1 - (1 - df3.tWert ** copt) ** (1 / copt)) + BImin_mean * alphaopt * df3.tWert
            df3["Leistung25"]=L_max * (1 - (1 - df3.tWert ** c25) ** (1 / c25))
            df3["Bestand25"]=BImin_mean * (1 - (1 - df3.tWert ** c25) ** (1 / c25)) + BImin_mean * alpha25 * df3.tWert
            df3["Reichweite"]=df3["Bestand"] / df3["Leistung"]
            df3.loc[0]=0, 0, 0, 0, 0, BImin_mean / L_max
            df3["FIFO-Durchlaufzeit"]=df3["Reichweite"] - ZDF_mean * ((ZDF_std / ZDF_mean) ** 2)

            fig, ax=plt.subplots(1, 1)
            # Teilt die x-Achse der Y-Achsen und generiert Sie eine sekundäre Achse
            ax_sub=ax.twinx()
            # Zeichnen der Daten
            l1,=ax.plot(df3["Bestand"], df3["Leistung"], color='#1f77b4', label='Leistung');
            l2,=ax_sub.plot(df3["Bestand"], df3["Reichweite"], color='#ff7f0e', label='Reichweite');
            l3,=ax_sub.plot(df3["Bestand"], df3["FIFO-Durchlaufzeit"], color='#2ca02c', label='Durchlaufzeit');
            l4,=ax.plot(df2["Bestand"], df2["Leistung"], 'o', color='#1f77b4', label="Betriebspunkte");
            l5,=ax_sub.plot(df2["Bestand"], df2["Reichweite"], 'o', color='#ff7f0e', label="Betriebspunkte");
            l6,=ax_sub.plot(df2["Bestand"], df2["FIFO-Durchlaufzeit"], 'o', color='#2ca02c', label="Betriebspunkte");
            l7,=ax.plot(df3["Bestand25"], df3["Leistung25"], '--', color='#1f77b4', label='Leistung-C025');

            # box=ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
            plt.legend(handles=[l1, l7, l2, l3, l4, l5, l6],
                       labels=['Leistungskennlinie', 'L.-Kennlinie c=0.25', 'Reichweite', 'Durchlaufzeit',
                               'Betriebspunkte-Leistung',
                               'Betriebspunkte-Reichweite', 'Betriebspunkte-DLZ'], loc='lower right')

            ax.axvline(x=BImin_mean, ymin=0, ymax=0.9, color="red", linestyle="--", label="Idealer Mindestbestand");
            ax.annotate('BI_min = %s Std.' % (round(BImin_mean, 2)), xy=(BImin_mean, 8 + 0.15), ha='center')

            ax.set_ylabel("Leistung [Std./BKT]")
            ax_sub.set_ylabel("Reichweite, Durchlaufzeit [BKT]");
            ax.set_xlabel("Umlaufbestand [Std.]")

            y2max=round(df3["Reichweite"].max() + 1, 2)

            ax.set_ylim(0, 9)
            ax.set_yticks(range(0, 9, 1))

            y2max=round(df2["Reichweite"].max() + 1, 2)
            mult=int(round((y2max // 9) + 1, 0))

            ax_sub.set_ylim(0, mult * 9)
            ax_sub.set_yticks(range(0, mult * 9, mult))

            xmax=round(df2["Bestand"].max()) + 5
            ax.set_xlim(0, xmax)

            ax.tick_params(axis='y')
            ax_sub.tick_params(axis='y')

            alpha=round(alphaopt)
            c=round(copt, 2)
            alpha025=round(alpha25)
            c025=round(c25, 2)

            ax.set_title(
                'Freigabeverfahren: %s | Produktionskennlinie: Arbeitssystem_%s \n($\\alpha_{1}^{opt}=$%s,'
                '$c^{opt}$=%s) || ($\\alpha_{1}^{25}=$%s,$c^{25}$=%s)' % (
                    Release, work_centre, alpha, c, alpha025, c025))

            textstring='$c_{Freigabe,mw}^{2}$' ": %s | "'$c_{Freigabe,std}^{2}$' ": %s\n" '$c_{ZAZ,mw}^{2}$' ": %s | " \
                       ""'$c_{ZAZ,std}^{2}$' ": %s\n" '$c_{ZAU,mw}^{2}$' ": %s | "'$c_{ZAU,std}^{2}$' ": %s\n" \
                       r'$\Delta$$L_{m,opt,abs}$' ": %s | "r'$\Delta$$L_{m,25,abs}$' ": %s" % (
                VKF_mean, VKF_std, VKA_mean, VKA_std, VKP_mean, VKP_std, mean_Fehler, mean_Fehler_25)
            props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=1)
            ax_sub.text(0.80, 0.6, textstring, transform=ax.transAxes, fontsize=12, horizontalalignment='center',
                        verticalalignment='center', bbox=props)

            plotName="Arbeitssystem_" + str(work_centre) + "_" + 'PKL' + '.svg'

            path=self.get_directory()

            subfolder = "Plot/PKL/"

            # save file
            file= path+subfolder+self.sim.model_panel.experiment_name+plotName

            fig.savefig(file)
            print(f"Save {file}")
            plt.close(fig)
            df[f"c_opt_{work_centre}"] = c
            df[f"alpha_opt_{work_centre}"] = alpha
            df[f"PKL_Fehler_opt_{work_centre}"] = mean_Fehler
            df[f"c_25_{work_centre}"]=c025
            df[f"alpha_25_{work_centre}"]= alpha025
            df[f"PKL_Fehler_25_{work_centre}"]=mean_Fehler_25

        self.pkl_database=df

        return

    # Calculation and printing of Operating Curves for all work centers based on mean of all runs-----------------------
    def PKL_mean(self, aggregated):
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.axes_style("darkgrid")
        sns.color_palette("tab10")

        # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Read File in Case of seperate use
        df= aggregated

        df["Variationskoeffizient_Freigabe_Quadrat"]=(df["std_interrelease_time"] / df["mean_interrelease_time"]) ** 2

        for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
            plt.clf()
            plt.cla()

            L_max= self.sim.model_panel.STANDARDCAPACITY
            # Release = df["Release"].iloc[0]
            Release= self.sim.policy_panel.release_control_method
            BImin_mean=df[f"ideal_minimum_wip_{work_centre}"].mean()
            BImin_std=df[f"ideal_minimum_wip_{work_centre}"].std()
            ZDF_mean=df[f"mean_operation_time_{work_centre}"].mean()
            ZDF_std=df[f"std_operation_time_{work_centre}"].mean()
            df[f"Variationskoeffizient_Zugang_Quadrat_{work_centre}"]=(df[f"std_interarrival_time_{work_centre}"] / df[
                f"mean_interarrival_time_{work_centre}"]) ** 2
            df[f"Variationskoeffizient_Prozesszeiten_Quadrat_{work_centre}"]=(df[f"std_process_time_{work_centre}"] /
                                                                              df[
                                                                                  f"mean_process_time_"
                                                                                  f"{work_centre}"]) ** 2

            VKF_mean=round(df["Variationskoeffizient_Freigabe_Quadrat"].mean(), 2)
            VKF_std=round(df["Variationskoeffizient_Freigabe_Quadrat"].std(), 2)
            VKA_mean=round(df[f"Variationskoeffizient_Zugang_Quadrat_{work_centre}"].mean(), 2)
            VKA_std=round(df[f"Variationskoeffizient_Zugang_Quadrat_{work_centre}"].std(), 2)
            VKP_mean=round(df[f"Variationskoeffizient_Prozesszeiten_Quadrat_{work_centre}"].mean(), 2)
            VKP_std=round(df[f"Variationskoeffizient_Prozesszeiten_Quadrat_{work_centre}"].std(), 2)

            list_avg_wip=[]
            list_avg_output=[]
            list_avg_reach=[]
            list_avg_ttp=[]

            list_avg_wip=df.groupby('Planned_Utilization')[f"mean_load_{work_centre}"].mean()
            list_avg_output=df.groupby('Planned_Utilization')[f"utilization_{work_centre}"].mean() * L_max
            list_avg_reach=list_avg_wip / list_avg_output
            list_avg_ttp=list_avg_reach - df.groupby('Planned_Utilization')[
                f"mean_operation_time_{work_centre}"].mean() * \
                         df.groupby('Planned_Utilization')[f"std_operation_time_{work_centre}"].mean() * \
                         df.groupby('Planned_Utilization')[f"std_operation_time_{work_centre}"].mean() / \
                         df.groupby('Planned_Utilization')[f"mean_operation_time_{work_centre}"].mean() / \
                         df.groupby('Planned_Utilization')[f"mean_operation_time_{work_centre}"].mean()

            zipped=list(zip(list_avg_wip, list_avg_output, list_avg_reach, list_avg_ttp))
            result=pd.DataFrame(zipped, columns=['Bestand', 'Leistung', 'Reichweite', 'FIFO-Durchlaufzeit'])
            print(result)

            Schritt=0
            calphaEbene=[]

            c=0.25
            c=round(c, 2)
            Fehlerspeicher_groß=10000000000000
            for alpha in range(1, 200, 1):
                df2=result.copy()
                Fehler=0.0
                Schritt=Schritt + 1

                for i in range(len(df2)):
                    b=df2.loc[i, "Bestand"]
                    L=df2.loc[i, "Leistung"]

                    func=lambda x: abs(
                        BImin_mean * x / 8 + BImin_mean * alpha * (np.abs(1 - (np.abs(1 - x / 8) ** c)) ** (1 / c)) - b)

                    x_guess=8
                    bnds=[(1, 8)]
                    x_solution= scipy.optimize.minimize(func, x_guess, bounds=bnds)

                    df2.loc[i, "Fehler"]=abs(x_solution.x - L)

                Fehler=df2['Fehler'].sum()
                calphaEbene.insert(Schritt, [c, alpha, Fehler])
                if Fehler <= Fehlerspeicher_groß:
                    Fehlerspeicher_groß=Fehler
                else:
                    break

            calpha=pd.DataFrame(calphaEbene)

            calpha2=calpha[2].idxmin()

            c25=calpha._get_value(calpha2, 0)
            alpha25=calpha._get_value(calpha2, 1)
            Fehlermin25=calpha._get_value(calpha2, 2)

            copt=c25
            alphaopt=alpha25

            for c in np.arange(0.26, 0.6, 0.01):
                Fehlerspeicher=100000000000
                c=round(c, 2)
                for alpha in range(1, alphaopt + 1, 1):
                    df2=result.copy()
                    Fehler=0.0
                    Schritt=Schritt + 1

                    for i in range(len(df2)):
                        b=df2.loc[i, "Bestand"]
                        L=df2.loc[i, "Leistung"]

                        func=lambda x: abs(
                            BImin_mean * x / self.sim.model_panel.STANDARDCAPACITY + BImin_mean * alpha * (
                                        np.abs(1 - (np.abs(1 - x / self.sim.model_panel.STANDARDCAPACITY) ** c)) ** (1 / c)) - b)

                        x_guess=8
                        bnds=[(1, 8)]
                        x_solution=scipy.optimize.minimize(func, x_guess, bounds=bnds)

                        df2.loc[i, "Fehler"]=abs(x_solution.x - L)

                    Fehler=df2['Fehler'].sum()
                    calphaEbene.insert(Schritt, [c, alpha, Fehler])
                    if Fehler <= Fehlerspeicher:
                        Fehlerspeicher=Fehler
                    else:
                        break

                calpha=pd.DataFrame(calphaEbene)
                calpha2=calpha[2].idxmin()

                Fehlermin=calpha._get_value(calpha2, 2)
                if Fehlermin <= Fehlerspeicher_groß:
                    Fehlerspeicher_groß=Fehlermin
                    copt=calpha._get_value(calpha2, 0)
                    alphaopt=calpha._get_value(calpha2, 1)
                else:
                    break

            if copt == 0.25:
                for c in np.arange(0.24, 0.1, -0.01):
                    Fehlerspeicher=100000000000
                    c=round(c, 2)
                    for alpha in range(alphaopt, 200, 1):
                        df2=result.copy()
                        Fehler=0.0
                        Schritt=Schritt + 1

                        for i in range(len(df2)):
                            b=df2.loc[i, "Bestand"]
                            L=df2.loc[i, "Leistung"]

                            func=lambda x: abs(
                                BImin_mean * x / self.sim.model_panel.STANDARDCAPACITY + BImin_mean * alpha * (
                                            np.abs(1 - (np.abs(1 - x / self.sim.model_panel.STANDARDCAPACITY) ** c)) ** (1 / c)) - b)

                            x_guess=8
                            bnds=[(1, 8)]
                            x_solution=scipy.optimize.minimize(func, x_guess, bounds=bnds)

                            df2.loc[i, "Fehler"]=abs(x_solution.x - L)

                        Fehler=df2['Fehler'].sum()
                        calphaEbene.insert(Schritt, [c, alpha, Fehler])
                        if Fehler <= Fehlerspeicher:
                            Fehlerspeicher=Fehler
                        else:
                            break

                    calpha=pd.DataFrame(calphaEbene)
                    calpha2=calpha[2].idxmin()

                    Fehlermin=calpha._get_value(calpha2, 2)
                    if Fehlermin <= Fehlerspeicher_groß:
                        Fehlerspeicher_groß=Fehlermin
                        copt=calpha._get_value(calpha2, 0)
                        alphaopt=calpha._get_value(calpha2, 1)
                    else:
                        break

            mean_Fehler_25=round(Fehlermin25 / len(result), 3)
            mean_Fehler=round(Fehlermin / len(result), 3)

            mylist=np.arange(0, 1, 0.00001).tolist()
            df3=pd.DataFrame(mylist)
            df3.columns=['tWert']
            df3["Leistung"]=L_max * (1 - (1 - df3.tWert ** copt) ** (1 / copt))
            df3["Bestand"]=BImin_mean * (1 - (1 - df3.tWert ** copt) ** (1 / copt)) + BImin_mean * alphaopt * df3.tWert
            df3["Leistung25"]=L_max * (1 - (1 - df3.tWert ** c25) ** (1 / c25))
            df3["Bestand25"]=BImin_mean * (1 - (1 - df3.tWert ** c25) ** (1 / c25)) + BImin_mean * alpha25 * df3.tWert
            df3["Reichweite"]=df3["Bestand"] / df3["Leistung"]
            df3.loc[0]=0, 0, 0, 0, 0, BImin_mean / L_max
            df3["FIFO-Durchlaufzeit"]=df3["Reichweite"] - ZDF_mean * ((ZDF_std / ZDF_mean) ** 2)

            fig, ax=plt.subplots(1, 1)
            # Teilt die x-Achse der Y-Achsen und generiert Sie eine sekundäre Achse
            ax_sub=ax.twinx()
            # Zeichnen der Daten
            l1,=ax.plot(df3["Bestand"], df3["Leistung"], color='#1f77b4', label='Leistung');
            l2,=ax_sub.plot(df3["Bestand"], df3["Reichweite"], color='#ff7f0e', label='Reichweite');
            l3,=ax_sub.plot(df3["Bestand"], df3["FIFO-Durchlaufzeit"], color='#2ca02c', label='Durchlaufzeit');
            l4,=ax.plot(df2["Bestand"], df2["Leistung"], 'o', color='#1f77b4', label="Betriebspunkte");
            l5,=ax_sub.plot(df2["Bestand"], df2["Reichweite"], 'o', color='#ff7f0e', label="Betriebspunkte");
            l6,=ax_sub.plot(df2["Bestand"], df2["FIFO-Durchlaufzeit"], 'o', color='#2ca02c', label="Betriebspunkte");
            l7,=ax.plot(df3["Bestand25"], df3["Leistung25"], '--', color='#1f77b4', label='Leistung-C025');

            # box=ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
            plt.legend(handles=[l1, l7, l2, l3, l4, l5, l6],
                       labels=['Leistungskennlinie', 'L.-Kennlinie c=0.25', 'Reichweite', 'Durchlaufzeit',
                               'Betriebspunkte-Leistung',
                               'Betriebspunkte-Reichweite', 'Betriebspunkte-DLZ'], loc='lower right')

            ax.axvline(x=BImin_mean, ymin=0, ymax=0.9, color="red", linestyle="--", label="Idealer Mindestbestand");
            ax.annotate('BI_min = %s Std.' % (round(BImin_mean, 2)), xy=(BImin_mean, 8 + 0.15), ha='center')

            ax.set_ylabel("Leistung [Std./BKT]")
            ax_sub.set_ylabel("Reichweite, Durchlaufzeit [BKT]");
            ax.set_xlabel("Umlaufbestand [Std.]")

            y2max=round(df3["Reichweite"].max() + 1, 2)

            ax.set_ylim(0, 9)
            ax.set_yticks(range(0, 9, 1))

            y2max=round(df2["Reichweite"].max() + 1, 2)
            mult=int(round((y2max // 9) + 1, 0))

            ax_sub.set_ylim(0, mult * 9)
            ax_sub.set_yticks(range(0, mult * 9, mult))

            xmax=round(df2["Bestand"].max()) + 5
            ax.set_xlim(0, xmax)

            ax.tick_params(axis='y')
            ax_sub.tick_params(axis='y')

            alpha=round(alphaopt)
            c=round(copt, 2)
            alpha025=round(alpha25)
            c025=round(c25, 2)

            ax.set_title(
                'Freigabeverfahren: %s | Produktionskennlinie: Arbeitssystem_%s \n($\\alpha_{1}^{opt}=$%s,'
                '$c^{opt}$=%s) || ($\\alpha_{1}^{25}=$%s,$c^{25}$=%s)' % (
                    Release, work_centre, alpha, c, alpha025, c025))

            textstring='$c_{Freigabe,mw}^{2}$' ": %s | "'$c_{Freigabe,std}^{2}$' ": %s\n" '$c_{ZAZ,mw}^{2}$' ": %s | " \
                       ""'$c_{ZAZ,std}^{2}$' ": %s\n" '$c_{ZAU,mw}^{2}$' ": %s | "'$c_{ZAU,std}^{2}$' ": %s\n" \
                       r'$\Delta$$L_{m,opt,abs}$' ": %s | "r'$\Delta$$L_{m,25,abs}$' ": %s" % (
                VKF_mean, VKF_std, VKA_mean, VKA_std, VKP_mean, VKP_std, mean_Fehler, mean_Fehler_25)
            props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=1)
            ax_sub.text(0.80, 0.6, textstring, transform=ax.transAxes, fontsize=12, horizontalalignment='center',
                        verticalalignment='center', bbox=props)

            plotName="Arbeitssystem_" + str(work_centre) + "_" + 'PKL_mean' + '.svg'

            path=self.get_directory()

            subfolder = "Plot/PKL/"

            # save file
            file= path+subfolder+self.sim.model_panel.experiment_name+plotName

            fig.savefig(file)
            print(f"Save {file}")
            plt.close(fig)
            df[f"c_opt_{work_centre}"] = c
            df[f"alpha_opt_{work_centre}"] = alpha
            df[f"PKL_Fehler_opt_{work_centre}"] = mean_Fehler
            df[f"c_25_{work_centre}"]=c025
            df[f"alpha_25_{work_centre}"]= alpha025
            df[f"PKL_Fehler_25_{work_centre}"]=mean_Fehler_25

        self.pkl_database_mean=df

        return

    # Calculation and printing of Throughput Diagram and KPIs-----------------------------------------------------------
    def DUDI(self, detailled, continous, aggregated):

        sns.set(rc={'figure.figsize':(11.7,8.27)})
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.axes_style("darkgrid")
        sns.color_palette("tab10")

        for i in np.arange(0, 4, 1):
            df2 = continous
            df3= detailled
            mean = df2['time'].mean()
            df = pd.DataFrame()

            zoom = i

            if i == 0:
                df=df2.loc[(df2['time'] >= mean - 100) & (df2['time'] <= mean + 100)]
            elif i == 1:
                df=df2.loc[(df2['time'] >= mean - 30) & (df2['time'] <= mean + 30)]
            elif i == 2:
                df=df2.loc[(df2['time'] >= mean - 10) & (df2['time'] <= mean + 10)]
            else:
                df=df2.loc[(df2['time'] >= mean - 5) & (df2['time'] <= mean + 5)]

            # In case of seperate use
            Release = self.sim.policy_panel.release_control_method

            # Schleife für Produktionskennlinie
            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                plt.clf()
                plt.cla()
                x = df["time"]
                y = df[f"Planned_Load_Input_{work_centre}"]
                y1 = df[f"Load_Input_{work_centre}"]
                y2=  df[f"Load_Output_{work_centre}"]
                z = df[f"Load_WIP_{work_centre}"]

                fig, ax = plt.subplots(1, 1)
                # Teilt die x-Achse der Y-Achsen und generiert Sie eine sekundäre Achse
                ax_sub = ax.twinx()
                # Zeichnen der Daten
                l1, = ax.plot(x, y1, drawstyle='steps-post', color='#1f77b4', label='Zugang');
                l2, = ax.plot(x, y2, drawstyle='steps-post', color='#ff7f0e',label='Abgang');
                l3, = ax.plot(x, y, drawstyle='steps-post', linestyle='dashed', color='#1f77b4', label='Plan-Zugangs');
                l4, = ax_sub.plot(x, z, drawstyle='steps-post', color='#2ca02c',label='Bestand');

                #box=ax.get_position()
                #ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                plt.legend(handles=[l1,l2,l3,l4], labels=['Zugangskurve', 'Abgangskurve', 'Plan-Zugangsskurve', 'Bestandskurve'], loc='upper left')

                ax.set_ylabel("Arbeitsinhalt [Std.]")
                ax_sub.set_ylabel("Bestand [Std.]");
                ax.set_xlabel("Zeit [BKT]")

                stepsize = 50
                ymax = (round(y1.max()) // stepsize) * stepsize + stepsize
                ymin =(round(y1.min()) // stepsize) * stepsize + -stepsize

                ticks = (ymax - ymin) / stepsize

                ax.set_ylim(ymin, ymax)
                ax.set_yticks(range(ymin, ymax, stepsize))

                sichtfaktor = 3
                ymax2= z.max() * sichtfaktor
                ymin2= 0

                stepsize2 = round((ymax2 - ymin2)  // ticks) + 1

                ymax2 = int(round(stepsize2 * ticks))


                ax_sub.set_axisbelow(True)

                ax.set_axisbelow(True)
                ax_sub.set_ylim(ymin2, ymax2)
                ax_sub.set_yticks(range(0, ymax2, stepsize2))

                ax.tick_params(axis='y')
                ax_sub.tick_params(axis='y')
                ax.set_title('Freigabeverfahren: %s | Durchlaufdiagramm: Arbeitssystem_%s' %(Release,work_centre), pad=20, fontsize=16)

            # Mean ZAZ, STD ZAZ , MEAN ZAU, STD ZAU

                meanzaz = round(df3[f"operation_interarrival_time_{work_centre}"].mean(),2)
                stdzaz = round(df3[f"operation_interarrival_time_{work_centre}"].std(),2)
                meanzau=round(df3[f"process_time_{work_centre}"].mean(),2)
                stdzau=round(df3[f"process_time_{work_centre}"].std(),2)
                meanZDL =round(df3[f"throughput_time_{work_centre}"].mean(),2)
                stdZDL=round(df3[f"throughput_time_{work_centre}"].std(),2)
                zdlmg= df3[f"throughput_time_{work_centre}"]* df3[f"process_time_{work_centre}"]
                meanzdlmg = round(zdlmg.sum() / df3[f"process_time_{work_centre}"].sum(),2)

                meanLeistung = (df2[f"Load_Output_{work_centre}"].max() - df2[f"Load_Output_{work_centre}"].min()) / (df2["time"].max() - df2["time"].min())
                meanBestand = round(df2[f"Load_WIP_{work_centre}"].mean(),2)
                stdBestand=  round(df2[f"Load_WIP_{work_centre}"].std(),2)
                meanReichweite = round(meanBestand / meanLeistung,2)



                minZeit =  int(round(df2["time"].min()))
                maxZeit =  int(round(df2["time"].max()))

                df_aggregat = pd.DataFrame()
                df_aggregat["time"] = np.arange(minZeit, maxZeit+1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Load_Input_{work_centre}"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Load_Input_{work_centre}"] - df_aggregat[f"Load_Input_{work_centre}"].shift(1)

                meanBela = round(df_aggregat['Difference'].mean(),2)
                stdBela= round(df_aggregat['Difference'].std(),2)

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Load_Output_{work_centre}"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Load_Output_{work_centre}"] - df_aggregat[f"Load_Output_{work_centre}"].shift(
                    1)

                meanLeistung= round(df_aggregat['Difference'].mean(),2)
                stdLeistung= round(df_aggregat['Difference'].std(),2)
                auslastung = round(meanLeistung/8,2)
                natStreu = round(stdzau * 2**0.5,2)

               #
                textstring = '$ZAZ_{m}$' ": %s | "'$ZAZ_{std}}$' ": %s\n" '$ZAU_{m}$' ": %s | "'$ZAU_{std}$' ": %s\n" '$ZDL_{m}}$' ": %s | "'$ZDL_{std}}$' ": %s" %(meanzaz,stdzaz,meanzau,stdzau,meanZDL,stdZDL)
                textstring2='$Bela_{m}$' ": %s | "'$Bela_{std}}$' ": %s\n" '$L_{m}$' ": %s | "'$L_{std}$' ": %s\n" '$B_{m}}$' \
                           ": %s | "'$B_{std}}$' ": %s\n" '$R_{m}$' ": %s | "'$ZDL_{mg}$' ": %s\n" '$nat.streuung}$' ": %s " % (
                meanBela, stdBela, meanLeistung, stdLeistung, meanBestand, stdBestand, meanReichweite, meanzdlmg,natStreu)

                props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=1)
                ax_sub.text(0.85, 0.5, textstring, transform=ax.transAxes, fontsize=12, horizontalalignment='center',
                      verticalalignment='center', bbox=props)

                ax_sub.text(0.45, 0.98, textstring2, transform=ax.transAxes, fontsize=12, horizontalalignment='center',
                        verticalalignment='top', bbox=props)

                plotName = "Arbeitssystem_" + str(work_centre) + "_" + str(auslastung) + "_" + 'DuDi' + str(zoom) + '.svg'

                path=self.get_directory()

                subfolder="Plot/DUDI/"

                # save file
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName

                fig.savefig(file)
                print(f"Save {file}")
                plt.close(fig)

            plotsystem: bool = True

            if plotsystem == True:
                x = df["time"]
                y = df[f"Planned_Load_Release"]
                y3 = df[f"Load_Release"]
                y1 = df[f"Load_Input"]
                y2=  df[f"Load_Output"]
                z = df[f"Load_WIP_Total"]
                z2 = df[f"Load_WIP_Pool"]

                fig, ax = plt.subplots(1, 1)
                # Teilt die x-Achse der Y-Achsen und generiert Sie eine sekundäre Achse
                ax_sub = ax.twinx()
                # Zeichnen der Daten
                l1, = ax.plot(x, y1, drawstyle='steps-post', color='#d62728', label='Zugang');
                l2, = ax.plot(x, y2, drawstyle='steps-post', color='#ff7f0e',label='Abgang');
                l3, = ax.plot(x, y3, drawstyle='steps-post', color='#1f77b4',label='Freigabe');
                l4, = ax.plot(x, y, drawstyle='steps-post', linestyle='dashed', color='#1f77b4', label='Plan-Freigabe');
                l5, = ax_sub.plot(x, z, drawstyle='steps-post', color='#2ca02c',label='Bestand');
                l6, = ax_sub.plot(x, z2, drawstyle='steps-post', linestyle='dashed', color='#2ca02c',label='Bestand2');

                #box=ax.get_position()
                #ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                plt.legend(handles=[l1,l2,l3,l4,l5, l6], labels=['Zugangskurve', 'Abgangskurve', 'Freigabekurve', 'Plan-Freigabekurve', 'Bestandskurve_Gesamt', 'Bestandskurve_Pool'], loc='upper left')

                ax.set_ylabel("Arbeitsinhalt [Std.]")
                ax_sub.set_ylabel("Bestand [Std.]");
                ax.set_xlabel("Zeit [BKT]")

                stepsize = 200 * (4-zoom)
                ymax = (round(y1.max()) // stepsize) * stepsize + stepsize
                ymin =(round(y2.min()) // stepsize) * stepsize + -stepsize

                ticks = (ymax - ymin) / stepsize

                ax.set_ylim(ymin, ymax)
                ax.set_yticks(range(ymin, ymax, stepsize))

                sichtfaktor = 3
                ymax2= z.max() * sichtfaktor
                ymin2= 0

                stepsize2 = round((ymax2 - ymin2)  // ticks) + 1

                ymax2 = int(round(stepsize2 * ticks))


                ax_sub.set_axisbelow(True)

                ax.set_axisbelow(True)
                ax_sub.set_ylim(ymin2, ymax2)
                ax_sub.set_yticks(range(0, ymax2, stepsize2))

                ax.tick_params(axis='y')
                ax_sub.tick_params(axis='y')
                ax.set_title('Freigabeverfahren: %s | Durchlaufdiagramm: Gesamtsystem' %(Release), pad=20, fontsize=16)

                # Mean ZAZ, STD ZAZ , MEAN ZAU, STD ZAU

                meanzaz = round(df3[f"interarrival_time"].mean(),2)
                stdzaz = round(df3[f"interarrival_time"].std(),2)
                meanzau=round(df3[f"total_process_time"].mean(),2)
                stdzau=round(df3[f"total_process_time"].std(),2)
                meanZDL =round(df3[f"throughput_time"].mean(),2)
                stdZDL=round(df3[f"throughput_time"].std(),2)
                zdlmg= df3[f"throughput_time"]* df3[f"total_process_time"]
                meanzdlmg = round(zdlmg.sum() / df3[f"total_process_time"].sum(),2)

                meanLeistung = (df2[f"Load_Output"].max() - df2[f"Load_Output"].min()) / (df2["time"].max() - df2["time"].min())
                meanBestand = round(df2[f"Load_WIP_Total"].mean(),2)
                stdBestand=  round(df2[f"Load_WIP_Total"].std(),2)
                meanBestand_Pool =round(df2[f"Load_WIP_Pool"].mean(), 2)
                stdBestand_Pool =round(df2[f"Load_WIP_Pool"].std(), 2)
                meanReichweite = round(meanBestand / meanLeistung,2)

                minZeit =  int(round(df2["time"].min()))
                maxZeit =  int(round(df2["time"].max()))

                df_aggregat = pd.DataFrame()
                df_aggregat["time"] = np.arange(minZeit, maxZeit+1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Load_Input"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Load_Input"] - df_aggregat[f"Load_Input"].shift(1)

                meanBela = round(df_aggregat['Difference'].mean(),2)
                stdBela= round(df_aggregat['Difference'].std(),2)

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Load_Output"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Load_Output"] - df_aggregat[f"Load_Output"].shift(
                    1)

                meanLeistung= round(df_aggregat['Difference'].mean(),2)
                stdLeistung= round(df_aggregat['Difference'].std(),2)

                auslastung = round(meanLeistung/48, 2)

                natStreu = round(stdzau * 2**0.5,2)

                #
                textstring = '$ZAZ_{m}$' ": %s | "'$ZAZ_{std}}$' ": %s\n" '$ZAU_{m}$' ": %s | "'$ZAU_{std}$' ": %s\n" '$ZDL_{m}}$' ": %s | "'$ZDL_{std}}$' ": %s" %(meanzaz,stdzaz,meanzau,stdzau,meanZDL,stdZDL)
                textstring2='$Bela_{m}$' ": %s | "'$Bela_{std}}$' ": %s\n" '$L_{m}$' ": %s | "'$L_{std}$' ": %s\n" '$B_{m}}$' \
                           ": %s | "'$B_{std}}$' ": %s\n" '$B_{m,p}}$' \
                           ": %s | "'$B_{std,p}}$' ": %s\n" '$R_{m}$' ": %s | "'$ZDL_{mg}$' ": %s\n" '$nat.streuung}$' ": %s " % (
                meanBela, stdBela, meanLeistung, stdLeistung, meanBestand, stdBestand, meanBestand_Pool, stdBestand_Pool, meanReichweite, meanzdlmg,natStreu)

                props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=1)
                ax_sub.text(0.85, 0.4, textstring, transform=ax.transAxes, fontsize=12, horizontalalignment='center',
                      verticalalignment='center', bbox=props)

                ax_sub.text(0.45, 0.98, textstring2, transform=ax.transAxes, fontsize=12, horizontalalignment='center',
                        verticalalignment='top', bbox=props)

                plotName = "Gesamtsystem_" + str(auslastung) + "_" + 'DuDi' + str(zoom) +'.svg'

                path=self.get_directory()

                subfolder="Plot/DUDI/"

                # save file
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName

                fig.savefig(file)
                print(f"Save {file}")
                plt.close(fig)

    # Calculation and printing of Throughput Diagram and KPIs using unweighted data-------------------------------------
    def DUDI_ung(self, detailled, continous, aggregated):

        sns.set(rc={'figure.figsize':(11.7,8.27)})
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.axes_style("darkgrid")
        sns.color_palette("tab10")

        for i in np.arange(0, 4, 1):
            df2 = continous
            df3= detailled
            mean = df2['time'].mean()
            df = pd.DataFrame()

            zoom = i

            if i == 0:
                df=df2.loc[(df2['time'] >= mean - 100) & (df2['time'] <= mean + 100)]
            elif i == 1:
                df=df2.loc[(df2['time'] >= mean - 30) & (df2['time'] <= mean + 30)]
            elif i == 2:
                df=df2.loc[(df2['time'] >= mean - 10) & (df2['time'] <= mean + 10)]
            else:
                df=df2.loc[(df2['time'] >= mean - 5) & (df2['time'] <= mean + 5)]

            # In case of seperate use
            Release = self.sim.policy_panel.release_control_method

            # Schleife für Produktionskennlinie
            for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:
                plt.clf()
                plt.cla()
                x = df["time"]
                y = df[f"Planned_Order_Input_{work_centre}"]
                y1 = df[f"Order_Input_{work_centre}"]
                y2=  df[f"Order_Output_{work_centre}"]
                z = df[f"Order_WIP_{work_centre}"]

                fig, ax = plt.subplots(1, 1)
                # Teilt die x-Achse der Y-Achsen und generiert Sie eine sekundäre Achse
                ax_sub = ax.twinx()
                # Zeichnen der Daten
                l1, = ax.plot(x, y1, drawstyle='steps-post', color='#1f77b4', label='Zugang');
                l2, = ax.plot(x, y2, drawstyle='steps-post', color='#ff7f0e',label='Abgang');
                l3, = ax.plot(x, y, drawstyle='steps-post', linestyle='dashed', color='#1f77b4', label='Plan-Zugangs');
                l4, = ax_sub.plot(x, z, drawstyle='steps-post', color='#2ca02c',label='Bestand');

                #box=ax.get_position()
                #ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                plt.legend(handles=[l1,l2,l3,l4], labels=['Zugangskurve', 'Abgangskurve', 'Plan-Zugangsskurve', 'Bestandskurve'], loc='upper left')

                ax.set_ylabel("Aufträge [Stk.]")
                ax_sub.set_ylabel("Bestand [Stk.]");
                ax.set_xlabel("Zeit [BKT]")

                stepsize = 50
                ymax = (round(y1.max()) // stepsize) * stepsize + stepsize
                ymin =(round(y1.min()) // stepsize) * stepsize + -stepsize

                ticks = (ymax - ymin) / stepsize

                ax.set_ylim(ymin, ymax)
                ax.set_yticks(range(ymin, ymax, stepsize))

                sichtfaktor = 3
                ymax2= z.max() * sichtfaktor
                ymin2= 0

                stepsize2 = round((ymax2 - ymin2)  // ticks) + 1

                ymax2 = int(round(stepsize2 * ticks))


                ax_sub.set_axisbelow(True)

                ax.set_axisbelow(True)
                ax_sub.set_ylim(ymin2, ymax2)
                ax_sub.set_yticks(range(0, ymax2, stepsize2))

                ax.tick_params(axis='y')
                ax_sub.tick_params(axis='y')
                ax.set_title('Freigabeverfahren: %s | Durchlaufdiagramm (ungewichtet): Arbeitssystem_%s' %(Release,work_centre), pad=20, fontsize=16)

            # Mean ZAZ, STD ZAZ , MEAN ZAU, STD ZAU

                meanzaz = round(df3[f"operation_interarrival_time_{work_centre}"].mean(),2)
                stdzaz = round(df3[f"operation_interarrival_time_{work_centre}"].std(),2)
                meanzau=round(df3[f"process_time_{work_centre}"].mean(),2)
                stdzau=round(df3[f"process_time_{work_centre}"].std(),2)
                meanZDL =round(df3[f"throughput_time_{work_centre}"].mean(),2)
                stdZDL=round(df3[f"throughput_time_{work_centre}"].std(),2)
                zdlmg= df3[f"throughput_time_{work_centre}"]* df3[f"process_time_{work_centre}"]
                meanzdlmg = round(zdlmg.sum() / df3[f"process_time_{work_centre}"].sum(),2)

                meanLeistung = (df2[f"Order_Output_{work_centre}"].max() - df2[f"Order_Output_{work_centre}"].min()) / (df2["time"].max() - df2["time"].min())
                meanBestand = round(df2[f"Order_WIP_{work_centre}"].mean(),2)
                stdBestand=  round(df2[f"Order_WIP_{work_centre}"].std(),2)
                meanReichweite = round(meanBestand / meanLeistung,2)



                minZeit =  int(round(df2["time"].min()))
                maxZeit =  int(round(df2["time"].max()))

                df_aggregat = pd.DataFrame()
                df_aggregat["time"] = np.arange(minZeit, maxZeit+1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Order_Input_{work_centre}"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Order_Input_{work_centre}"] - df_aggregat[f"Order_Input_{work_centre}"].shift(1)

                meanBela = round(df_aggregat['Difference'].mean(),2)
                stdBela= round(df_aggregat['Difference'].std(),2)

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Order_Output_{work_centre}"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Order_Output_{work_centre}"] - df_aggregat[f"Order_Output_{work_centre}"].shift(
                    1)

                meanLeistung= round(df_aggregat['Difference'].mean(),2)
                stdLeistung= round(df_aggregat['Difference'].std(),2)
                auslastung = round(meanLeistung/8,2)
                natStreu = round(stdzau * 2**0.5,2)

               #
                textstring = '$ZAZ_{m}$' ": %s | "'$ZAZ_{std}}$' ": %s\n" '$ZAU_{m}$' ": %s | "'$ZAU_{std}$' ": %s\n" '$ZDL_{m}}$' ": %s | "'$ZDL_{std}}$' ": %s" %(meanzaz,stdzaz,meanzau,stdzau,meanZDL,stdZDL)
                textstring2='$Bela_{m}$' ": %s | "'$Bela_{std}}$' ": %s\n" '$LA_{m}$' ": %s | "'$LA_{std}$' ": %s\n" '$BA_{m}}$' \
                           ": %s | "'$BA_{std}}$' ": %s\n" '$ZDL_{vir}$' ": %s | "'$ZDL_{mg}$' ": %s\n" '$nat.streuung}$' ": %s " % (
                meanBela, stdBela, meanLeistung, stdLeistung, meanBestand, stdBestand, meanReichweite, meanzdlmg,natStreu)

                props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=1)
                ax_sub.text(0.85, 0.5, textstring, transform=ax.transAxes, fontsize=12, horizontalalignment='center',
                      verticalalignment='center', bbox=props)

                ax_sub.text(0.45, 0.98, textstring2, transform=ax.transAxes, fontsize=12, horizontalalignment='center',
                        verticalalignment='top', bbox=props)

                plotName = "Arbeitssystem_" + str(work_centre) + "_" + str(auslastung) + "_" + 'DuDi_ung' + str(zoom) + '.svg'

                path=self.get_directory()

                subfolder="Plot/DUDI/"

                # save file
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName

                fig.savefig(file)
                print(f"Save {file}")
                plt.close(fig)

            plotsystem: bool = True

            if plotsystem == True:
                x = df["time"]
                y = df[f"Planned_Order_Release"]
                y3 = df[f"Order_Release"]
                y1 = df[f"Order_Input"]
                y2=  df[f"Order_Output"]
                z = df[f"Order_WIP_Total"]
                z2 = df[f"Order_WIP_Pool"]

                fig, ax = plt.subplots(1, 1)
                # Teilt die x-Achse der Y-Achsen und generiert Sie eine sekundäre Achse
                ax_sub = ax.twinx()
                # Zeichnen der Daten
                l1, = ax.plot(x, y1, drawstyle='steps-post', color='#d62728', label='Zugang');
                l2, = ax.plot(x, y2, drawstyle='steps-post', color='#ff7f0e',label='Abgang');
                l3, = ax.plot(x, y3, drawstyle='steps-post', color='#1f77b4',label='Freigabe');
                l4, = ax.plot(x, y, drawstyle='steps-post', linestyle='dashed', color='#1f77b4', label='Plan-Freigabe');
                l5, = ax_sub.plot(x, z, drawstyle='steps-post', color='#2ca02c',label='Bestand');
                l6, = ax_sub.plot(x, z2, drawstyle='steps-post', linestyle='dashed', color='#2ca02c',label='Bestand2');

                #box=ax.get_position()
                #ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                plt.legend(handles=[l1,l2,l3,l4,l5, l6], labels=['Zugangskurve', 'Abgangskurve', 'Freigabekurve', 'Plan-Freigabekurve', 'Bestandskurve_Gesamt', 'Bestandskurve_Pool'], loc='upper left')

                ax.set_ylabel("Aufträge [Stk.]")
                ax_sub.set_ylabel("Bestand [Stk.]")
                ax.set_xlabel("Zeit [BKT]")

                stepsize = 200 * (4-zoom)
                ymax = (round(y1.max()) // stepsize) * stepsize + stepsize
                ymin =(round(y2.min()) // stepsize) * stepsize + -stepsize

                ticks = (ymax - ymin) / stepsize

                ax.set_ylim(ymin, ymax)
                ax.set_yticks(range(ymin, ymax, stepsize))

                sichtfaktor = 3
                ymax2= z.max() * sichtfaktor
                ymin2= 0

                stepsize2 = round((ymax2 - ymin2)  // ticks) + 1

                ymax2 = int(round(stepsize2 * ticks))


                ax_sub.set_axisbelow(True)

                ax.set_axisbelow(True)
                ax_sub.set_ylim(ymin2, ymax2)
                ax_sub.set_yticks(range(0, ymax2, stepsize2))

                ax.tick_params(axis='y')
                ax_sub.tick_params(axis='y')
                ax.set_title('Freigabeverfahren: %s | Durchlaufdiagramm: Gesamtsystem' %(Release), pad=20, fontsize=16)

                # Mean ZAZ, STD ZAZ , MEAN ZAU, STD ZAU

                meanzaz = round(df3[f"interarrival_time"].mean(),2)
                stdzaz = round(df3[f"interarrival_time"].std(),2)
                meanzau=round(df3[f"total_process_time"].mean(),2)
                stdzau=round(df3[f"total_process_time"].std(),2)
                meanZDL =round(df3[f"throughput_time"].mean(),2)
                stdZDL=round(df3[f"throughput_time"].std(),2)
                zdlmg= df3[f"throughput_time"]* df3[f"total_process_time"]
                meanzdlmg = round(zdlmg.sum() / df3[f"total_process_time"].sum(),2)

                meanLeistung = (df2[f"Order_Output"].max() - df2[f"Order_Output"].min()) / (df2["time"].max() - df2["time"].min())
                meanBestand = round(df2[f"Order_WIP_Total"].mean(),2)
                stdBestand=  round(df2[f"Order_WIP_Total"].std(),2)
                meanBestand_Pool=round(df2[f"Order_WIP_Pool"].mean(), 2)
                stdBestand_Pool=round(df2[f"Order_WIP_Pool"].std(), 2)
                meanReichweite = round(meanBestand / meanLeistung,2)

                minZeit =  int(round(df2["time"].min()))
                maxZeit =  int(round(df2["time"].max()))

                df_aggregat = pd.DataFrame()
                df_aggregat["time"] = np.arange(minZeit, maxZeit+1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Order_Input"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Order_Input"] - df_aggregat[f"Order_Input"].shift(1)

                meanBela = round(df_aggregat['Difference'].mean(),2)
                stdBela= round(df_aggregat['Difference'].std(),2)

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Order_Output"]], on='time', how='left')
                df_aggregat['Difference']=df_aggregat[f"Order_Output"] - df_aggregat[f"Order_Output"].shift(
                    1)

                meanLeistung= round(df_aggregat['Difference'].mean(),2)
                stdLeistung= round(df_aggregat['Difference'].std(),2)

                auslastung = round(meanLeistung/8, 2)

                natStreu = round(stdzau * 2**0.5,2)

                #
                textstring = '$ZAZ_{m}$' ": %s | "'$ZAZ_{std}}$' ": %s\n" '$ZAU_{m}$' ": %s | "'$ZAU_{std}$' ": %s\n" '$ZDL_{m}}$' ": %s | "'$ZDL_{std}}$' ": %s" %(meanzaz,stdzaz,meanzau,stdzau,meanZDL,stdZDL)
                textstring2='$Bela_{m}$' ": %s | "'$Bela_{std}}$' ": %s\n" '$LA_{m}$' ": %s | "'$LA_{std}$' ": %s\n" '$BA_{m}}$' \
                           ": %s | "'$BA_{std}}$' ": %s\n" '$BA_{m,p}}$' \
                           ": %s | "'$BA_{std,p}}$' ": %s\n" '$ZDL_{vir}$' ": %s | "'$ZDL_{mg}$' ": %s\n" '$nat.streuung}$' ": %s " % (
                meanBela, stdBela, meanLeistung, stdLeistung, meanBestand, stdBestand, meanBestand_Pool, stdBestand_Pool, meanReichweite, meanzdlmg,natStreu)

                props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=1)
                ax_sub.text(0.85, 0.4, textstring, transform=ax.transAxes, fontsize=12, horizontalalignment='center',
                      verticalalignment='center', bbox=props)

                ax_sub.text(0.45, 0.98, textstring2, transform=ax.transAxes, fontsize=12, horizontalalignment='center',
                        verticalalignment='top', bbox=props)

                plotName = "Gesamtsystem_" + str(auslastung) + "_" + 'DuDi_ung' + str(zoom) +'.svg'

                path=self.get_directory()

                subfolder="Plot/DUDI/"

                # save file
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName

                fig.savefig(file)
                print(f"Save {file}")
                plt.close(fig)

    # Calculation and printing of Routing Matrix------------------------------------------------------------------------
    def MaterialflussMatrix(self, detailled, continous, aggregated):
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        df = detailled[0]

        matrix = np.zeros([len(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT) + 2, len(self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT) + 2])

        routingvec = df['Routing']
        # routingvec = routingvec.str.replace("[WC]","")

        for i in routingvec:
            test=np.matrix(i).getT()
            for j in range(0, test.size):
                if test[j] == "WC0":
                    num=1
                if test[j] == "WC1":
                    num=2
                if test[j] == "WC2":
                    num=3
                if test[j] == "WC3":
                    num=4
                if test[j] == "WC4":
                    num=5
                if test[j] == "WC5":
                    num=6

                # num = int(test[j])+1
                if j==0:
                    matrix[0, num] += 1
                if j >= 0 and j < test.size-1:
                    if test[j+1] == "WC0":
                        num2=1
                    if test[j+1] == "WC1":
                        num2=2
                    if test[j+1] == "WC2":
                        num2=3
                    if test[j+1] == "WC3":
                        num2=4
                    if test[j+1] == "WC4":
                        num2=5
                    if test[j+1] == "WC5":
                        num2=6

                    #num2=int(test[j+1]) + 1
                    matrix[num, num2]+=1
                if j==test.size-1:
                    matrix[num,7] += 1

            #print (i)
        df = pd.DataFrame(matrix)
        df_O = df

        df2 = df.transpose()
        df2 = df2/df2.sum()
        df = df2.transpose()

        df.columns = ["Freigabe", "WC0", "WC1", "WC2", "WC3", "WC4", "WC5", "Kunde"]
        df.index = ["Freigabe", "WC0", "WC1", "WC2", "WC3", "WC4", "WC5", "Kunde"]
        df.drop(df.tail(1).index,inplace=True)
        df.drop(columns=df.columns[0], axis=1,  inplace=True)


        df_O.columns = ["Freigabe", "WC0", "WC1", "WC2", "WC3", "WC4", "WC5", "Kunde"]
        df_O.index = ["Freigabe", "WC0", "WC1", "WC2", "WC3", "WC4", "WC5", "Kunde"]
        df_O.drop(df_O.tail(1).index,inplace=True)
        df_O.drop(columns=df_O.columns[0], axis=1,  inplace=True)


        fig, ax = plt.subplots()

        # fig.tight_layout()
        ax = sns.heatmap(data=df, cmap="crest", annot=True, linewidth=.5)
        #ax[1] = sns.heatmap(ax=ax[1], data=df_O, cmap="crest", annot=True, linewidth=.5)
        ax.set_title("Materialflussmatrix", pad=20, fontsize=16)
        ax.set(xlabel="nach", ylabel="von")
        plt.yticks(rotation =0)
        ax.xaxis.tick_top()

        plotName="Materialflussmatrix" + '.svg'

        path=self.get_directory()

        subfolder="Plot/"

        # save file
        file=path + subfolder + self.sim.model_panel.experiment_name + plotName

        fig.savefig(file)
        print(f"Save {file}")
        plt.close(fig)

    # Calculation and printing of Load and WIP Analysation--------------------------------------------------------------
    def BELA(self, detailled, continous):
        plt.clf()
        plt.cla()
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.axes_style("darkgrid")
        sns.color_palette("tab10")

        for i in np.arange(0, 4, 1):
            df2=continous
            df3=detailled
            mean=df2['time'].mean()
            df=pd.DataFrame()

            auslastung= self.sim.model_panel.AIMED_UTILIZATION
            zoom=i

            plotsystem: bool=True

            if plotsystem == True:

                minZeit=int(round(df2["time"].min()))
                maxZeit=int(round(df2["time"].max()))

                df_aggregat=pd.DataFrame()
                df_aggregat["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat=pd.merge(df_aggregat, df2[['time', f"Load_Input"]], on='time', how='left')
                df_aggregat['Load_Input_Daily']=df_aggregat[f"Load_Input"] - df_aggregat[f"Load_Input"].shift(1)

                meanBela_Load=round(df_aggregat['Load_Input_Daily'].mean(), 2)
                stdBela_Load=round(df_aggregat['Load_Input_Daily'].std(), 2)
                minBela_Load=round(df_aggregat['Load_Input_Daily'].min(), 2)
                maxBela_Load=round(df_aggregat['Load_Input_Daily'].max(), 2)

                df_aggregat2=pd.DataFrame()
                df_aggregat2["time"]=np.arange(minZeit, maxZeit + 1, 1)
                df_aggregat2=pd.merge(df_aggregat2, df2[['time', f"Order_Input"]], on='time', how='left')
                df_aggregat2['Order_Input_Daily']=df_aggregat2[f"Order_Input"] - df_aggregat2[f"Order_Input"].shift(1)

                meanBela_Order=round(df_aggregat2['Order_Input_Daily'].mean(), 2)
                stdBela_Order=round(df_aggregat2['Order_Input_Daily'].std(), 2)
                minBela_Order=round(df_aggregat2['Order_Input_Daily'].min(), 2)
                maxBela_Order=round(df_aggregat2['Order_Input_Daily'].max(), 2)

                df=pd.merge(df_aggregat, df_aggregat2, left_on='time', right_on='time', how='left')
                df2=df

                if i == 0:
                    df=df.loc[(df2['time'] >= mean - 500) & (df2['time'] <= mean + 500)]
                elif i == 1:
                    df=df2.loc[(df2['time'] >= mean - 250) & (df2['time'] <= mean + 250)]
                elif i == 2:
                    df=df2.loc[(df2['time'] >= mean - 100) & (df2['time'] <= mean + 100)]
                else:
                    df=df2.loc[(df2['time'] >= mean - 50) & (df2['time'] <= mean + 50)]

                x=df["time"]
                y=df[f"Load_Input_Daily"]
                y1=df[f"Order_Input_Daily"]

                gs_kw=dict(width_ratios=[2.3, 1])

                fig, axs=plt.subplots(ncols=2, nrows=2, gridspec_kw=gs_kw, figsize=(11.7, 8.27))

                fig.suptitle('Plan-Auslastung: %s | Belastungsanalyse: Gesamtsystem' % (auslastung), y=0.99,
                             fontsize=16)

                axs[0, 0].plot(x, y, color='#1f77b4', label='Load')
                axs[1, 0].plot(x, y1, color='#ff7f0e', label='Order')

                axs[0, 0].legend(labels=['Belastung in Stunden'], loc='upper left')
                axs[1, 0].legend(labels=['Belastung in Aufträgen'], loc='upper left')

                axs[0, 0].set_ylabel("Arbeitsinhalt [Std/BKT]")
                axs[1, 0].set_ylabel("Anzahl Aufträge [Anz/BKT]")

                axs[0, 0].set_xlabel("Zeit [BKT]")
                axs[1, 0].set_xlabel("Zeit [BKT]")

                ymax=round(y.max()) * 1.3
                y1max=round(y1.max()) * 1.3

                axs[0, 0].set_ylim(0, ymax)
                axs[1, 0].set_ylim(0, y1max)

                textstring='$Bela_{m}$' ": %s | "'$Bela_{std}}$' ": %s\n" '$Bela_{min}$' ": %s | "'$Bela_{max}$' ": " \
                           "%s" % (
                    meanBela_Load, stdBela_Load, minBela_Load, maxBela_Load)

                textstring2='$Bela_{m}$' ": %s | "'$Bela_{std}}$' ": %s\n" '$Bela_{min}$' ": %s | "'$Bela_{max}$' ": " \
                            "%s" % (
                    meanBela_Order, stdBela_Order, minBela_Order, maxBela_Order)

                props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=1)
                axs[0, 0].text(0.78, 0.90, textstring, transform=axs[0, 0].transAxes, fontsize=10,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)

                axs[1, 0].text(0.78, 0.95, textstring2, transform=axs[1, 0].transAxes, fontsize=10,
                               horizontalalignment='center',
                               verticalalignment='top', bbox=props)

                axs[1, 1].hist(df3["total_process_time"], bins=30, density=True, histtype='bar', stacked=False,
                               rwidth=0.8)

                axs[0, 1].hist(df3["interarrival_time"], bins=30, density=True, histtype='bar', stacked=False,
                               rwidth=0.8)

                axs[0, 1].legend(labels=['Zwischenankunftszeit'], loc='upper right')
                axs[1, 1].legend(labels=['Arbeitsinhalt'], loc='upper right')

                axs[0, 1].set_xlabel("Zwischenankunftszeit [BKT]")
                axs[1, 1].set_xlabel("Arbeitsinhalt [Std]")

                meanZAZ=round(df3["interarrival_time"].mean(), 2)
                stdZAZ=round(df3["interarrival_time"].std(), 2)
                minZAZ=round(df3["interarrival_time"].min(), 2)
                maxZAZ=round(df3["interarrival_time"].max(), 2)
                meanZAU=round(df3["total_process_time"].mean(), 2)
                stdZAU=round(df3["total_process_time"].std(), 2)
                minZAU=round(df3["total_process_time"].min(), 2)
                maxZAU=round(df3["total_process_time"].max(), 2)

                textstring3='$ZAZ_{m}$' ": %s | "'$ZAZ_{std}}$' ": %s\n" '$ZAZ_{min}$' ": %s | "'$ZAZ_{max}$' ": %s" % (
                    meanZAZ, stdZAZ, minZAZ, maxZAZ)

                textstring4='$ZAU_{m}$' ": %s | "'$ZAU_{std}}$' ": %s\n" '$ZAU_{min}$' ": %s | "'$ZAU_{max}$' ": %s" % (
                    meanZAU, stdZAU, minZAU, maxZAU)

                props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=1)
                axs[0, 1].text(1.42, 0.82, textstring3, transform=axs[0, 0].transAxes, fontsize=9,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)

                axs[1, 1].text(1.42, 0.87, textstring4, transform=axs[1, 0].transAxes, fontsize=9,
                               horizontalalignment='center',
                               verticalalignment='top', bbox=props)

                axs[0, 1].axvline(x=meanZAZ, ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[1, 1].axvline(x=meanZAU, ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)

                axs[0, 0].axhline(y=meanBela_Load, color="red", linestyle="--", alpha=0.35)
                axs[1, 0].axhline(y=meanBela_Order, color="red", linestyle="--", alpha=0.35)

                plt.subplots_adjust(left=0.07,
                                    bottom=0.07,
                                    right=0.93,
                                    top=0.93,
                                    wspace=0.2,
                                    hspace=0.2)

                plotName="Gesamtsystem_" + str(auslastung) + "_" + 'BeLa' + str(zoom) + '.svg'


                path=self.get_directory()

                subfolder="Plot/BELA/"

                # save file
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName

                fig.savefig(file)
                print(f"Save {file}")
                plt.close(fig)

    # Calculation and printing of Deviation Analyses--------------------------------------------------------------------
    def TAX_ASys(self, detailled):

        plt.clf()
        plt.cla()
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.axes_style("darkgrid")
        sns.color_palette("crest", as_cmap=True)

        freigabe= self.sim.policy_panel.release_control_method

        df= detailled

        # Schleife für Produktionskennlinie
        for work_centre in self.sim.model_panel.MANUFACTURING_FLOOR_LAYOUT:

            # TAZ Arbeitssystem
            df[f"TAZ_System"]=df[f"deviation_input_{work_centre}"]
            y1=df["TAZ_System"]
            df[f"input_{work_centre}_rank_deviation"]=df[f"actual_input_{work_centre}_Rank"] - df[
                f"planned_input_{work_centre}_Rank"]
            df[f"input_{work_centre}_rank_deviation_weighted"]=df[f"actual_input_{work_centre}_Rank_weighted"] - df[
                f"planned_input_{work_centre}_Rank_weighted"]
            df["TAZ_System_rf"]=df[f"input_{work_centre}_rank_deviation"] * df.loc[:,
                                                                            f"operation_interarrival_time_" \
                                                                            f"{work_centre}"].mean()
            df["TAZ_System_rf_weighted"]=df[f"input_{work_centre}_rank_deviation"] * df.loc[:,
                                                                                     f"operation_interarrival_time_" \
                                                                                     f"{work_centre}"].mean() / df.loc[
                                                                                                                                            :,
                                                                                                                                            f"process_time_{work_centre}"].mean()
            df["TAZ_System_rs"]=df[f"TAZ_System"] - df["TAZ_System_rf"]
            df["TAZ_System_rs_weighted"]=df[f"TAZ_System"] - df["TAZ_System_rf_weighted"]

            mean_taz_system_mg=round(
                (df["TAZ_System"] * df.loc[:, f"process_time_{work_centre}"]).sum() / df.loc[:,
                                                                                      f"process_time_"
                                                                                      f"{work_centre}"].sum(),
                2)
            mean_taz_system_mg_rf=round(
                (df.loc[:, "TAZ_System_rf"] * df.loc[:, f"process_time_{work_centre}"]).sum() / df.loc[:,
                                                                                                f"process_time_"
                                                                                                f"{work_centre}"].sum(),
                2)
            mean_taz_system_mg_rs=round(
                (df.loc[:, "TAZ_System_rs"] * df.loc[:, f"process_time_{work_centre}"]).sum() / df.loc[:,
                                                                                                f"process_time_"
                                                                                                f"{work_centre}"].sum(),
                2)

            # TAA Arbeitssystem
            df[f"TAA_System"]=df[f"deviation_output_{work_centre}"]
            y2=df[f"TAA_System"]
            df[f"output_{work_centre}_rank_deviation"]=df[f"actual_output_{work_centre}_Rank"] - df[
                f"planned_output_{work_centre}_Rank"]
            df[f"output_{work_centre}_rank_deviation_weighted"]=df[f"actual_output_{work_centre}_Rank_weighted"] - df[
                f"planned_output_{work_centre}_Rank_weighted"]
            df["TAA_System_rf"]=df[f"output_{work_centre}_rank_deviation"] * df.loc[:,
                                                                             f"operation_interarrival_time_" \
                                                                             f"{work_centre}"].mean()
            df["TAA_System_rf_weighted"]=df[f"output_{work_centre}_rank_deviation"] * df.loc[:,
                                                                                      f"operation_interarrival_time_" \
                                                                                      f"{work_centre}"].mean() / df.loc[
                                                                                                                                             :,
                                                                                                                                             f"process_time_{work_centre}"].mean()
            df["TAA_System_rs"]=df[f"TAA_System"] - df["TAA_System_rf"]
            df["TAA_System_rs_weighted"]=df[f"TAA_System"] - df["TAA_System_rf_weighted"]

            mean_taa_system_mg=round(
                (df["TAA_System"] * df.loc[:, f"process_time_{work_centre}"]).sum() / df.loc[:,
                                                                                      f"process_time_"
                                                                                      f"{work_centre}"].sum(),
                2)
            mean_taa_system_mg_rf=round(
                (df.loc[:, "TAA_System_rf"] * df.loc[:, f"process_time_{work_centre}"]).sum() / df.loc[:,
                                                                                                f"process_time_"
                                                                                                f"{work_centre}"].sum(),
                2)
            mean_taa_system_mg_rs=round(
                (df.loc[:, "TAA_System_rs"] * df.loc[:, f"process_time_{work_centre}"]).sum() / df.loc[:,
                                                                                                f"process_time_"
                                                                                                f"{work_centre}"].sum(),
                2)

            # TAR Arbeitssystem
            df[f"TAR_System"]=df[f"TAA_System"] - df[f"TAZ_System"]
            y3=df[f"TAR_System"]
            df["TAR_System_rf"]=df["TAA_System_rf"] - df["TAZ_System_rf"]
            df["TAR_System_rf_weighted"]=df["TAA_System_rf_weighted"] - df["TAZ_System_rf_weighted"]
            df["TAR_System_rs"]=df["TAA_System_rs"] - df["TAZ_System_rf"]
            df["TAR_System_rs_weighted"]=df["TAA_System_rs_weighted"] - df["TAZ_System_rs_weighted"]

            mean_tar_system_mg=round((df["TAR_System"] * df.loc[:, f"process_time_{work_centre}"]).sum() / df.loc[:,
                                                                                                           f"process_time_{work_centre}"].sum(),
                                     2)
            mean_tar_system_mg_rf=round(
                (df.loc[:, "TAR_System_rf"] * df.loc[:, f"process_time_{work_centre}"]).sum() / df.loc[:,
                                                                                                f"process_time_"
                                                                                                f"{work_centre}"].sum(),
                2)
            mean_tar_system_mg_rs=round(
                (df.loc[:, "TAR_System_rs"] * df.loc[:, f"process_time_{work_centre}"]).sum() / df.loc[:,
                                                                                                f"process_time_"
                                                                                                f"{work_centre}"].sum(),
                2)

            df["Durchlaufzeit_Gesamt"]=df[f"throughput_time_{work_centre}"]
            y4=df["Durchlaufzeit_Gesamt"]
            mean_zdl_mg=round((df["Durchlaufzeit_Gesamt"] * df.loc[:, f"process_time_{work_centre}"]).sum() / df.loc[:,
                                                                                                              f"process_time_{work_centre}"].sum(),
                              2)

            plotgesamt: bool=True
            plotzerlegung: bool=True
            plotzerlegung_gewichtet: bool=True
            korrelationsplot: bool=True

            if plotgesamt == True:
                plt.clf()
                plt.cla()
                gs_kw=dict(width_ratios=[1, 1])

                fig, axs=plt.subplots(ncols=2, nrows=2, gridspec_kw=gs_kw, figsize=(11.7, 8.27))

                fig.suptitle('Freigabe: %s | TAX-Analyse: Arbeitssystem %s' % (freigabe, work_centre), y=0.98,
                             fontsize=16)

                plt.subplots_adjust(left=0.07,
                                    bottom=0.07,
                                    right=0.93,
                                    top=0.90,
                                    wspace=0.3,
                                    hspace=0.3)

                axs[0, 0].hist(y1, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                               rwidth=0.8)

                axs[0, 1].hist(y2, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                               rwidth=0.8)

                axs[1, 0].hist(y3, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                               rwidth=0.8)

                axs[1, 1].hist(y4, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                               rwidth=0.8)

                axs[0, 0].axvline(x=y1.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[0, 1].axvline(x=y2.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[1, 0].axvline(x=y3.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[1, 1].axvline(x=y4.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)

                axs[0, 0].set_xlabel("Terminabweichung [BKT]")
                axs[0, 1].set_xlabel("Terminabweichung [BKT]")
                axs[1, 0].set_xlabel("Terminabweichung [BKT]")
                axs[1, 1].set_xlabel("Durchlaufzeit [BKT]")

                axs[0, 0].title.set_text("Terminabweichung im Zugang")
                axs[0, 1].title.set_text("Terminabweichung im Abgang")
                axs[1, 0].title.set_text("Relative Terminabweichung")
                axs[1, 1].title.set_text("Durchlaufzeit")

                textstring1='$TAZ_{m}$' ": %s | "'$TAZ_{std}}$' ": %s\n" '$TAZ_{min}$' ": %s | "'$TAZ_{max}$' ": " \
                            "%s\n" '$TAZ_{mg}$' ": %s | "'$TAZ_{med}$' ": " "%s" % (
                                round(y1.mean(), 2), round(y1.std(), 2), round(y1.min(), 2), round(y1.max(), 2),
                                mean_taz_system_mg,
                                round(y1.median(), 2))

                textstring2='$TAA_{m}$' ": %s | "'$TAA_{std}}$' ": %s\n" '$TAA_{min}$' ": %s | "'$TAA_{max}$' ": " \
                            "%s\n" '$TAA_{mg}$' ": %s | "'$TAA_{med}$' ": " "%s" % (
                                round(y2.mean(), 2), round(y2.std(), 2), round(y2.min(), 2), round(y2.max(), 2),
                                mean_taa_system_mg,
                                round(y2.median(), 2))

                textstring3='$TAR_{m}$' ": %s | "'$TAR_{std}}$' ": %s\n" '$TAR_{min}$' ": %s | "'$TAR_{max}$' ": " \
                            "%s\n" '$TAR_{mg}$' ": %s | "'$TAR_{med}$' ": " "%s" % (
                                round(y3.mean(), 2), round(y3.std(), 2), round(y3.min(), 2), round(y3.max(), 2),
                                mean_tar_system_mg,
                                round(y3.median(), 2))

                textstring4='$ZDL_{m}$' ": %s | "'$ZDL_{std}}$' ": %s\n" '$ZDL_{min}$' ": %s | "'$ZDL_{max}$' ": " \
                            "%s\n" '$ZDL_{mg}$' ": %s | "'$ZDL_{med}$' ": " "%s" % (
                                round(y4.mean(), 2), round(y4.std(), 2), round(y4.min(), 2), round(y4.max(), 2),
                                mean_zdl_mg,
                                round(y4.median(), 2))

                props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=0.7)
                axs[0, 0].text(0.66, 0.87, textstring1, transform=axs[0, 0].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[0, 1].text(0.66, 0.87, textstring2, transform=axs[0, 1].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[1, 0].text(0.66, 0.87, textstring3, transform=axs[1, 0].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[1, 1].text(0.66, 0.87, textstring4, transform=axs[1, 1].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)

                plotName="Arbeitssystem_" + str(work_centre) + "_" + 'TAX' + '.svg'
                path=self.get_directory()
                subfolder="Plot/TAX/"
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName
                fig.savefig(file)
                print(f"Save {file}")
                plt.close(fig)

            if plotzerlegung == True:
                plt.clf()
                plt.cla()
                y1=df["TAZ_System_rf"]
                y2=df["TAR_System_rf"]
                y3=df["TAA_System_rf"]
                y4=df["TAZ_System_rs"]
                y5=df["TAR_System_rs"]
                y6=df["TAA_System_rs"]

                gs_kw=dict(width_ratios=[1, 1, 1])

                fig, axs=plt.subplots(ncols=3, nrows=2, gridspec_kw=gs_kw, figsize=(11.7, 8.27))

                fig.suptitle(
                    'Freigabe: %s | TAX-Analyse: Arbeitssystem %s | (Zerl. via Rangabw.)' % (freigabe, work_centre),
                    y=0.98,
                    fontsize=16)

                plt.subplots_adjust(left=0.07,
                                    bottom=0.07,
                                    right=0.93,
                                    top=0.90,
                                    wspace=0.3,
                                    hspace=0.3)

                axs[0, 0].hist(y1, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                               rwidth=0.8)

                axs[0, 1].hist(y2, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                               rwidth=0.8)

                axs[0, 2].hist(y3, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                               rwidth=0.8)

                axs[1, 0].hist(y4, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                               rwidth=0.8)

                axs[1, 1].hist(y5, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                               rwidth=0.8)

                axs[1, 2].hist(y6, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                               rwidth=0.8)

                axs[0, 0].axvline(x=y1.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[0, 1].axvline(x=y2.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[0, 2].axvline(x=y3.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[1, 0].axvline(x=y4.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[1, 1].axvline(x=y5.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[1, 2].axvline(x=y6.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)

                axs[0, 0].set_xlabel("Terminabweichung [BKT]")
                axs[0, 1].set_xlabel("Terminabweichung [BKT]")
                axs[0, 2].set_xlabel("Terminabweichung [BKT]")
                axs[1, 0].set_xlabel("Terminabweichung [BKT]")
                axs[1, 1].set_xlabel("Terminabweichung [BKT]")
                axs[1, 2].set_xlabel("Terminabweichung [BKT]")

                axs[0, 0].title.set_text("$TAZ_{rf}$ im Zugang")
                axs[0, 1].title.set_text("$TAR_{rf}$ (relativ)")
                axs[0, 2].title.set_text("$TAA_{rf}$ im Abgang")
                axs[1, 0].title.set_text("$TAZ_{rs}$ im Zugang")
                axs[1, 1].title.set_text("$TAR_{rs}$ (relativ)")
                axs[1, 2].title.set_text("$TAA_{rs}$ im Abgang")

                textstring1='$TAZ_{m,rf}$' ": %s | "'$TAZ_{std,rf}}$' ": %s\n" '$TAZ_{min,rf}$' ": %s | "'$TAZ_{max,' \
                            'rf}$' ": " \
                            "%s\n" '$TAZ_{mg,rf}$' ": %s | "'$TAZ_{med,rf}$' ": " "%s" % (
                                round(y1.mean(), 2), round(y1.std(), 2), round(y1.min(), 2), round(y1.max(), 2),
                                mean_taz_system_mg_rf,
                                round(y1.median(), 2))

                textstring2='$TAR_{m,rf}$' ": %s | "'$TAR_{std,rf}}$' ": %s\n" '$TAR_{min,rf}$' ": %s | "'$TAR_{max,' \
                            'rf}$' ": " \
                            "%s\n" '$TAR_{mg,rf}$' ": %s | "'$TAR_{med,rf}$' ": " "%s" % (
                                round(y2.mean(), 2), round(y2.std(), 2), round(y2.min(), 2), round(y2.max(), 2),
                                mean_taz_system_mg_rf,
                                round(y2.median(), 2))

                textstring3='$TAA_{m,rf}$' ": %s | "'$TAA_{std,rf}}$' ": %s\n" '$TAA_{min,rf}$' ": %s | "'$TAA_{max,' \
                            'rf}$' ": " \
                            "%s\n" '$TAA_{mg,rf}$' ": %s | "'$TAA_{med,rf}$' ": " "%s" % (
                                round(y3.mean(), 2), round(y3.std(), 2), round(y3.min(), 2), round(y3.max(), 2),
                                mean_taa_system_mg_rf,
                                round(y3.median(), 2))

                textstring4='$TAZ_{m,rs}$' ": %s | "'$TAZ_{std,rs}}$' ": %s\n" '$TAZ_{min,rs}$' ": %s | "'$TAZ_{max,' \
                            'rs}$' ": " \
                            "%s\n" '$TAZ_{mg,rs}$' ": %s | "'$TAZ_{med,rs}$' ": " "%s" % (
                                round(y4.mean(), 2), round(y4.std(), 2), round(y4.min(), 2), round(y4.max(), 2),
                                mean_taz_system_mg_rs,
                                round(y4.median(), 2))

                textstring5='$TAR_{m,rs}$' ": %s | "'$TAR_{std,rs}}$' ": %s\n" '$TAR_{min,rs}$' ": %s | "'$TAR_{max,' \
                            'rs}$' ": " \
                            "%s\n" '$TAR_{mg,rs}$' ": %s | "'$TAR_{med,rs}$' ": " "%s" % (
                                round(y5.mean(), 2), round(y5.std(), 2), round(y5.min(), 2), round(y5.max(), 2),
                                mean_tar_system_mg_rs,
                                round(y5.median(), 2))

                textstring6='$TAA_{m,rs}$' ": %s | "'$TAA_{std,rs}}$' ": %s\n" '$TAA_{min,rs}$' ": %s | "'$TAA_{max,' \
                            'rs}$' ": " \
                            "%s\n" '$TAA_{mg,rs}$' ": %s | "'$TAA_{med,rs}$' ": " "%s" % (
                                round(y6.mean(), 2), round(y6.std(), 2), round(y6.min(), 2), round(y6.max(), 2),
                                mean_taa_system_mg_rs,
                                round(y6.median(), 2))

                props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=0.7)
                axs[0, 0].text(0.66, 0.87, textstring1, transform=axs[0, 0].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[0, 1].text(0.66, 0.87, textstring2, transform=axs[0, 1].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[0, 2].text(0.66, 0.87, textstring3, transform=axs[0, 2].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[1, 0].text(0.66, 0.87, textstring4, transform=axs[1, 0].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[1, 1].text(0.66, 0.87, textstring5, transform=axs[1, 1].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[1, 2].text(0.66, 0.87, textstring6, transform=axs[1, 2].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)

                plotName="Arbeitssystem_" + str(work_centre) + "_" + 'TAX_R' + '.svg'
                path=self.get_directory()
                subfolder="Plot/TAX/"
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName
                fig.savefig(file)
                print(f"Save {file}")
                plt.close(fig)

            if plotzerlegung_gewichtet == True:
                plt.clf()
                plt.cla()
                y1=df["TAZ_System_rf_weighted"]
                y2=df["TAR_System_rf_weighted"]
                y3=df["TAA_System_rf_weighted"]
                y4=df["TAZ_System_rs_weighted"]
                y5=df["TAR_System_rs_weighted"]
                y6=df["TAA_System_rs_weighted"]

                gs_kw=dict(width_ratios=[1, 1, 1])

                fig, axs=plt.subplots(ncols=3, nrows=2, gridspec_kw=gs_kw, figsize=(11.7, 8.27))

                fig.suptitle('Freigabe: %s | TAX-Analyse: Arbeitssystem %s | (Zerl. via gewichteter Rangabw.)' % (
                freigabe, work_centre), y=0.98, fontsize=16)

                plt.subplots_adjust(left=0.07,
                                    bottom=0.07,
                                    right=0.93,
                                    top=0.90,
                                    wspace=0.3,
                                    hspace=0.3)

                axs[0, 0].hist(y1, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                               rwidth=0.8)

                axs[0, 1].hist(y2, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                               rwidth=0.8)

                axs[0, 2].hist(y3, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                               rwidth=0.8)

                axs[1, 0].hist(y4, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                               rwidth=0.8)

                axs[1, 1].hist(y5, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                               rwidth=0.8)

                axs[1, 2].hist(y6, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                               rwidth=0.8)

                axs[0, 0].axvline(x=y1.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[0, 1].axvline(x=y2.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[0, 2].axvline(x=y3.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[1, 0].axvline(x=y4.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[1, 1].axvline(x=y5.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
                axs[1, 2].axvline(x=y6.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)

                axs[0, 0].set_xlabel("Terminabweichung [BKT]")
                axs[0, 1].set_xlabel("Terminabweichung [BKT]")
                axs[0, 2].set_xlabel("Terminabweichung [BKT]")
                axs[1, 0].set_xlabel("Terminabweichung [BKT]")
                axs[1, 1].set_xlabel("Terminabweichung [BKT]")
                axs[1, 2].set_xlabel("Terminabweichung [BKT]")

                axs[0, 0].title.set_text("$TAZ_{rf,g}$ im Zugang")
                axs[0, 1].title.set_text("$TAR_{rf,g}$ (relativ)")
                axs[0, 2].title.set_text("$TAA_{rf,g}$ im Abgang")
                axs[1, 0].title.set_text("$TAZ_{rs,g}$ im Zugang")
                axs[1, 1].title.set_text("$TAR_{rs,g}$ (relativ)")
                axs[1, 2].title.set_text("$TAA_{rs,g}$ im Abgang")

                textstring1='$TAZ_{m,rfg}$' ": %s | "'$TAZ_{std,rfg}}$' ": %s\n" '$TAZ_{min,rfg}$' ": %s | "'$TAZ_{' \
                            'max,rfg}$' ": " \
                            "%s\n"'$TAZ_{med,rfg}$' ": " "%s" % (
                                round(y1.mean(), 2), round(y1.std(), 2), round(y1.min(), 2), round(y1.max(), 2),
                                round(y1.median(), 2))

                textstring2='$TAR_{m,rfg}$' ": %s | "'$TAR_{std,rfg}}$' ": %s\n" '$TAR_{min,rfg}$' ": %s | "'$TAR_{' \
                            'max,rfg}$' ": " \
                            "%s\n"'$TAR_{med,rfg}$' ": " "%s" % (
                                round(y2.mean(), 2), round(y2.std(), 2), round(y2.min(), 2), round(y2.max(), 2),
                                round(y2.median(), 2))

                textstring3='$TAA_{m,rfg}$' ": %s | "'$TAA_{std,rfg}}$' ": %s\n" '$TAA_{min,rfg}$' ": %s | "'$TAA_{' \
                            'max,rfg}$' ": " \
                            "%s\n"'$TAA_{med,rfg}$' ": " "%s" % (
                                round(y3.mean(), 2), round(y3.std(), 2), round(y3.min(), 2), round(y3.max(), 2),
                                round(y3.median(), 2))

                textstring4='$TAZ_{m,rsg}$' ": %s | "'$TAZ_{std,rsg}}$' ": %s\n" '$TAZ_{min,rsg}$' ": %s | "'$TAZ_{' \
                            'max,rsg}$' ": " \
                            "%s\n"'$TAZ_{med,rsg}$' ": " "%s" % (
                                round(y4.mean(), 2), round(y4.std(), 2), round(y4.min(), 2), round(y4.max(), 2),
                                round(y4.median(), 2))

                textstring5='$TAR_{m,rsg}$' ": %s | "'$TAR_{std,rsg}}$' ": %s\n" '$TAR_{min,rsg}$' ": %s | "'$TAR_{' \
                            'max,rsg}$' ": " \
                            "%s\n"'$TAR_{med,rsg}$' ": " "%s" % (
                                round(y5.mean(), 2), round(y5.std(), 2), round(y5.min(), 2), round(y5.max(), 2),
                                round(y5.median(), 2))

                textstring6='$TAA_{m,rsg}$' ": %s | "'$TAA_{std,rsg}}$' ": %s\n" '$TAA_{min,rsg}$' ": %s | "'$TAA_{' \
                            'max,rsg}$' ": " \
                            "%s\n"'$TAA_{med,rsg}$' ": " "%s" % (
                                round(y6.mean(), 2), round(y6.std(), 2), round(y6.min(), 2), round(y6.max(), 2),
                                round(y6.median(), 2))

                props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=0.7)
                axs[0, 0].text(0.66, 0.87, textstring1, transform=axs[0, 0].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[0, 1].text(0.66, 0.87, textstring2, transform=axs[0, 1].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[0, 2].text(0.66, 0.87, textstring3, transform=axs[0, 2].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[1, 0].text(0.66, 0.87, textstring4, transform=axs[1, 0].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[1, 1].text(0.66, 0.87, textstring5, transform=axs[1, 1].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)
                axs[1, 2].text(0.66, 0.87, textstring6, transform=axs[1, 2].transAxes, fontsize=7,
                               horizontalalignment='center',
                               verticalalignment='center', bbox=props)

                plotName="Arbeitssystem_" + str(work_centre) + "_" + 'TAX_RG' + '.svg'
                path=self.get_directory()
                subfolder="Plot/TAX/"
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName
                fig.savefig(file)
                print(f"Save {file}")
                plt.close(fig)

            if korrelationsplot == True:
                plt.clf()
                plt.cla()
                # Korrelationsplot 1
                cols_to_plot=['TAZ_System', 'TAA_System', "TAZ_System_rf", "TAZ_System_rs", "TAA_System_rf",
                              "TAA_System_rs", f"process_time_{work_centre}", f"throughput_time_{work_centre}",
                              'length_of_routing']
                g=sns.pairplot(df[cols_to_plot], palette="crest_r", diag_kws={'common_norm': True}, hue="length_of_routing")

                plotName="Arbeitssystem_" + str(work_centre) + "_" + 'Korrelation_ZA_Routing' + '.png'
                path=self.get_directory()
                subfolder="Plot/TAX/"
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName
                g.figure.savefig(file, dpi=400)
                print(f"Save {file}")
                g.fig.clf()
                plt.close(g.fig)

                cols_to_plot=['TAZ_System', 'TAA_System', "TAZ_System_rf", "TAZ_System_rs", "TAA_System_rf",
                              "TAA_System_rs", f"process_time_{work_centre}", f"throughput_time_{work_centre}",
                              f"operation_number_{work_centre}"]
                g=sns.pairplot(df[cols_to_plot], palette="crest_r", diag_kws={'common_norm': True}, hue=f"operation_number_{work_centre}")

                plotName="Arbeitssystem_" + str(work_centre) + "_" + 'Korrelation_ZA_OPNUM' + '.png'
                path=self.get_directory()
                subfolder="Plot/TAX/"
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName
                g.figure.savefig(file, dpi=400)
                print(f"Save {file}")
                g.fig.clf()
                plt.close(g.fig)

                cols_to_plot=['TAZ_System', 'TAR_System', "TAZ_System_rf", "TAZ_System_rs", "TAR_System_rf",
                              "TAR_System_rs",
                              f"process_time_{work_centre}", f"throughput_time_{work_centre}", 'length_of_routing']
                g=sns.pairplot(df[cols_to_plot], palette="crest_r", diag_kws={'common_norm': True}, hue="length_of_routing")

                plotName="Arbeitssystem_" + str(work_centre) + "_" + 'Korrelation_ZR_Routing' + '.png'
                path=self.get_directory()
                subfolder="Plot/TAX/"
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName
                g.figure.savefig(file, dpi=400)
                print(f"Save {file}")
                g.fig.clf()
                plt.close(g.fig)

                cols_to_plot=['TAZ_System', 'TAR_System', "TAZ_System_rf", "TAZ_System_rs", "TAR_System_rf",
                              "TAR_System_rs",
                              f"process_time_{work_centre}", f"throughput_time_{work_centre}",
                              f"operation_number_{work_centre}"]
                g=sns.pairplot(df[cols_to_plot], palette="crest_r", diag_kws={'common_norm': True}, hue=f"operation_number_{work_centre}")

                plotName="Arbeitssystem_" + str(work_centre) + "_" + 'Korrelation_ZR_OPNUM' + '.png'
                path=self.get_directory()
                subfolder="Plot/TAX/"
                file=path + subfolder + self.sim.model_panel.experiment_name + plotName
                g.figure.savefig(file, dpi=400)
                print(f"Save {file}")
                g.fig.clf()
                plt.close(g.fig)

    # Calculation and printing of Deviation Analyses for entire system--------------------------------------------------
    def TAX_Gesamtsystem(self, detailled):

        plt.clf()
        plt.cla()

        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.axes_style("darkgrid")
        sns.color_palette("crest", as_cmap=True)

        df= detailled

        freigabe = self.sim.policy_panel.release_control_method

        # TAZ System
        df["TAZ_System"]=df["deviation_input"]
        y1=df["TAZ_System"]
        df["entry_rank_deviation"]=df["actual_entry_Rank"] - df["planned_entry_Rank"]
        df["entry_rank_deviation_weighted"]=df["actual_entry_Rank_weighted"] - df["planned_entry_Rank_weighted"]
        df["TAZ_System_rf"]=df["entry_rank_deviation"] * df.loc[:, "interarrival_time"].mean()
        df["TAZ_System_rf_weighted"]=df["entry_rank_deviation_weighted"] * df.loc[:,
                                                                           "interarrival_time"].mean() / df.loc[:,
                                                                                                         "total_process_time"].mean()
        df["TAZ_System_rs"]=df["TAZ_System"] - df["TAZ_System_rf"]
        df["TAZ_System_rs_weighted"]=df["TAZ_System"] - df["TAZ_System_rf_weighted"]

        mean_taz_system_mg=round(
            (df["TAZ_System"] * df.loc[:, "total_process_time"]).sum() / df.loc[:, "total_process_time"].sum(), 2)
        mean_taz_system_mg_rf=round((df.loc[:, "TAZ_System_rf"] * df.loc[:, "total_process_time"]).sum() / df.loc[:,
                                                                                                           "total_process_time"].sum(),
                                    2)
        mean_taz_system_mg_rs=round((df.loc[:, "TAZ_System_rs"] * df.loc[:, "total_process_time"]).sum() / df.loc[:,
                                                                                                           "total_process_time"].sum(),
                                    2)

        # TAZ Produktion
        df["TAZ_Produktion"]=df["deviation_release"]
        y2=df["TAZ_Produktion"]
        df["release_rank_deviation"]=df["actual_release_Rank"] - df["planned_release_Rank"]
        df["release_rank_deviation_weighted"]=df["actual_release_Rank_weighted"] - df["planned_release_Rank_weighted"]
        df["TAZ_Produktion_rf"]=df["release_rank_deviation"] * df.loc[:, "interarrival_time"].mean()
        df["TAZ_Produktion_rf_weighted"]=df["release_rank_deviation_weighted"] * df.loc[:,
                                                                                 "interarrival_time"].mean() / df.loc[:,
                                                                                                               "total_process_time"].mean()
        df["TAZ_Produktion_rs"]=df["TAZ_Produktion"] - df["TAZ_Produktion_rf"]
        df["TAZ_Produktion_rs_weighted"]=df["TAZ_Produktion"] - df["TAZ_Produktion_rf_weighted"]

        mean_taz_produktion_mg=round(
            (df["TAZ_Produktion"] * df.loc[:, "total_process_time"]).sum() / df.loc[:, "total_process_time"].sum(), 2)
        mean_taz_produktion_mg_rf=round(
            (df.loc[:, "TAZ_Produktion_rf"] * df.loc[:, "total_process_time"]).sum() / df.loc[:,
                                                                                       "total_process_time"].sum(), 2)
        mean_taz_produktion_mg_rs=round(
            (df.loc[:, "TAZ_Produktion_rs"] * df.loc[:, "total_process_time"]).sum() / df.loc[:,
                                                                                       "total_process_time"].sum(), 2)

        # TAA Produktion
        df["TAA_Produktion"]=df["lateness_prod"]
        y3=df["TAA_Produktion"]
        df["finish_rank_deviation"]=df["actual_finish_Rank"] - df["planned_finish_Rank"]
        df["finish_rank_deviation_weighted"]=df["actual_finish_Rank_weighted"] - df["planned_finish_Rank_weighted"]
        df["TAA_Produktion_rf"]=df["finish_rank_deviation"] * df.loc[:, "interarrival_time"].mean()
        df["TAA_Produktion_rf_weighted"]=df["finish_rank_deviation_weighted"] * df.loc[:,
                                                                                "interarrival_time"].mean() / df.loc[:,
                                                                                                              "total_process_time"].mean()
        df["TAA_Produktion_rs"]=df["TAA_Produktion"] - df["TAA_Produktion_rf"]
        df["TAA_Produktion_rs_weighted"]=df["TAA_Produktion"] - df["TAA_Produktion_rf_weighted"]

        mean_taa_produktion_mg=round(
            (df["TAA_Produktion"] * df.loc[:, "total_process_time"]).sum() / df.loc[:, "total_process_time"].sum(), 2)
        mean_taa_produktion_mg_rf=round(
            (df.loc[:, "TAA_Produktion_rf"] * df.loc[:, "total_process_time"]).sum() / df.loc[:,
                                                                                       "total_process_time"].sum(), 2)
        mean_taa_produktion_mg_rs=round(
            (df.loc[:, "TAA_Produktion_rs"] * df.loc[:, "total_process_time"]).sum() / df.loc[:,
                                                                                       "total_process_time"].sum(), 2)

        df["Durchlaufzeit_Gesamt"]=df["throughput_time"]
        df["Durchlaufzeit_Pool"]=df["pool_throughput_time"]
        df["Durchlaufzeit_Shop"]=df["shop_throughput_time"]

        y4=df["Durchlaufzeit_Pool"]
        mean_zdl_pool_mg=round(
            (df["Durchlaufzeit_Pool"] * df.loc[:, "total_process_time"]).sum() / df.loc[:, "total_process_time"].sum(),
            2)
        y5=df["Durchlaufzeit_Shop"]
        mean_zdl_shop_mg=round(
            (df["Durchlaufzeit_Shop"] * df.loc[:, "total_process_time"]).sum() / df.loc[:, "total_process_time"].sum(),
            2)
        y6=df["Durchlaufzeit_Gesamt"]
        mean_zdl_gesamt_mg=round((df["Durchlaufzeit_Gesamt"] * df.loc[:, "total_process_time"]).sum() / df.loc[:,
                                                                                                        "total_process_time"].sum(),
                                 2)

        plotgesamt: bool=True
        plotzerlegung: bool=True
        plotzerlegung_gewichtet: bool=True
        korrelationsplot: bool=True

        if plotgesamt == True:
            plt.clf()
            plt.cla()
            gs_kw=dict(width_ratios=[1, 1, 1])

            fig, axs=plt.subplots(ncols=3, nrows=2, gridspec_kw=gs_kw, figsize=(11.7, 8.27))

            fig.suptitle('Freigabe: %s | TAX-Analyse: Gesamtsystem' % (freigabe), y=0.98,
                         fontsize=16)

            plt.subplots_adjust(left=0.07,
                                bottom=0.07,
                                right=0.93,
                                top=0.90,
                                wspace=0.3,
                                hspace=0.3)

            axs[0, 0].hist(y1, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                           rwidth=0.8)

            axs[0, 1].hist(y2, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                           rwidth=0.8)

            axs[0, 2].hist(y3, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                           rwidth=0.8)

            axs[1, 0].hist(y4, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                           rwidth=0.8)

            axs[1, 1].hist(y5, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                           rwidth=0.8)

            axs[1, 2].hist(y6, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                           rwidth=0.8)

            axs[0, 0].axvline(x=y1.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[0, 1].axvline(x=y2.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[0, 2].axvline(x=y3.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[1, 0].axvline(x=y4.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[1, 1].axvline(x=y5.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[1, 2].axvline(x=y6.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)

            axs[0, 0].set_xlabel("Terminabweichung [BKT]")
            axs[0, 1].set_xlabel("Terminabweichung [BKT]")
            axs[0, 2].set_xlabel("Terminabweichung [BKT]")
            axs[1, 0].set_xlabel("Durchlaufzeit [BKT]")
            axs[1, 1].set_xlabel("Durchlaufzeit [BKT]")
            axs[1, 2].set_xlabel("Durchlaufzeit [BKT]")

            axs[0, 0].title.set_text("Terminabweichung im Systemzugang")
            axs[0, 1].title.set_text("Terminabweichung im Zugang Produktion")
            axs[0, 2].title.set_text("Terminabweichung im Abgang Produktion")
            axs[1, 0].title.set_text("Pool-Durchlaufzeit")
            axs[1, 1].title.set_text("Shop-Durchlaufzeit")
            axs[1, 2].title.set_text("Gesamt-Durchlaufzeit")

            textstring1='$TAZ_{m}$' ": %s | "'$TAZ_{std}}$' ": %s\n" '$TAZ_{min}$' ": %s | "'$TAZ_{max}$' ": " \
                        "%s\n" '$TAZ_{mg}$' ": %s | "'$TAZ_{med}$' ": " "%s" % (
                            round(y1.mean(), 2), round(y1.std(), 2), round(y1.min(), 2), round(y1.max(), 2),
                            mean_taz_system_mg,
                            round(y1.median(), 2))

            textstring2='$TAZ_{m}$' ": %s | "'$TAZ_{std}}$' ": %s\n" '$TAZ_{min}$' ": %s | "'$TAZ_{max}$' ": " \
                        "%s\n" '$TAZ_{mg}$' ": %s | "'$TAZ_{med}$' ": " "%s" % (
                            round(y2.mean(), 2), round(y2.std(), 2), round(y2.min(), 2), round(y2.max(), 2),
                            mean_taz_produktion_mg,
                            round(y2.median(), 2))

            textstring3='$TAA_{m}$' ": %s | "'$TAA_{std}}$' ": %s\n" '$TAA_{min}$' ": %s | "'$TAA_{max}$' ": " \
                        "%s\n" '$TAA_{mg}$' ": %s | "'$TAA_{med}$' ": " "%s" % (
                            round(y3.mean(), 2), round(y3.std(), 2), round(y3.min(), 2), round(y3.max(), 2),
                            mean_taa_produktion_mg,
                            round(y3.median(), 2))

            textstring4='$ZDL_{m}$' ": %s | "'$ZDL_{std}}$' ": %s\n" '$ZDL_{min}$' ": %s | "'$ZDL_{max}$' ": " \
                        "%s\n" '$ZDL_{mg}$' ": %s | "'$ZDL_{med}$' ": " "%s" % (
                            round(y4.mean(), 2), round(y4.std(), 2), round(y4.min(), 2), round(y4.max(), 2),
                            mean_zdl_pool_mg,
                            round(y4.median(), 2))

            textstring5='$ZDL_{m}$' ": %s | "'$ZDL_{std}}$' ": %s\n" '$ZDL_{min}$' ": %s | "'$ZDL_{max}$' ": " \
                        "%s\n" '$ZDL_{mg}$' ": %s | "'$ZDL_{med}$' ": " "%s" % (
                            round(y5.mean(), 2), round(y5.std(), 2), round(y5.min(), 2), round(y5.max(), 2),
                            mean_zdl_shop_mg,
                            round(y5.median(), 2))

            textstring6='$ZDL_{m}$' ": %s | "'$ZDL_{std}}$' ": %s\n" '$ZDL_{min}$' ": %s | "'$ZDL_{max}$' ": " \
                        "%s\n" '$ZDL_{mg}$' ": %s | "'$ZDL_{med}$' ": " "%s" % (
                            round(y6.mean(), 2), round(y6.std(), 2), round(y6.min(), 2), round(y6.max(), 2),
                            mean_zdl_gesamt_mg,
                            round(y6.median(), 2))

            props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=0.7)
            axs[0, 0].text(0.66, 0.87, textstring1, transform=axs[0, 0].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[0, 1].text(0.66, 0.87, textstring2, transform=axs[0, 1].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[0, 2].text(0.66, 0.87, textstring3, transform=axs[0, 2].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[1, 0].text(0.66, 0.87, textstring4, transform=axs[1, 0].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[1, 1].text(0.66, 0.87, textstring5, transform=axs[1, 1].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[1, 2].text(0.66, 0.87, textstring6, transform=axs[1, 2].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)

            plotName="Gesamtsystem_" + 'TAX' + '.svg'
            path=self.get_directory()
            subfolder="Plot/TAX/"
            file=path + subfolder + self.sim.model_panel.experiment_name + plotName
            fig.savefig(file)
            print(f"Save {file}")
            plt.close(fig)

        if plotzerlegung == True:
            plt.clf()
            plt.cla()
            y1=df["TAZ_System_rf"]
            y2=df["TAZ_Produktion_rf"]
            y3=df["TAA_Produktion_rf"]
            y4=df["TAZ_System_rs"]
            y5=df["TAZ_Produktion_rs"]
            y6=df["TAA_Produktion_rs"]

            gs_kw=dict(width_ratios=[1, 1, 1])

            fig, axs=plt.subplots(ncols=3, nrows=2, gridspec_kw=gs_kw, figsize=(11.7, 8.27))

            fig.suptitle('Freigabe: %s | TAX-Analyse: Gesamtsystem | Zerlegung mittels Rangabweichung' % (freigabe),
                         y=0.98,
                         fontsize=16)

            plt.subplots_adjust(left=0.07,
                                bottom=0.07,
                                right=0.93,
                                top=0.90,
                                wspace=0.3,
                                hspace=0.3)

            axs[0, 0].hist(y1, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                           rwidth=0.8)

            axs[0, 1].hist(y2, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                           rwidth=0.8)

            axs[0, 2].hist(y3, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                           rwidth=0.8)

            axs[1, 0].hist(y4, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                           rwidth=0.8)

            axs[1, 1].hist(y5, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                           rwidth=0.8)

            axs[1, 2].hist(y6, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                           rwidth=0.8)

            axs[0, 0].axvline(x=y1.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[0, 1].axvline(x=y2.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[0, 2].axvline(x=y3.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[1, 0].axvline(x=y4.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[1, 1].axvline(x=y5.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[1, 2].axvline(x=y6.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)

            axs[0, 0].set_xlabel("Terminabweichung [BKT]")
            axs[0, 1].set_xlabel("Terminabweichung [BKT]")
            axs[0, 2].set_xlabel("Terminabweichung [BKT]")
            axs[1, 0].set_xlabel("Terminabweichung [BKT]")
            axs[1, 1].set_xlabel("Terminabweichung [BKT]")
            axs[1, 2].set_xlabel("Terminabweichung [BKT]")

            axs[0, 0].title.set_text("$TAZ_{rf}$ im Systemzugang")
            axs[0, 1].title.set_text("$TAZ_{rf}$ im Zugang Produktion")
            axs[0, 2].title.set_text("$TAA_{rf}$ im Abgang Produktion")
            axs[1, 0].title.set_text("$TAZ_{rs}$ im Systemzugang")
            axs[1, 1].title.set_text("$TAZ_{rs}$ im Zugang Produktion")
            axs[1, 2].title.set_text("$TAA_{rs}$ im Abgang Produktion")

            textstring1='$TAZ_{m,rf}$' ": %s | "'$TAZ_{std,rf}}$' ": %s\n" '$TAZ_{min,rf}$' ": %s | "'$TAZ_{max,' \
                        'rf}$' ": " \
                        "%s\n" '$TAZ_{mg,rf}$' ": %s | "'$TAZ_{med,rf}$' ": " "%s" % (
                            round(y1.mean(), 2), round(y1.std(), 2), round(y1.min(), 2), round(y1.max(), 2),
                            mean_taz_system_mg_rf,
                            round(y1.median(), 2))

            textstring2='$TAZ_{m,rf}$' ": %s | "'$TAZ_{std,rf}}$' ": %s\n" '$TAZ_{min,rf}$' ": %s | "'$TAZ_{max,' \
                        'rf}$' ": " \
                        "%s\n" '$TAZ_{mg,rf}$' ": %s | "'$TAZ_{med,rf}$' ": " "%s" % (
                            round(y2.mean(), 2), round(y2.std(), 2), round(y2.min(), 2), round(y2.max(), 2),
                            mean_taz_produktion_mg_rf,
                            round(y2.median(), 2))

            textstring3='$TAA_{m,rf}$' ": %s | "'$TAA_{std,rf}}$' ": %s\n" '$TAA_{min,rf}$' ": %s | "'$TAA_{max,' \
                        'rf}$' ": " \
                        "%s\n" '$TAA_{mg,rf}$' ": %s | "'$TAA_{med,rf}$' ": " "%s" % (
                            round(y3.mean(), 2), round(y3.std(), 2), round(y3.min(), 2), round(y3.max(), 2),
                            mean_taa_produktion_mg_rf,
                            round(y3.median(), 2))

            textstring4='$TAZ_{m,rs}$' ": %s | "'$TAZ_{std,rs}}$' ": %s\n" '$TAZ_{min,rs}$' ": %s | "'$TAZ_{max,' \
                        'rs}$' ": " \
                        "%s\n" '$TAZ_{mg,rs}$' ": %s | "'$TAZ_{med,rs}$' ": " "%s" % (
                            round(y4.mean(), 2), round(y4.std(), 2), round(y4.min(), 2), round(y4.max(), 2),
                            mean_taz_system_mg_rs,
                            round(y4.median(), 2))

            textstring5='$TAZ_{m,rs}$' ": %s | "'$TAZ_{std,rs}}$' ": %s\n" '$TAZ_{min,rs}$' ": %s | "'$TAZ_{max,' \
                        'rs}$' ": " \
                        "%s\n" '$TAZ_{mg,rs}$' ": %s | "'$TAZ_{med,rs}$' ": " "%s" % (
                            round(y5.mean(), 2), round(y5.std(), 2), round(y5.min(), 2), round(y5.max(), 2),
                            mean_taz_produktion_mg_rs,
                            round(y5.median(), 2))

            textstring6='$TAA_{m,rs}$' ": %s | "'$TAA_{std,rs}}$' ": %s\n" '$TAA_{min,rs}$' ": %s | "'$TAA_{max,' \
                        'rs}$' ": " \
                        "%s\n" '$TAA_{mg,rs}$' ": %s | "'$TAA_{med,rs}$' ": " "%s" % (
                            round(y6.mean(), 2), round(y6.std(), 2), round(y6.min(), 2), round(y6.max(), 2),
                            mean_taa_produktion_mg_rs,
                            round(y6.median(), 2))

            props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=0.7)
            axs[0, 0].text(0.66, 0.87, textstring1, transform=axs[0, 0].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[0, 1].text(0.66, 0.87, textstring2, transform=axs[0, 1].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[0, 2].text(0.66, 0.87, textstring3, transform=axs[0, 2].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[1, 0].text(0.66, 0.87, textstring4, transform=axs[1, 0].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[1, 1].text(0.66, 0.87, textstring5, transform=axs[1, 1].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[1, 2].text(0.66, 0.87, textstring6, transform=axs[1, 2].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)

            plotName="Gesamtsystem_" + 'TAX_R' + '.svg'
            path=self.get_directory()
            subfolder="Plot/TAX/"
            file=path + subfolder + self.sim.model_panel.experiment_name + plotName
            fig.savefig(file)
            print(f"Save {file}")
            plt.close(fig)

        if plotzerlegung_gewichtet == True:
            plt.clf()
            plt.cla()
            y1=df["TAZ_System_rf_weighted"]
            y2=df["TAZ_Produktion_rf_weighted"]
            y3=df["TAA_Produktion_rf_weighted"]
            y4=df["TAZ_System_rs_weighted"]
            y5=df["TAZ_Produktion_rs_weighted"]
            y6=df["TAA_Produktion_rs_weighted"]

            gs_kw=dict(width_ratios=[1, 1, 1])

            fig, axs=plt.subplots(ncols=3, nrows=2, gridspec_kw=gs_kw, figsize=(11.7, 8.27))

            fig.suptitle(
                'Freigabe: %s | TAX-Analyse: Gesamtsystem | Zerlegung mittels gewichteter Rangabweichung' % (freigabe),
                y=0.98,
                fontsize=16)

            plt.subplots_adjust(left=0.07,
                                bottom=0.07,
                                right=0.93,
                                top=0.90,
                                wspace=0.3,
                                hspace=0.3)

            axs[0, 0].hist(y1, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                           rwidth=0.8)

            axs[0, 1].hist(y2, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                           rwidth=0.8)

            axs[0, 2].hist(y3, bins=20, density=True, histtype='bar', color='#1f77b4', stacked=False,
                           rwidth=0.8)

            axs[1, 0].hist(y4, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                           rwidth=0.8)

            axs[1, 1].hist(y5, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                           rwidth=0.8)

            axs[1, 2].hist(y6, bins=20, density=True, histtype='bar', color='#2ca02c', stacked=False,
                           rwidth=0.8)

            axs[0, 0].axvline(x=y1.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[0, 1].axvline(x=y2.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[0, 2].axvline(x=y3.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[1, 0].axvline(x=y4.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[1, 1].axvline(x=y5.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)
            axs[1, 2].axvline(x=y6.mean(), ymin=0, ymax=1, color="red", linestyle="--", alpha=0.35)

            axs[0, 0].set_xlabel("Terminabweichung [BKT]")
            axs[0, 1].set_xlabel("Terminabweichung [BKT]")
            axs[0, 2].set_xlabel("Terminabweichung [BKT]")
            axs[1, 0].set_xlabel("Terminabweichung [BKT]")
            axs[1, 1].set_xlabel("Terminabweichung [BKT]")
            axs[1, 2].set_xlabel("Terminabweichung [BKT]")

            axs[0, 0].title.set_text("$TAZ_{rf,g}$ im Systemzugang")
            axs[0, 1].title.set_text("$TAZ_{rf,g}$ im Zugang Produktion")
            axs[0, 2].title.set_text("$TAA_{rf,g}$ im Abgang Produktion")
            axs[1, 0].title.set_text("$TAZ_{rs,g}$ im Systemzugang")
            axs[1, 1].title.set_text("$TAZ_{rs,g}$ im Zugang Produktion")
            axs[1, 2].title.set_text("$TAA_{rs,g}$ im Abgang Produktion")

            textstring1='$TAZ_{m,rfg}$' ": %s | "'$TAZ_{std,rfg}}$' ": %s\n" '$TAZ_{min,rfg}$' ": %s | "'$TAZ_{max,' \
                        'rfg}$' ": " \
                        "%s\n"'$TAZ_{med,rfg}$' ": " "%s" % (
                            round(y1.mean(), 2), round(y1.std(), 2), round(y1.min(), 2), round(y1.max(), 2),
                            round(y1.median(), 2))

            textstring2='$TAZ_{m,rfg}$' ": %s | "'$TAZ_{std,rfg}}$' ": %s\n" '$TAZ_{min,rfg}$' ": %s | "'$TAZ_{max,' \
                        'rfg}$' ": " \
                        "%s\n"'$TAZ_{med,rfg}$' ": " "%s" % (
                            round(y2.mean(), 2), round(y2.std(), 2), round(y2.min(), 2), round(y2.max(), 2),
                            round(y2.median(), 2))

            textstring3='$TAA_{m,rfg}$' ": %s | "'$TAA_{std,rfg}}$' ": %s\n" '$TAA_{min,rfg}$' ": %s | "'$TAA_{max,' \
                        'rfg}$' ": " \
                        "%s\n"'$TAA_{med,rfg}$' ": " "%s" % (
                            round(y3.mean(), 2), round(y3.std(), 2), round(y3.min(), 2), round(y3.max(), 2),
                            round(y3.median(), 2))

            textstring4='$TAZ_{m,rsg}$' ": %s | "'$TAZ_{std,rsg}}$' ": %s\n" '$TAZ_{min,rsg}$' ": %s | "'$TAZ_{max,' \
                        'rsg}$' ": " \
                        "%s\n"'$TAZ_{med,rsg}$' ": " "%s" % (
                            round(y4.mean(), 2), round(y4.std(), 2), round(y4.min(), 2), round(y4.max(), 2),
                            round(y4.median(), 2))

            textstring5='$TAZ_{m,rsg}$' ": %s | "'$TAZ_{std,rsg}}$' ": %s\n" '$TAZ_{min,rsg}$' ": %s | "'$TAZ_{max,' \
                        'rsg}$' ": " \
                        "%s\n"'$TAZ_{med,rsg}$' ": " "%s" % (
                            round(y5.mean(), 2), round(y5.std(), 2), round(y5.min(), 2), round(y5.max(), 2),
                            round(y5.median(), 2))

            textstring6='$TAA_{m,rsg}$' ": %s | "'$TAA_{std,rsg}}$' ": %s\n" '$TAA_{min,rsg}$' ": %s | "'$TAA_{max,' \
                        'rsg}$' ": " \
                        "%s\n"'$TAA_{med,rsg}$' ": " "%s" % (
                            round(y6.mean(), 2), round(y6.std(), 2), round(y6.min(), 2), round(y6.max(), 2),
                            round(y6.median(), 2))

            props=dict(boxstyle='round', facecolor='#d2d2d2', alpha=0.7)
            axs[0, 0].text(0.66, 0.87, textstring1, transform=axs[0, 0].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[0, 1].text(0.66, 0.87, textstring2, transform=axs[0, 1].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[0, 2].text(0.66, 0.87, textstring3, transform=axs[0, 2].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[1, 0].text(0.66, 0.87, textstring4, transform=axs[1, 0].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[1, 1].text(0.66, 0.87, textstring5, transform=axs[1, 1].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)
            axs[1, 2].text(0.66, 0.87, textstring6, transform=axs[1, 2].transAxes, fontsize=7,
                           horizontalalignment='center',
                           verticalalignment='center', bbox=props)

            plotName="Gesamtsystem_" + 'TAX_RG' + '.svg'
            path=self.get_directory()
            subfolder="Plot/TAX/"
            file=path + subfolder + self.sim.model_panel.experiment_name + plotName
            fig.savefig(file)
            print(f"Save {file}")
            plt.close(fig)

        if korrelationsplot == True:
            plt.clf()
            plt.cla()
            # Korrelationsplot 1
            cols_to_plot=['TAZ_System', 'TAZ_Produktion', "TAZ_System_rf", "TAZ_System_rs", "TAZ_Produktion_rf",
                          "TAZ_Produktion_rs", 'total_process_time', 'throughput_time', 'length_of_routing']
            g=sns.pairplot(df[cols_to_plot], palette="crest_r", diag_kws={'common_norm': True}, hue="length_of_routing")

            plotName="Gesamtsystem_" + 'Korrelation_ZF_Routing' + '.png'
            path=self.get_directory()
            subfolder="Plot/TAX/"
            file=path + subfolder + self.sim.model_panel.experiment_name + plotName
            g.figure.savefig(file, dpi=400)
            print(f"Save {file}")
            g.fig.clf()
            plt.close(g.fig)

            # Korrelationsplot 2
            cols_to_plot=['TAZ_System', 'TAZ_Produktion', "TAZ_System_rf_weighted", "TAZ_System_rs_weighted",
                          "TAZ_Produktion_rf_weighted", "TAZ_Produktion_rs_weighted", 'total_process_time',
                          'throughput_time', 'length_of_routing']
            g=sns.pairplot(df[cols_to_plot], palette="crest_r", diag_kws={'common_norm': True}, hue="length_of_routing")

            plotName="Gesamtsystem_" + 'Korrelation_ZF_Routing_Gew' + '.png'
            path=self.get_directory()
            subfolder="Plot/TAX/"
            file=path + subfolder + self.sim.model_panel.experiment_name + plotName
            g.figure.savefig(file, dpi=400)
            print(f"Save {file}")
            g.fig.clf()
            plt.close(g.fig)

            # Korrelationsplot 3
            cols_to_plot=['TAZ_Produktion', 'TAA_Produktion', "TAZ_Produktion_rf", "TAZ_Produktion_rs", "TAA_Produktion_rf",
                          "TAA_Produktion_rs", 'total_process_time', 'throughput_time', 'length_of_routing']
            g=sns.pairplot(df[cols_to_plot], palette="crest_r", diag_kws={'common_norm': True}, hue="length_of_routing")

            plotName="Gesamtsystem_" + 'Korrelation_FA_Routing' + '.png'
            path=self.get_directory()
            subfolder="Plot/TAX/"
            file=path + subfolder + self.sim.model_panel.experiment_name + plotName
            g.figure.savefig(file, dpi=400)
            print(f"Save {file}")
            g.fig.clf()
            plt.close(g.fig)

            # Korrelationsplot 4
            cols_to_plot=['TAZ_Produktion', 'TAA_Produktion', "TAZ_Produktion_rf_weighted", "TAZ_Produktion_rs_weighted",
                          "TAA_Produktion_rf_weighted", "TAA_Produktion_rs_weighted", 'total_process_time',
                          'throughput_time', 'length_of_routing']
            g=sns.pairplot(df[cols_to_plot], palette="crest_r", diag_kws={'common_norm': True}, hue="length_of_routing")

            plotName="Gesamtsystem_" + 'Korrelation_FA_Routing_Gew' + '.png'
            path=self.get_directory()
            subfolder="Plot/TAX/"
            file=path + subfolder + self.sim.model_panel.experiment_name + plotName
            g.figure.savefig(file, dpi=400)
            print(f"Save {file}")
            g.fig.clf()
            plt.close(g.fig)

            # Korrelationsplot 5
            cols_to_plot=['TAZ_System', 'TAA_Produktion', "TAZ_System_rf", "TAZ_System_rs", "TAA_Produktion_rf",
                          "TAA_Produktion_rs", 'total_process_time', 'throughput_time', 'length_of_routing']
            g=sns.pairplot(df[cols_to_plot], palette="crest_r", diag_kws={'common_norm': True}, hue="length_of_routing")

            plotName="Gesamtsystem_" + 'Korrelation_ZA_Routing' + '.png'
            path=self.get_directory()
            subfolder="Plot/TAX/"
            file=path + subfolder + self.sim.model_panel.experiment_name + plotName
            g.figure.savefig(file, dpi=400)
            print(f"Save {file}")
            g.fig.clf()
            plt.close(g.fig)

            # Korrelationsplot 6
            cols_to_plot=['TAZ_System', 'TAA_Produktion', "TAZ_System_rf_weighted", "TAZ_System_rs_weighted",
                          "TAA_Produktion_rf_weighted", "TAA_Produktion_rs_weighted", 'total_process_time',
                          'throughput_time', 'length_of_routing']
            g=sns.pairplot(df[cols_to_plot], palette="crest_r", diag_kws={'common_norm': True}, hue="length_of_routing")

            plotName="Gesamtsystem_" + 'Korrelation_ZA_Routing_Gew' + '.png'
            path=self.get_directory()
            subfolder="Plot/TAX/"
            file=path + subfolder + self.sim.model_panel.experiment_name + plotName
            g.figure.savefig(file, dpi=400)
            print(f"Save {file}")
            g.fig.clf()
            plt.close(g.fig)

    # Creation of needed folders to save analyses-----------------------------------------------------------------------
    def create_subfolder(self, name):
        base = self.get_directory()
        path= base + "Plot/DUDI"

        try:
            os.makedirs(path)
        except OSError:
            #print("Creation of the directory %s failed" % path)
            print(".")
        else:
            print("Successfully created the directory %s " % path)

        path=base + "Plot/TAX"

        try:
            os.makedirs(path)
        except OSError:
            #print("Creation of the directory %s failed" % path)
            print(".")
        else:
            print("Successfully created the directory %s " % path)

        path=base + "Plot/BELA"

        try:
            os.makedirs(path)
        except OSError:
            # print("Creation of the directory %s failed" % path)
            print(".")
        else:
            print("Successfully created the directory %s " % path)

        path=base + "Plot/PKL"

        try:
            os.makedirs(path)
        except OSError:
            # print("Creation of the directory %s failed" % path)
            print(".")
        else:
            print("Successfully created the directory %s " % path)
