
import random
import numpy as np
import pandas as pd
import os
import sys
import statistics

rng = np.random.default_rng(12345)

random_generator = random.Random()
random_generator.seed(99999)

truncation = 4 #1 + (int(sys.argv[1]))

choice = "Erlang-1" #str(sys.argv[2])

Ergebnisliste = []
Zwischen = []

import random
import numpy as np
import pandas as pd
import os
import sys
import statistics

random_generator = random.Random()
random_generator.seed(99999)

truncation = 1 + (int(sys.argv[1]))

choice = str(sys.argv[2])

Ergebnisliste = []
Zwischen = []

if choice in ["LogNormal" , "Normal", "Weibull"]:

    if choice == "Normal":
        j = 0
        j_limit = 10000
        j_increment = 100
        k_start = 100
        k_limit = 10000
        k_increment = 100
    if choice == "LogNormal":
        j = -2000
        j_limit = 3000
        j_increment = 10
        k_start = 10
        k_limit = 3000
        k_increment = 10
    if choice == "Weibull":
        j = 10
        j_limit = 3000
        j_increment = 10
        k_start = 10
        k_limit = 3500
        k_increment = 10

    while j<j_limit:
        if truncation <= j/1000:
            j += j_increment
            continue
        k = k_start
        while k<k_limit:

            l = 0
            iteration = 0
            liste = []
            while l<10:
                fail = False
                return_value = np.inf
                while return_value > truncation or return_value <= 0:
                    if choice == "Normal":
                        return_value = random_generator.normalvariate(j/1000, k/1000)
                    if choice == "LogNormal":
                        return_value = random_generator.lognormvariate(j/1000, k/1000)
                        iteration += 1
                        if iteration > 1000000 and l<10:
                            fail = True
                            break
                    if choice == "Weibull":
                        return_value = random_generator.weibullvariate(j/1000,k/1000)
                if fail == True:
                    break
                liste.append(return_value)
                l += 1
            if fail == True:
                k += k_increment

            else:
                result_mean = statistics.mean(liste)
                result_std  = statistics.stdev(liste)

                Zwischen = []
                Zwischen.append(j/1000)
                Zwischen.append(k/1000)
                Zwischen.append(result_mean)
                Zwischen.append(result_std)
                Zwischen.append(truncation)
                Ergebnisliste.append(Zwischen)
                k += k_increment
        j += j_increment

    df_run = pd.DataFrame(Ergebnisliste)
    df_run.columns = ["Mean","Std","result_mean","result_stv","truncated"]


if choice in ["Erlang-1", "Erlang-2"]:

    j = 100

    while j < 10000:
        if truncation <= j / 1000:
            j += 1
            continue

        if choice == "Erlang-1":
            return_value = rng.exponential(j/1000, 100000000)
        if choice == "Erlang-2":
            return_value = rng.gamma(2,j/1000/2, 100000000)


        clipped_data=return_value[(truncation >= return_value)]

        result_mean = clipped_data.mean()

        Zwischen = []
        Zwischen.append(j/1000)
        Zwischen.append(result_mean)
        Zwischen.append(truncation)
        Ergebnisliste.append(Zwischen)
        j += 1

    df_run = pd.DataFrame(Ergebnisliste)
    df_run.columns = ["Mean", "result_mean", "truncated"]

number = str(truncation)
path = "/bigwork/nhk2mue1/Distribution2/"
file = path + choice + number + '.csv'
df_run.to_csv(file, index=False)



