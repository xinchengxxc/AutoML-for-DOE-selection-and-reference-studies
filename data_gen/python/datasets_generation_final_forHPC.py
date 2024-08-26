# %%
import pandas as pd
import numpy as np
from doepy import build
import os
from sklearn.preprocessing import MinMaxScaler
import re
from scipy.linalg import det
import time
import shutil
from datetime import datetime
import heapq
import sys
# %%
import GPy
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction
from emukit.core.optimization import GradientAcquisitionOptimizer
# %%
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from modAL.models import ActiveLearner, CommitteeRegressor
# %%
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import tqdm
from baal.modelwrapper import ModelWrapper
from baal.bayesian import MCDropoutConnectModule
# %%
#define the function to choose a model
def choose_model(model):
    if model == 'Model41':
        circuit = 'R0-p(R1,C1)'
        param_space = {
            'R0': [1, 3],
            'R1': [1, 3],
            'C1': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model42':
        circuit = 'W0-p(R1,C1)'
        param_space = {
            'W0': [0, 1.7],
            'R1': [1, 3],
            'C1': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model43':
        circuit = 'T0'
        param_space = {
            'T0': [1, 3],
            'T1': [2, 4],
            'T2': [0, 1.7],
            'T3': [2.30,2.30],
            'f' : [-4, 5]
        }
    elif model == 'Model44':
        circuit = 'R0-p(R1,C1)-Wo1-L1-Gs1-T1-CPE1-TLMQ1-W1-Zarc1'
        param_space = {
            'R0': [2.7, 2.7],
            'R1': [1, 3],#
            'C1': [-6,-2],#
            'Wo1': [2.7,2.7],
            'Wo2': [3.7,3.7],
            'L1': [-4,-2],#
            'Gs1': [2.7,2.7],
            'Gs2': [1,1],
            'Gs3': [-0.3,-0.3],
            'T1': [2.7,2.7],
            'T2' : [3,3],
            'T3' : [1.3,1.3],
            'T4' : [2.3,2.3],
            'CPE1' : [-0.1,-0.1],
            'CPE2' : [-0.1,-0.1],
            'TLMQ1' : [2,2],
            'TLMQ2' : [2,2],
            'TLMQ3' : [-1,-1],
            'W1' : [0.7,0.7],
            'Zarc1' : [2,2],
            'Zarc2' : [2,2],
            'Zarc3' : [0.17,0.17],
            'f' : [-4, 5]
        }
    elif model == 'Model81':
        circuit = 'R0-p(R1,C1)-p(R2,C2)-p(R3,C3)'
        param_space = {
            'R0': [1, 3],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'R3': [1, 3],
            'C3': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model82':
        circuit = 'Gs0-p(R1,C1)-p(R2,C2)'
        param_space = {
            'Gs0': [1, 3],
            'Gs1': [0, 2],
            'Gs2': [-1, 0],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model83':
        circuit = 'T0-p(R1,C1)-Wo1'
        param_space = {
            'T0': [1, 3],
            'T1': [2, 4],
            'T2': [0, 1.7],
            'T3': [2.30,2.30],#######################
            'R1': [1, 3],
            'C1': [-6,-2],
            'Wo1': [1, 3],
            'Wo2': [2.7, 3.7],
            'f' : [-4, 5]
        }
    elif model == 'Model84':
        circuit = 'R0-p(R1,C1)-Wo1-L1-Gs1-T1-CPE1-TLMQ1-W1-Zarc1'
        param_space = {
            'R0': [2.7, 2.7],
            'R1': [1, 3],#
            'C1': [-6,-2],#
            'Wo1': [1,3],#
            'Wo2': [2.7,3.7],#
            'L1': [-4,-2],#
            'Gs1': [2.7,2.7],
            'Gs2': [1,1],
            'Gs3': [-0.3,-0.3],
            'T1': [2.7,2.7],
            'T2' : [3,3],
            'T3' : [1.3,1.3],
            'T4' : [2.3,2.3],
            'CPE1' : [-0.39,0.3],#
            'CPE2' : [-0.3,0],#
            'TLMQ1' : [2,2],
            'TLMQ2' : [2,2],
            'TLMQ3' : [-1,-1],
            'W1' : [0.7,0.7],
            'Zarc1' : [2,2],
            'Zarc2' : [2,2],
            'Zarc3' : [0.17,0.17],
            'f' : [-4, 5]
        }
    elif model == 'Model101':
        circuit = 'R0-p(R1,C1)-p(R2,C2)-p(R3,C3)-p(R4,C4)'
        param_space = {
            'R0': [1, 3],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'R3': [1, 3],
            'C3': [-6,-2],
            'R4': [1, 3],
            'C4': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model102':
        circuit = 'TLMQ0-p(R1,C1)-p(R2,C2)-Wo1'
        param_space = {
            'TLMQ1': [1, 3],
            'TLMQ2': [1, 3],
            'TLMQ3': [-1, 0],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'Wo1': [1, 3],
            'Wo2': [2.7, 3.7],
            'f' : [-4, 5]
        }
    elif model == 'Model103':
        circuit = 'L1-Gs1-T1-p(R1,C1)'
        param_space = {
            'L1': [-4, -2],
            'Gs1': [1, 3],
            'Gs2': [0, 2],
            'Gs3': [-1, 0],
            'T1': [1, 3],
            'T2': [2, 4],
            'T3': [0, 1.7],
            'T4': [2.30,2.30],
            'R1': [1, 3],
            'C1': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model104':
        circuit = 'R0-p(R1,C1)-Wo1-L1-Gs1-T1-CPE1-TLMQ1-W1-Zarc1'
        param_space = {
            'R0': [1, 3],#
            'R1': [1, 3],#
            'C1': [-6,-2],#
            'Wo1': [1,3],#
            'Wo2': [2.7,3.7],#
            'L1': [-4,-2],#
            'Gs1': [2.7,2.7],
            'Gs2': [1,1],
            'Gs3': [-0.3,-0.3],
            'T1': [2.7,2.7],
            'T2' : [3,3],
            'T3' : [1.3,1.3],
            'T4' : [2.3,2.3],
            'CPE1' : [-0.39,0.3],#
            'CPE2' : [-0.3,0],#
            'TLMQ1' : [2,2],
            'TLMQ2' : [2,2],
            'TLMQ3' : [-1,-1],
            'W1' : [0,1.7],#
            'Zarc1' : [2,2],
            'Zarc2' : [2,2],
            'Zarc3' : [0.17,0.17],
            'f' : [-4, 5]
        }
    elif model == 'Model121':
        circuit = 'R0-p(R1,C1)-p(R2,C2)-p(R3,C3)-p(R4,C4)-p(R5,C5)'
        param_space = {
            'R0': [1, 3],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'R3': [1, 3],
            'C3': [-6,-2],
            'R4': [1, 3],
            'C4': [-6,-2],
            'R5': [1, 3],
            'C5': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model122':
        circuit = 'CPE1-Zarc0-p(R1,C1)-p(R2,C2)-p(R3,C3)'
        param_space = {
            'CPE1': [-0.39,0.3],
            'CPE2': [-0.3,0],
            'Zarc0': [1, 3],
            'Zarc1': [1, 3],
            'Zarc2': [-1, 0.176],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'R3': [1, 3],
            'C3': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model123':
        circuit = 'CPE0-Zarc0-Wo0-p(R1,C1)-p(R2,C2)'
        param_space = {
            'CPE0': [-0.39,0.3],
            'CPE1': [-0.3,0],
            'Zarc0': [1, 3],
            'Zarc1': [1, 3],
            'Zarc2': [-1, 0.176],
            'Wo0': [1, 3],
            'Wo1': [2.7, 3.7],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model124':
        circuit = 'R0-p(R1,C1)-Wo1-L1-Gs1-T1-CPE1-TLMQ1-W1-Zarc1'
        param_space = {
            'R0': [1, 3],#
            'R1': [1, 3],#
            'C1': [-6,-2],#
            'Wo1': [1,3],#
            'Wo2': [2.7,3.7],#
            'L1': [-4,-2],#
            'Gs1': [1,3],#
            'Gs2': [0,2],#
            'Gs3': [-0.3,-0.3],
            'T1': [2.7,2.7],
            'T2' : [3,3],
            'T3' : [1.3,1.3],
            'T4' : [2.3,2.3],
            'CPE1' : [-0.39,0.3],#
            'CPE2' : [-0.3,0],#
            'TLMQ1' : [2,2],
            'TLMQ2' : [2,2],
            'TLMQ3' : [-1,-1],
            'W1' : [0,1.7],#
            'Zarc1' : [2,2],
            'Zarc2' : [2,2],
            'Zarc3' : [0.17,0.17],
            'f' : [-4, 5]
        }
    elif model == 'Model161':
        circuit = 'R0-p(R1,C1)-p(R2,C2)-p(R3,C3)-p(R4,C4)-p(R5,C5)-p(R6,C6)-p(R7,C7)'
        param_space = {
            'R0': [1, 3],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'R3': [1, 3],
            'C3': [-6,-2],
            'R4': [1, 3],
            'C4': [-6,-2],
            'R5': [1, 3],
            'C5': [-6,-2],
            'R6': [1, 3],
            'C6': [-6,-2],
            'R7': [1, 3],
            'C7': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model162':
        circuit = 'T0-TLMQ0-L0-p(R1,C1)-p(R2,C2)-p(R3,C3)-p(R4,C4)'
        param_space = {
            'T0': [1, 3],
            'T1': [2, 4],
            'T2': [0, 1.7],
            'T3': [2.30,2.30],#################################
            'TLMQ0': [1, 3],
            'TLMQ1': [1, 3],
            'TLMQ2': [-1, 0],
            'L0': [-4, -2],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'R3': [1, 3],
            'C3': [-6,-2],
            'R4': [1, 3],
            'C4': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model163':
        circuit = 'p(CPE0,R1,C1)-Wo1-p(R2,C2-W1)-Zarc1-T0'
        param_space = {
            'CPE0': [-0.39,0.3],
            'CPE1': [-0.3,0],
            'R1': [1, 3],
            'C1': [-6,-2],
            'Wo1': [1, 3],
            'Wo2': [2.7, 3.7],
            'R2': [1, 3],
            'C2': [-6,-2],
            'W1': [0, 1.7],
            'Zarc1': [1, 3],
            'Zarc2': [1, 3],
            'Zarc3': [-1, 0.176],
            'T0': [1, 3], 
            'T1': [1, 3],
            'T2': [-1, 0],
            'T3': [2.30,2.30],
            'f' : [-4, 5]
        }
    elif model == 'Model164':
        circuit = 'R0-p(R1,C1)-Wo1-L1-Gs1-T1-CPE1-TLMQ1-W1-Zarc1'
        param_space = {
            'R0': [1, 3],#
            'R1': [1, 3],#
            'C1': [-6,-2],#
            'Wo1': [1,3],#
            'Wo2': [2.7,3.7],#
            'L1': [-4,-2],#
            'Gs1': [1,3],#
            'Gs2': [0,2],#
            'Gs3': [-1,0],#
            'T1': [1,3],#
            'T2' : [2,4],#
            'T3' : [0,1.7],#
            'T4' : [2.3,2.3],
            'CPE1' : [-0.39,0.3],#
            'CPE2' : [-0.3,0],#
            'TLMQ1' : [2,2],
            'TLMQ2' : [2,2],
            'TLMQ3' : [-1,-1],
            'W1' : [0,1.7],#
            'Zarc1' : [2,2],
            'Zarc2' : [2,2],
            'Zarc3' : [0.17,0.17],
            'f' : [-4, 5]
        }
    elif model == 'Model201':
        circuit = 'R0-p(R1,C1)-p(R2,C2)-p(R3,C3)-p(R4,C4)-p(R5,C5)-p(R6,C6)-p(R7,C7)-p(R8,C8)-p(R9,C9)'
        param_space = {
            'R0': [1, 3],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'R3': [1, 3],
            'C3': [-6,-2],
            'R4': [1, 3],
            'C4': [-6,-2],
            'R5': [1, 3],
            'C5': [-6,-2],
            'R6': [1, 3],
            'C6': [-6,-2],
            'R7': [1, 3],
            'C7': [-6,-2],
            'R8': [1, 3],
            'C8': [-6,-2],
            'R9': [1, 3],
            'C9': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model202':
        circuit = 'Zarc0-W0-Gs0-p(R1,C1)-p(R2,C2)-p(R3,C3)-p(R4,C4)-p(R5,C5)-p(R6,C6)'
        param_space = {
            'Zarc0': [1, 3],
            'Zarc1': [1, 3],
            'Zarc2': [-1, 0.176],
            'W0': [0, 1.7],
            'Gs0': [1, 3],
            'Gs1': [0, 2],
            'Gs2': [-1, 0],
            'R1': [1, 3],
            'C1': [-6,-2],
            'R2': [1, 3],
            'C2': [-6,-2],
            'R3': [1, 3],
            'C3': [-6,-2],
            'R4': [1, 3],
            'C4': [-6,-2],
            'R5': [1, 3],
            'C5': [-6,-2],
            'R6': [1, 3],
            'C6': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model203':
        circuit = 'p(T1,R1,C1)-p(Gs1,Zarc1)-TLMQ1-L1-CPE1-p(R2,C2)'
        param_space = {
            'T1': [1, 3],
            'T2': [2, 4],
            'T3': [0, 1.7],
            'T4': [2.30,2.30],
            'R1': [1, 3],
            'C1': [-6,-2],
            'Gs1': [1, 3],
            'Gs2': [0, 2],
            'Gs3': [-1, 0],
            'Zarc1': [1, 3],
            'Zarc2': [1, 3],
            'Zarc3': [-1, 0.176],
            'TLMQ1': [1, 3],
            'TLMQ2': [1, 3],
            'TLMQ3': [-1, 0],
            'L1': [-4, -2],
            'CPE1': [-0.39,0.3],
            'CPE2': [-0.3,0],
            'R2': [1, 3],
            'C2': [-6,-2],
            'f' : [-4, 5]
        }
    elif model == 'Model204':
        circuit = 'R0-p(R1,C1)-Wo1-L1-Gs1-T1-CPE1-TLMQ1-W1-Zarc1'
        param_space = {
            'R0': [1, 3],#
            'R1': [1, 3],#
            'C1': [-6,-2],#
            'Wo1': [1,3],#
            'Wo2': [2.7,3.7],#
            'L1': [-4,-2],#
            'Gs1': [1,3],#
            'Gs2': [0,2],#
            'Gs3': [-1,0],#
            'T1': [1,3],#
            'T2' : [2,4],#
            'T3' : [0,1.7],#
            'T4' : [2.3,2.3],
            'CPE1' : [-0.39,0.3],#
            'CPE2' : [-0.3,0],#
            'TLMQ1' : [1,3],#
            'TLMQ2' : [1,3],#
            'TLMQ3' : [-1,0],#
            'W1' : [0,1.7],#
            'Zarc1' : [2,2],
            'Zarc2' : [2,2],
            'Zarc3' : [-1,0.176],#
            'f' : [-4, 5]
        }
    return circuit, param_space

# %%
#add noise to the data
def add_noise_to_data(x, data, my_model):
    # Set seed
    global seed, case, noise, path
    np.random.seed(seed)
    
    if case == 1:
        range_data_path = f'{path}/{my_model}/LHD_data_200000.csv'
        while not os.path.exists(range_data_path):
            time.sleep(1)
        range_data = pd.read_csv(range_data_path)
        # Range of last column
        range_data = range_data.iloc[:, -1]
        # Case 1: Total x% uncertainty, uniform distribution
        # Take the 95% quantile of the data as the value range
        valuerange = range_data.max() - range_data.min()
        n = data.size
        noise_dis = valuerange * np.random.random(size=n) * x
        direction = np.random.randint(0, 2, n) * 2 - 1
        noise_dis = noise_dis * direction
        data_with_noise = data.flatten() + noise_dis
        data_with_noise = data_with_noise.reshape(data.shape)

    elif case == 2:
        # Case 2: Value dependent x% uncertainty, uniform distribution
        random_array = data.flatten()
        noise_dis = random_array * x * np.random.random(size=len(random_array))
        direction = np.random.randint(0, 2, len(random_array)) * 2 - 1
        noise_dis = noise_dis * direction
        data_with_noise = random_array + noise_dis
        data_with_noise = data_with_noise.reshape(data.shape)

    elif case == 3:
        # Case 3: Normal distribution noise
        range_data_path = f'{path}/{my_model}/LHD_data_200000.csv'
        while not os.path.exists(range_data_path):
            time.sleep(1)
        range_data = pd.read_csv(range_data_path)
        # Range of last column
        range_data = range_data.iloc[:, -1]
        # Case 1: Total x% uncertainty, uniform distribution
        # Take the 95% quantile of the data as the value range
        valuerange = range_data.max() - range_data.min()
        random_array = data.flatten()
        noise_dis = random_array + np.random.normal(0, x * valuerange, len(random_array))
        data_with_noise = noise_dis.reshape(data.shape)

    elif case == 4:
        # Case 4: Case 2 + Case 3
        range_data_path = f'{path}/{my_model}/LHD_data_200000.csv'
        while not os.path.exists(range_data_path):
            time.sleep(1)
        range_data = pd.read_csv(range_data_path)
        # Range of last column
        range_data = range_data.iloc[:, -1]
        # Case 1: Total x% uncertainty, uniform distribution
        # Take the 95% quantile of the data as the value range
        valuerange = range_data.max() - range_data.min()
        random_array = data.flatten()
        noise_dis = random_array * x * np.random.random(size=len(random_array))
        direction = np.random.randint(0, 2, len(random_array)) * 2 - 1
        noise_dis = noise_dis * direction
        noise_dis = noise_dis + np.random.normal(0, x * valuerange, len(random_array))
        data_with_noise = random_array + noise_dis
        data_with_noise = data_with_noise.reshape(data.shape)

    else:
        raise ValueError("Invalid case number. Choose 1, 2, 3 or 4.")

    return data_with_noise
# %%
"""
# Space filling
"""

# %%
def LHD_data_generation(my_model, ratio = []):
    global seed, noise, path
    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility
    print("Generating LHD data for",my_model)
    circuit, param_space =choose_model(my_model)
    if 'T3' in param_space:
        new_column_value = [param_space["T3"][0]] # T3 is always a constant
        del param_space['T3']
        print(param_space)
        last_digit_removed = re.sub(r'\d(?=[^\d]*$)', '', my_model)
        # 找出剩下的所有数字
        fifth_number = int(re.findall(r'\d+', last_digit_removed)[0])

        data_amounts = [x * fifth_number for x in ratio]
        n=0
        for data_amount in data_amounts:
            doe = build.space_filling_lhs(param_space, data_amount)
            columns = list(param_space.keys())
            tmp = pd.DataFrame(doe, columns=columns)

            # 在T2后面插入新列T3
            tmp.insert(loc=tmp.columns.get_loc('T2') + 1, column='T3', value=new_column_value*tmp.shape[0])    

            for i in tmp.columns:#transform the data into log
                tmp[i] = 10**tmp[i]

            frequency = tmp["f"]
            tmp = tmp.drop(columns=["f"])
            for i in range(len(tmp)):
                from impedance.models.circuits import  CustomCircuit #####################
                CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
                Z = CustomCircuit.predict(np.array([frequency[i]]))
                Z_mod = np.abs(Z)
                if i == 0:
                    data = Z_mod
                else:
                    data = np.vstack((data, Z_mod))
            ############################## add noise to the data ################################
            data = add_noise_to_data(noise, data, my_model)
            tmp['f'] = frequency
            tmp['Z_mod'] = data
            ############################## save the data ################################
            os.makedirs(f'{path}/{my_model}', exist_ok=True)
            tmp.to_csv(f'{path}/{my_model}/LHD_data_{data_amount}.csv', index=False)
            n=n+1
            print("dataset",n," is online")
    else:
        last_digit_removed = re.sub(r'\d(?=[^\d]*$)', '', my_model)

        # 找出剩下的所有数字
        fifth_number = int(re.findall(r'\d+', last_digit_removed)[0])

        data_amounts = [x * fifth_number for x in ratio]
        n=0
        for data_amount in data_amounts:
            doe = build.space_filling_lhs(param_space, data_amount)
            columns = list(param_space.keys())
            tmp = pd.DataFrame(doe, columns=columns)

            for i in tmp.columns:#transform the data into log
                tmp[i] = 10**tmp[i]

            frequency = tmp["f"]
            tmp = tmp.drop(columns=["f"])
            for i in range(len(tmp)):
                from impedance.models.circuits import  CustomCircuit #####################
                CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
                Z = CustomCircuit.predict(np.array([frequency[i]]))
                Z_mod = np.abs(Z)
                if i == 0:
                    data = Z_mod
                else:
                    data = np.vstack((data, Z_mod))
            ############################## add noise to the data ################################
            data = add_noise_to_data(noise, data, my_model)
            tmp['f'] = frequency
            tmp['Z_mod'] = data
            ############################## save the data ################################
            os.makedirs(f'{path}/{my_model}', exist_ok=True)
            tmp.to_csv(f'{path}/{my_model}/LHD_data_{data_amount}.csv', index=False)
            n=n+1
            print("dataset",n," is online")
# %%
def LHD_data_replication(my_model, ratio = []):
    global seed, noise, path
    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility
    print("Generating LHD_data_replication data for",my_model)
    replications = [1,3,7]
    for Ra in ratio:
        circuit, param_space =choose_model(my_model)
        last_digit_removed = re.sub(r'\d(?=[^\d]*$)', '', my_model)
        # 找出剩下的所有数字
        devision = int(re.findall(r'\d+', last_digit_removed)[0])
        data_amounts = devision * Ra
        for Re in replications:
            n = int(devision * Ra / (Re+1))
            doe = build.space_filling_lhs(param_space, n)
            # Re times the data
            doe = np.vstack([doe]*(Re+1))
            columns = list(param_space.keys())
            tmp = pd.DataFrame(doe, columns=columns)
            for i in tmp.columns:
                tmp[i] = 10**tmp[i]
            frequency = tmp["f"]
            tmp = tmp.drop(columns=["f"])
            for i in range(len(tmp)):
                from impedance.models.circuits import  CustomCircuit
                CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
                Z = CustomCircuit.predict(np.array([frequency[i]]))
                Z_mod = np.abs(Z)
                if i == 0:
                    data = Z_mod
                else:
                    data = np.vstack((data, Z_mod))
            ############################## add noise to the data ################################
            data = add_noise_to_data(noise, data, my_model)
            tmp['f'] = frequency
            tmp['Z_mod'] = data
            ############################## save the data ################################
            os.makedirs(f'{path}/{my_model}', exist_ok=True)
            tmp.to_csv(f'{path}/{my_model}/LHD{Re}replication_data_{data_amounts}.csv', index=False)
            print("dataset",Re,"data replication is online")
# %%
def LHD_mean_replication(my_model, ratio = []):
    global seed, noise, path
    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility
    print("Generating LHD_mean_replication data for",my_model)
    replications = [1,3,7]
    for Ra in ratio:
        circuit, param_space =choose_model(my_model)
        last_digit_removed = re.sub(r'\d(?=[^\d]*$)', '', my_model)
        # 找出剩下的所有数字
        devision = int(re.findall(r'\d+', last_digit_removed)[0])
        data_amounts = devision * Ra
        for Re in replications:
            n = int(devision * Ra / (Re+1))
            doe = build.space_filling_lhs(param_space, n)
            # Re times the data
            doe = np.vstack([doe]*(Re+1))
            columns = list(param_space.keys())
            tmp = pd.DataFrame(doe, columns=columns)
            for i in tmp.columns:
                tmp[i] = 10**tmp[i]
            frequency = tmp["f"]
            tmp = tmp.drop(columns=["f"])
            for i in range(len(tmp)):
                from impedance.models.circuits import  CustomCircuit
                CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
                Z = CustomCircuit.predict(np.array([frequency[i]]))
                Z_mod = np.abs(Z)
                if i == 0:
                    data = Z_mod
                else:
                    data = np.vstack((data, Z_mod))
            ############################## add noise to the data ################################
            data = add_noise_to_data(noise, data, my_model)
            tmp['f'] = frequency
            tmp['Z_mod'] = data
            ############################## combine the data with same input ################################
            columns = tmp.columns
            last_column = columns[-1]
            other_columns = columns[:-1]
            tmp = tmp.groupby(list(other_columns), as_index=False)[last_column].mean()
            ############################## save the data ################################
            os.makedirs(f'{path}/{my_model}', exist_ok=True)
            tmp.to_csv(f'{path}/{my_model}/LHD{Re}meanreplication_data_{data_amounts}.csv', index=False)
            print("dataset",Re,"mean replication is online")

# %%
def LHD_initial_data_generation(my_model):
################################# generate initial datasets for active learning #################################
    global path, noise
    circuit, param_space =choose_model(my_model)
    if 'T3' in param_space:
        new_column_value = [param_space["T3"][0]] # T3 is always a constant
        del param_space['T3']
        print(param_space)

        pool_size=200000 # corresponding data volumns

        doe = build.space_filling_lhs(param_space, pool_size)
        columns = list(param_space.keys())
        tmp = pd.DataFrame(doe, columns=columns)
        tmp.insert(loc=tmp.columns.get_loc('T2') + 1, column='T3', value=new_column_value*tmp.shape[0])    

        for i in tmp.columns:#transform the data into log
            tmp[i] = 10**tmp[i]

        frequency = tmp["f"]
        tmp = tmp.drop(columns=["f"])
        for i in range(len(tmp)):
            from impedance.models.circuits import  CustomCircuit #####################
            CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
            Z = CustomCircuit.predict(np.array([frequency[i]]))
            Z_mod = np.abs(Z)
            if i == 0:
                data = Z_mod
            else:
                data = np.vstack((data, Z_mod))
        tmp['f'] = frequency
        tmp['Z_mod'] = data
        ############################## save the data ################################
        os.makedirs(f'{path}/{my_model}', exist_ok=True)
        tmp.to_csv(f'{path}/{my_model}/LHD_data_{pool_size}.csv', index=False)
        print("dataset"," is online")
    else:
        pool_size=200000 # corresponding data volumns


        doe = build.space_filling_lhs(param_space, pool_size)
        columns = list(param_space.keys())
        tmp = pd.DataFrame(doe, columns=columns)
        for i in tmp.columns:#transform the data into log
            tmp[i] = 10**tmp[i]

        frequency = tmp["f"]
        tmp = tmp.drop(columns=["f"])
        for i in range(len(tmp)):
            from impedance.models.circuits import  CustomCircuit #####################
            CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
            Z = CustomCircuit.predict(np.array([frequency[i]]))
            Z_mod = np.abs(Z)
            if i == 0:
                data = Z_mod
            else:
                data = np.vstack((data, Z_mod))
        tmp['f'] = frequency
        tmp['Z_mod'] = data
        ############################## save the data ################################
        os.makedirs(f'{path}/{my_model}', exist_ok=True)
        tmp.to_csv(f'{path}/{my_model}/LHD_data_{pool_size}.csv', index=False)
        print("dataset"," is online")
        
    ################################################# generate initial datasets for testing #################################################
    circuit, param_space =choose_model(my_model)

    if 'T3' in param_space:
        new_column_value = [param_space["T3"][0]] # T3 is always a constant
        del param_space['T3']
        print(param_space)

        pool_size=100000 # corresponding data volumns

        doe = build.space_filling_lhs(param_space, pool_size)
        columns = list(param_space.keys())
        tmp = pd.DataFrame(doe, columns=columns)
        tmp.insert(loc=tmp.columns.get_loc('T2') + 1, column='T3', value=new_column_value*tmp.shape[0])    

        for i in tmp.columns:#transform the data into log
            tmp[i] = 10**tmp[i]

        frequency = tmp["f"]
        tmp = tmp.drop(columns=["f"])
        for i in range(len(tmp)):
            from impedance.models.circuits import  CustomCircuit #####################
            CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
            Z = CustomCircuit.predict(np.array([frequency[i]]))
            Z_mod = np.abs(Z)
            if i == 0:
                data = Z_mod
            else:
                data = np.vstack((data, Z_mod))
        ############################## add noise to the data ################################
        data = add_noise_to_data(noise, data, my_model)
        tmp['f'] = frequency
        tmp['Z_mod'] = data
        os.makedirs(f'{path}/{my_model}', exist_ok=True)
        tmp.to_csv(f'{path}/{my_model}/LHD_data_{pool_size}.csv', index=False)
        print("dataset"," is online")
    else:
        pool_size=100000 # corresponding data volumns
        doe = build.space_filling_lhs(param_space, pool_size)
        columns = list(param_space.keys())
        tmp = pd.DataFrame(doe, columns=columns)
        for i in tmp.columns:#transform the data into log
            tmp[i] = 10**tmp[i]

        frequency = tmp["f"]
        tmp = tmp.drop(columns=["f"])
        for i in range(len(tmp)):
            from impedance.models.circuits import  CustomCircuit #####################
            CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
            Z = CustomCircuit.predict(np.array([frequency[i]]))
            Z_mod = np.abs(Z)
            if i == 0:
                data = Z_mod
            else:
                data = np.vstack((data, Z_mod))
        ############################## add noise to the data ################################
        data = add_noise_to_data(noise, data, my_model)
        tmp['f'] = frequency
        tmp['Z_mod'] = data
        os.makedirs(f'{path}/{my_model}', exist_ok=True)
        tmp.to_csv(f'{path}/{my_model}/LHD_data_{pool_size}.csv', index=False)
        print("dataset"," is online")
# %%
"""
# BBD and CCD
"""
# %%
def d_optimal_design(X, m):
    """
    Select a D-optimal subset of m points from the design matrix X.
    Parameters:
    X (numpy.ndarray): The design matrix from which to select points.
    m (int): The number of points to select for the D-optimal design.
    Returns:
    numpy.ndarray: The selected D-optimal design points.
    """
    global seed
    if seed is not None:
        np.random.seed(seed)  # Set the seed if provided
    n, k = X.shape
    start_time = time.time()
    time_limit = 120
    # Initial random selection of m points
    selected_indices = np.random.choice(n, m, replace=False)
    X_selected = X[selected_indices]
    # Compute the initial determinant of the information matrix
    M_selected = X_selected.T @ X_selected
    best_det = det(M_selected)
    improved = True
    while improved:
        improved = False
        for i in range(m):
            for j in range(n):
                if j not in selected_indices:
                    # Create a new candidate set by swapping points
                    candidate_indices = selected_indices.copy()
                    candidate_indices[i] = j
                    X_candidate = X[candidate_indices]
                    
                    # Compute the determinant of the new information matrix
                    M_candidate = X_candidate.T @ X_candidate
                    candidate_det = det(M_candidate)
                    
                    # If the candidate determinant is better, update the selection
                    if candidate_det > best_det and time.time() - start_time < time_limit:
                        selected_indices = candidate_indices
                        best_det = candidate_det
                        improved = True
                        break
            if improved:
                break
    return X[selected_indices]

def BBD_CCD_data_generation(my_model, ratio = []):
    global seed, noise, path
    print("now is:",my_model, 'BBD/CCD generating')
    circuit, param_space =choose_model(my_model)
    t3 = 0

    if 'T3' in param_space:
        new_column_value = [param_space["T3"][0]] # T3 is always a constant
        del param_space['T3']
        t3 = 1
        # print(param_space)
    n=1

    doe = build.central_composite(param_space,face='cci')
    columns = list(param_space.keys())
    tmp = pd.DataFrame(doe, columns=columns)
    if t3 == 1:
        tmp.insert(loc=tmp.columns.get_loc('T2') + 1, column='T3', value=new_column_value*tmp.shape[0]) 

    for i in tmp.columns:#transform the data into log
        tmp[i] = 10**tmp[i]

    frequency = tmp["f"]
    tmp = tmp.drop(columns=["f"])
    for i in range(len(tmp)):
        from impedance.models.circuits import  CustomCircuit #####################
        CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
        Z = CustomCircuit.predict(np.array([frequency[i]]))
        Z_mod = np.abs(Z)
        if i == 0:
            data = Z_mod
        else:
            data = np.vstack((data, Z_mod))
    ############################## add noise to the data ################################
    data = add_noise_to_data(noise, data, my_model)
    tmp['f'] = frequency
    tmp['Z_mod'] = data
    ############################## save the data ################################
    os.makedirs(f'{path}/{my_model}', exist_ok=True)
    tmp.to_csv(f'{path}/{my_model}/CCD_data_{len(tmp)}.csv', index=False)
    print("dataset",n," is online")
    ########################### D-optimal design data generation ########################################
    data = pd.read_csv(f'{path}/{my_model}/CCD_data_{len(tmp)}.csv')
    columns = list(data.keys())
    # ratio = [3, 4 ,5, 6, 8, 10, 15, 20, 50]
    num = re.search(r'\d+$', my_model).group()
    dimension = re.sub(r'\d(?=[^\d]*$)', '', num)
    for i in ratio:
        data_amount = int(i) * int(dimension)
        if data_amount > len(tmp):
            break
        data = pd.read_csv(f'{path}/{my_model}/CCD_data_{len(tmp)}.csv')
        data = d_optimal_design(data.values, data_amount)
        data = pd.DataFrame(data)
        data.columns = columns
        data.to_csv(f'{path}/{my_model}/CCD_data_{data_amount}.csv', index=False)

    circuit, param_space =choose_model(my_model)
    t3 = 0
    if 'T3' in param_space:
        new_column_value = [param_space["T3"][0]] # T3 is always a constant
        del param_space['T3']
        t3 = 1
    n=1
    doe = build.box_behnken(param_space)
    columns = list(param_space.keys())
    tmp = pd.DataFrame(doe, columns=columns)
    if t3 == 1:
        tmp.insert(loc=tmp.columns.get_loc('T2') + 1, column='T3', value=new_column_value*tmp.shape[0])
    for i in tmp.columns:#transform the data into log
        tmp[i] = 10**tmp[i]
    frequency = tmp["f"]
    tmp = tmp.drop(columns=["f"])
    for i in range(len(tmp)):
        from impedance.models.circuits import  CustomCircuit #####################
        float_0 = tmp.iloc[i].values.astype(float) # float32 to float
        CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=float_0) 
        Z = CustomCircuit.predict(np.array([frequency[i]]))
        Z_mod = np.abs(Z)
        if i == 0:
            data = Z_mod
        else:
            data = np.vstack((data, Z_mod))
    ############################## add noise to the data ################################
    data = add_noise_to_data(noise, data, my_model)
    tmp['f'] = frequency
    tmp['Z_mod'] = data
    ############################## save the data ################################
    os.makedirs(f'{path}/{my_model}', exist_ok=True)
    tmp.to_csv(f'{path}/{my_model}/BBD_data_{len(tmp)}.csv', index=False)
    ########################### D-optimal design data generation ########################################
    data = pd.read_csv(f'{path}/{my_model}/BBD_data_{len(tmp)}.csv')
    columns = list(data.keys())
    # ratio = [3, 4 ,5, 6, 8, 10, 15, 20, 50]
    num = re.search(r'\d+$', my_model).group()
    dimension = re.sub(r'\d(?=[^\d]*$)', '', num)
    for i in ratio:
        data_amount = int(i) * int(dimension)
        if data_amount > len(tmp):
            break
        data = pd.read_csv(f'{path}/{my_model}/BBD_data_{len(tmp)}.csv')
        data = d_optimal_design(data.values, data_amount)
        data = pd.DataFrame(data)
        data.columns = columns
        data.to_csv(f'{path}/{my_model}/BBD_data_{data_amount}.csv', index=False)
    print("dataset",n," is online")
# %%
"""
# Active Learning
"""
# %%
def AL_initial_data_generation(my_model, ratio = []):
    circuit, param_space =choose_model(my_model)
    global seed, noise, num_AL_initial, path
    if seed is not None:
        np.random.seed(seed)
    if 'T3' in param_space:
        new_column_value = [param_space["T3"][0]] # T3 is always a constant
        del param_space['T3']
        print(param_space)
        data_amounts = [num_AL_initial]
        n=0
        for data_amount in data_amounts:
            data_amount = int(data_amount)##############
            doe = build.space_filling_lhs(param_space, data_amount)
            columns = list(param_space.keys())
            tmp = pd.DataFrame(doe, columns=columns)

            # 在T2后面插入新列T3
            tmp.insert(loc=tmp.columns.get_loc('T2') + 1, column='T3', value=new_column_value*tmp.shape[0])    

            for i in tmp.columns:#transform the data into log
                tmp[i] = 10**tmp[i]

            frequency = tmp["f"]
            tmp = tmp.drop(columns=["f"])
            for i in range(len(tmp)):
                from impedance.models.circuits import  CustomCircuit #####################
                CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
                Z = CustomCircuit.predict(np.array([frequency[i]]))
                Z_mod = np.abs(Z)
                if i == 0:
                    data = Z_mod
                else:
                    data = np.vstack((data, Z_mod))
            ############################## add noise to the data ################################
            data = add_noise_to_data(noise, data, my_model)
            tmp['f'] = frequency
            tmp['Z_mod'] = data
            ############################## save the data ################################s
            os.makedirs(f'{path}/{my_model}/AL_initial', exist_ok=True)
            tmp.to_csv(f'{path}/{my_model}/AL_initial/LHD_data_{data_amount}.csv', index=False)
            n=n+1
            print("dataset",n," is online")
    else:
        data_amounts = [num_AL_initial]
        n=0
        for data_amount in data_amounts:
            data_amount = int(data_amount)##############
            doe = build.space_filling_lhs(param_space, data_amount)
            columns = list(param_space.keys())
            tmp = pd.DataFrame(doe, columns=columns)
            for i in tmp.columns:#transform the data into log
                tmp[i] = 10**tmp[i]
            frequency = tmp["f"]
            tmp = tmp.drop(columns=["f"])
            for i in range(len(tmp)):
                from impedance.models.circuits import  CustomCircuit #####################
                CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=tmp.iloc[i].values)
                Z = CustomCircuit.predict(np.array([frequency[i]]))
                Z_mod = np.abs(Z)
                if i == 0:
                    data = Z_mod
                else:
                    data = np.vstack((data, Z_mod))
            ############################## add noise to the data ################################
            data = add_noise_to_data(noise, data, my_model)
            tmp['f'] = frequency
            tmp['Z_mod'] = data
            ############################## save the data ################################
            os.makedirs(f'{path}/{my_model}/AL_initial', exist_ok=True)
            tmp.to_csv(f'{path}/{my_model}/AL_initial/LHD_data_{data_amount}.csv', index=False)
            n=n+1
            print("dataset",n," is online")
    #end of the function
    return
# %%
def EIS_function(X_new,my_model,myscaler,mydf):
    global seed, noise
    if seed is not None:
        np.random.seed(seed)
    df_X_new = pd.DataFrame([X_new], columns=mydf.columns)
    X_new_inverse=df_X_new*(mydf.max() - mydf.min())+mydf.min()
    circuit, _ =choose_model(my_model)
    frequency = X_new_inverse.values[:, -1]
    X_new_inverse_without_frequency = X_new_inverse.values[:, :-1]
    from impedance.models.circuits import  CustomCircuit #####################
    CustomCircuit = CustomCircuit(circuit=circuit, initial_guess=X_new_inverse_without_frequency.flatten()) 
    Z = CustomCircuit.predict(frequency)
    # add noise to the data
    Z = np.abs(Z)
    Z = add_noise_to_data(noise, Z, my_model)
    return np.abs(Z)
# %%
"""
# Emukit
"""
# %%
def Emukit_us_data_generation(my_model, ratio = []):
    global seed, noise, num_AL_initial, num_AL , path
    if seed is not None:
        np.random.seed(seed)
    print("now is:",my_model, 'Emu_us generating')
    circuit, param_space =choose_model(my_model)
    last_digit_removed = re.sub(r'\d(?=[^\d]*$)', '', my_model)
    DD = int(re.findall(r'\d+', last_digit_removed)[0])
    if 'T3' in param_space:
        dimension = DD+1
    else:
        dimension = DD
    ########################################################################
    directory_path = f'{path}/{my_model}/AL_initial/'
    #if cant find the directory, wait 5 seconds
    while not os.path.exists(directory_path):
        time.sleep(5)
    # for filename in os.listdir(directory_path):
    for rat in ratio:
        iterations = num_AL
        # print(filename)
        # iterations = int(re.findall(r'\d+', filename)[0])
        ini_file = directory_path + f'LHD_data_{num_AL_initial}.csv'
        while not os.path.exists(ini_file):
            time.sleep(5)
        df_initial = pd.read_csv(ini_file)
        Y_init=df_initial["Z_mod"].values
        df_initial = df_initial.drop(columns=["Z_mod"])
        # normalization without 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized = pd.DataFrame(scaler.fit_transform(df_initial), columns=df_initial.columns)
        # df_normalized = (df_initial - df_initial.min()) / (df_initial.max() - df_initial.min())
        X_init=df_normalized.values
        emukit_pslist=[]
        for col in df_normalized.columns:
            emukit_pslist.append(ContinuousParameter(col,df_normalized[col].min(),df_normalized[col].max()))
        ParameterSpace_emu=ParameterSpace(emukit_pslist)
        # ker = GPy.kern.Bias(input_dim=dimension) + GPy.kern.Bias(1.0) * GPy.kern.RBF(input_dim=dimension, variance=1., lengthscale=1.,ARD=True) + GPy.kern.White(1) # R2=0.87
        ker = GPy.kern.Bias(input_dim=dimension) + GPy.kern.Bias(1.0) * GPy.kern.RBF(input_dim=dimension, variance=1., lengthscale=1.,ARD=True)+ GPy.kern.Bias(1.0) *GPy.kern.Matern32(input_dim=dimension, variance=1., lengthscale=1.)
        for iters in range(iterations):
            # 生成GP高斯模型
            gpy_model = GPy.models.GPRegression(X_init, Y_init[:, None], ker)
            gpy_model.optimize(max_f_eval=10000)
            # y_pre = gpy_model.predict_quantiles(X_init)[0]
            # print('GP model(training), R2=%.2f' % r2_score(Y_init, y_pre))
            # 生成emukit模型 # 迭代
            emukit_model = GPyModelWrapper(gpy_model)
            us_acquisition = ModelVariance(emukit_model)
            # ivr_acquisition = IntegratedVarianceReduction(emukit_model, ParameterSpace_emu)
            optimizer = GradientAcquisitionOptimizer(ParameterSpace_emu)
            x_new, _ = optimizer.optimize(us_acquisition)
        #     print("new data point according to emukit:",x_new)
            y_new = EIS_function(x_new.flatten(),my_model,myscaler=scaler,mydf=df_initial)
        #     print("new target value from EIS:",y_new)
        #     df_x_new = pd.DataFrame([x_new.flatten()], columns=df_initial.columns)
            X_init = np.append(X_init, x_new, axis=0)
            Y_init = np.append(Y_init, y_new, axis=0) ########## y_new[0]
        # print("new dataset:",X_init.shape,Y_init.shape)
        df_X_init = pd.DataFrame(X_init, columns=df_initial.columns)
        df_emukit=df_X_init*(df_initial.max() - df_initial.min())+df_initial.min()
        df_emukit['Z_mod']=Y_init
        df_emukit.to_csv(f'{path}/{my_model}/Emu_us_data_{iterations+df_initial.shape[0]}.csv', index=False)
# %%
def Emukit_ivr_data_generation(my_model, ratio = []):
    global seed, noise, num_AL_initial, num_AL, path
    if seed is not None:
        np.random.seed(seed)
    print("now is:",my_model, 'Emu_ivr generating')
    circuit, param_space =choose_model(my_model)
    last_digit_removed = re.sub(r'\d(?=[^\d]*$)', '', my_model)
    DD = int(re.findall(r'\d+', last_digit_removed)[0])
    if 'T3' in param_space:
        dimension = DD + 1
    else:
        dimension = DD
    # my_param_space = {
    #     'R0': [1, 3],
    #     'R1': [1, 3],
    #     'C1': [-6,-2],
    #     'f' : [-4, 5]
    # }
    # my_param_space_lg = {key: [10**val for val in value] for key, value in my_param_space.items()}
    # print(my_param_space_lg)
    ########################################################################
    directory_path = f'{path}/{my_model}/AL_initial/'
    #if cant find the directory, wait 5 seconds
    while not os.path.exists(directory_path):
        time.sleep(5)
    # for filename in os.listdir(directory_path):
    #     print(filename)
    for rat in ratio:
        iterations = num_AL
        ini_file = directory_path + f'LHD_data_{num_AL_initial}.csv'
        while not os.path.exists(ini_file):
            time.sleep(5)
        # iterations = int(re.findall(r'\d+', filename)[0])
        df_initial = pd.read_csv(ini_file)
        Y_init=df_initial["Z_mod"].values
        df_initial = df_initial.drop(columns=["Z_mod"])
        # normalization without 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized = pd.DataFrame(scaler.fit_transform(df_initial), columns=df_initial.columns)
        # df_normalized = (df_initial - df_initial.min()) / (df_initial.max() - df_initial.min())
        X_init=df_normalized.values
        emukit_pslist=[]
        for col in df_normalized.columns:
            emukit_pslist.append(ContinuousParameter(col,df_normalized[col].min(),df_normalized[col].max()))
        ParameterSpace_emu=ParameterSpace(emukit_pslist)
        # ker = GPy.kern.Bias(input_dim=dimension) + GPy.kern.Bias(1.0) * GPy.kern.RBF(input_dim=dimension, variance=1., lengthscale=1.,ARD=True) + GPy.kern.White(1) # R2=0.87
        ker = GPy.kern.Bias(input_dim=dimension) + GPy.kern.Bias(1.0) * GPy.kern.RBF(input_dim=dimension, variance=1., lengthscale=1.,ARD=True)+ GPy.kern.Bias(1.0) *GPy.kern.Matern32(input_dim=dimension, variance=1., lengthscale=1.)
        for iters in range(iterations):
            # 生成GP高斯模型
            gpy_model = GPy.models.GPRegression(X_init, Y_init[:, None], ker)
            gpy_model.optimize(max_f_eval=10000)
            # y_pre = gpy_model.predict_quantiles(X_init)[0]
            # print('GP model(training), R2=%.2f' % r2_score(Y_init, y_pre))
            # 生成emukit模型 # 迭代
            emukit_model = GPyModelWrapper(gpy_model)
    #       us_acquisition = ModelVariance(emukit_model)
            ivr_acquisition = IntegratedVarianceReduction(emukit_model, ParameterSpace_emu)
            optimizer = GradientAcquisitionOptimizer(ParameterSpace_emu)
            x_new, _ = optimizer.optimize(ivr_acquisition)
        #     print("new data point according to emukit:",x_new)
            y_new = EIS_function(x_new.flatten(),my_model,myscaler=scaler,mydf=df_initial)
        #     print("new target value from EIS:",y_new)
        #     df_x_new = pd.DataFrame([x_new.flatten()], columns=df_initial.columns)
            X_init = np.append(X_init, x_new, axis=0)
            Y_init = np.append(Y_init, y_new, axis=0)
        # print("new dataset:",X_init.shape,Y_init.shape)
        df_X_init = pd.DataFrame(X_init, columns=df_initial.columns)
        df_emukit=df_X_init*(df_initial.max() - df_initial.min())+df_initial.min()
        df_emukit['Z_mod']=Y_init
        df_emukit.to_csv(f'{path}/{my_model}/Emu_ivr_data_{iterations+df_initial.shape[0]}.csv', index=False)
# %%
"""
# Modal
"""
# %%
def ModaL_data_generation(my_model,ratio = []):
    global seed, noise, num_AL_initial, num_AL, path
    if seed is not None:
        np.random.seed(seed)
    print("now is:",my_model, 'ModAL generating')
    # active learning sampling with modAL
    last_digit_removed = re.sub(r'\d(?=[^\d]*$)', '', my_model)
    # 找出剩下的所有数字
    dimention = int(re.findall(r'\d+', last_digit_removed)[0])
    # my_param_space = {
    #     'R0': [1, 3],
    #     'R1': [1, 3],
    #     'C1': [-6,-2],
    #     'f' : [-4, 5]
    # }
    # my_param_space_lg = {key: [10**val for val in value] for key, value in my_param_space.items()}
    # print(my_param_space_lg)
    ########################################################################
    directory_path = f'{path}/{my_model}/AL_initial/'
    #if cant find the directory, wait 5 seconds
    while not os.path.exists(directory_path):
        time.sleep(5)
    # for filename in os.listdir(directory_path):
        # print(filename)
        # iterations = int(re.findall(r'\d+', filename)[0]) ## iterations for AL,50% initial
    for rat in ratio:
        iterations = num_AL
        ini_file = directory_path + f'LHD_data_{num_AL_initial}.csv'
        while not os.path.exists(ini_file):
            time.sleep(5)
        df_initial = pd.read_csv(ini_file)
        pool_file = f'{path}/{my_model}/LHD_data_100000.csv'
        while not os.path.exists(pool_file):
            time.sleep(5)
        df_pool = pd.read_csv(pool_file)
        Y_init=df_initial["Z_mod"].values
        Y_pool=df_pool["Z_mod"].values
        df_initial = df_initial.drop(columns=["Z_mod"])
        df_pool = df_pool.drop(columns=["Z_mod"])
        # normalization without 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized = pd.DataFrame(scaler.fit_transform(df_initial), columns=df_initial.columns)
        df_pool_normalized = pd.DataFrame(scaler.fit_transform(df_pool), columns=df_pool.columns)
        # df_normalized = (df_initial - df_initial.min()) / (df_initial.max() - df_initial.min())
        X_init=df_normalized.values
        X_pool=df_pool_normalized.values
    ########################################################################################
        kernels = []
        Num_estimators = [50, 80, 200]
        samples_splits = [2,5,8]
        for n in Num_estimators:
            for s in samples_splits:
                kernels.append(RandomForestRegressor(n_estimators=n, min_samples_split=s))
                kernels.append(AdaBoostRegressor(n_estimators=n))
        kernels.append(linear_model.LinearRegression())
        kernels.append(linear_model.BayesianRidge())
        hidden_layer_sizes = [(8,24,12,6),(12,24,12,6),(16,24,12,6)]
        activation = ['relu']
        batch_size = [10 ,5, 2]
        for h in hidden_layer_sizes:
            for a in activation:
                for b in batch_size:
                    kernels.append(MLPRegressor(hidden_layer_sizes=h, activation=a, batch_size=b, max_iter=5000))
    ########################################################################################
        # initialize learner
        learner_list = []
        for kernel in kernels:
                learner = ActiveLearner(
                    estimator=kernel,
                    X_training=X_init,
                    y_training=Y_init
                )
                learner_list.append(learner)
        for iters in range(iterations):
            pre, std = CommitteeRegressor(learner_list).predict(X_pool, return_std=True)
            uncertainty = std
            query_idx = np.argmax(uncertainty)
            query_label = Y_pool[query_idx]
            ################################
            X_init = np.concatenate((X_init, X_pool[query_idx].reshape(1, -1)), axis=0)
            Y_init = np.concatenate((Y_init, query_label.reshape(1,)), axis=0)
            print('train size:', X_init.shape[0])
            print('pool size:', X_pool.shape[0])
            learner_list = []
            for kernel in kernels:
                learner = ActiveLearner(
                    estimator=kernel,
                    X_training=X_init,
                    y_training=Y_init
                )
                learner_list.append(learner)
            X_pool = np.delete(X_pool, query_idx, axis=0)
            Y_pool = np.delete(Y_pool, query_idx)
        # print("new dataset:",X_init.shape,Y_init.shape)
        df_X_init = pd.DataFrame(X_init, columns=df_initial.columns)
        df_modal=df_X_init*(df_initial.max() - df_initial.min())+df_initial.min()
        df_modal['Z_mod']=Y_init
        df_modal.to_csv(f'{path}/{my_model}/ModAL_data_{iterations+df_initial.shape[0]}.csv', index=False)


# %%
"""
# Baal
"""
# %%
def Baal_data_generation(my_model, ratio = []):
    global seed, noise, num_AL_initial, num_AL, path
    print("now is:",my_model, 'Baal generating')
    directory_path = f'{path}/{my_model}/AL_initial/'
    #if cant find the directory, wait 5 seconds
    while not os.path.exists(directory_path):
        time.sleep(5)
    last_digit_removed = re.sub(r'\d(?=[^\d]*$)', '', my_model)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # 找出剩下的所有数字
    dimension = int(re.findall(r'\d+', last_digit_removed)[0])
    DD = int(re.findall(r'\d+', last_digit_removed)[0])
    if my_model == 'Model43' or my_model == 'Model83' or my_model == 'Model162':
        dimension += 1
    # for filename in os.listdir(directory_path):
        # print(filename)
        # iterations = int(re.findall(r'\d+', filename)[0]) ## iterations for AL,50% initial
    for rat in ratio:
        iterations = num_AL
        ini_file = directory_path + f'LHD_data_{num_AL_initial}.csv'
        while not os.path.exists(ini_file):
            time.sleep(5)
        df_initial = pd.read_csv(ini_file)
        pool_file = f'{path}/{my_model}/LHD_data_100000.csv'
        while not os.path.exists(pool_file):
            time.sleep(5)
        df_pool = pd.read_csv(pool_file)
        Y_init=df_initial["Z_mod"].values
        Y_pool=df_pool["Z_mod"].values
        df_initial = df_initial.drop(columns=["Z_mod"])
        df_pool_X = df_pool.drop(columns=["Z_mod"])
        # normalization without 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized = pd.DataFrame(scaler.fit_transform(df_initial), columns=df_initial.columns)
        df_pool_X_normalized = pd.DataFrame(scaler.fit_transform(df_pool_X), columns=df_pool_X.columns)
        
        # df_normalized = (df_initial - df_initial.min()) / (df_initial.max() - df_initial.min())
        X_init=df_normalized.values
        X_pool=df_pool_X_normalized.values
        
        # # Read data
        # data = fetch_california_housing()
        # X, y = data.data, data.target
        
        # # train-test split for model evaluation
        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
        
        # Convert to 2D PyTorch tensors
        X_init = torch.tensor(X_init, dtype=torch.float32)
        Y_init = torch.tensor(Y_init, dtype=torch.float32).reshape(-1, 1)
        X_pool = torch.tensor(X_pool, dtype=torch.float32)
        Y_pool = torch.tensor(Y_pool, dtype=torch.float32).reshape(-1, 1)
        
        # Define the model
        model = nn.Sequential(
                nn.Linear(dimension, dimension*4),
                nn.ReLU(),
                nn.Linear(dimension*4, dimension*3),
                nn.ReLU(),
                nn.Linear(dimension*3, dimension*2),
                nn.ReLU(),
                nn.Linear(dimension*2, dimension),
                nn.ReLU(),
                nn.Linear(dimension, 1)
        )
        
        for iters in range(iterations):
            # loss function and optimizer
            loss_fn = nn.MSELoss()  # mean square error
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            
            n_epochs = 3000   # number of epochs to run
            batch_size = 30  # size of each batch
            batch_start = torch.arange(0, len(X_init), batch_size)
            
            # Hold the best model
            best_mse = np.inf   # init to infinity
            best_weights = None
            history = []
            for epoch in range(n_epochs):
                model.train()
                with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                    bar.set_description(f"Epoch {epoch}")
                    for start in bar:
                        # take a batch
                        X_batch = X_init[start:start+batch_size]
                        y_batch = Y_init[start:start+batch_size]
                        # forward pass
                        y_pred = model(X_batch)
                        loss = loss_fn(y_pred, y_batch)
                        # backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        # update weights
                        optimizer.step()
                        # print progress
                        bar.set_postfix(mse=float(loss))
                model.eval()
                y_pred = model(X_pool)
                mse = loss_fn(y_pred, Y_pool)
                mse = float(mse)
                history.append(mse)
                if mse < best_mse:
                    best_mse = mse
                    best_weights = copy.deepcopy(model.state_dict())
            # restore model and return best accuracy
            model.load_state_dict(best_weights)
            model = MCDropoutConnectModule(model, layers=['Linear'], weight_dropout=0.5)
            wrapped_model = ModelWrapper(model, torch.nn.MSELoss() , replicate_in_memory=False)
            with torch.no_grad():
                predictions = wrapped_model.predict_on_batch(X_pool, iterations=500)
            
            #
        
            reshaped_predictions = predictions.squeeze(1)
            
            # 计算每一行的标准差
            std_per_row = torch.std(reshaped_predictions, dim=1)
            
            max_index = torch.argmax(std_per_row)
            added_X=torch.unsqueeze(X_pool[max_index.item()], dim=0)
            added_Y=torch.unsqueeze(Y_pool[max_index.item()], dim=0)
            
            X_init=torch.cat((X_init, added_X), dim=0)
            Y_init=torch.cat((Y_init, added_Y), dim=0)
            # 要删除的行的索引
            row_index_to_remove = max_index.item()  
            X_pool = torch.cat((X_pool[:row_index_to_remove], X_pool[row_index_to_remove+1:]), dim=0)
            Y_pool = torch.cat((Y_pool[:row_index_to_remove], Y_pool[row_index_to_remove+1:]), dim=0)
            print('train size:', X_init.shape[0])
            print('pool size:', X_pool.shape[0])

        df_X_init = pd.DataFrame(X_init.numpy(), columns=df_initial.columns)
        df_modal=df_X_init*(df_initial.max() - df_initial.min())+df_initial.min()
        df_modal['Z_mod']=Y_init.numpy()
        df_modal.to_csv(f'{path}/{my_model}/Baal_data_{iterations+df_initial.shape[0]}.csv', index=False)

# %%
# # my_model_list = ['Model41','Model42','Model43','Model81','Model82','Model83','Model101','Model161','Model162']
# my_model_list = ['Model41'] #for test
# random_seed = 480
# ratio = [3]
# if __name__ == '__main__':
#     for my_model in my_model_list:
#         LHD_data_generation(my_model,random_seed,ratio)
#         BBD_CCD_data_generation(my_model,random_seed,ratio)
#         AL_initial_data_generation(my_model,random_seed,ratio)
#         Emukit_us_data_generation(my_model,random_seed,ratio)
#         ##################### Emukit_ivr_data_generation(my_model)
#         ModaL_data_generation(my_model,random_seed,ratio)
#         Baal_data_generation(my_model,random_seed,ratio)
# %%
"""
HPC code
"""
experiment = {
    'task_id': int(os.getenv('SLURM_ARRAY_TASK_ID')),
    # 'task_id': 0, #for test
    'experiment_date_time': datetime.now().strftime("%Y-%m-%d_%H%M%S")
}
# %%
def int_to_3d_coord(num, size_a, size_b, size_c):
    if num < 0 or num >= size_a * size_b * size_c:
        raise ValueError("Invalid input number. Number must be within the valid range.")
    z = num // (size_a * size_b)
    y = (num - z * size_a * size_b) // size_a
    x = num % size_a
    
    return x, y, z
# %%
#use the task_id to select the model and function
# my_model_list = ['Model41','Model42','Model43','Model81','Model82','Model83']
my_model_list = ['Model83']
# fun_list = [LHD_data_generation, AL_initial_data_generation, Emukit_us_data_generation, ModaL_data_generation, Baal_data_generation,BBD_CCD_data_generation]
fun_list = [LHD_data_generation, AL_initial_data_generation, Emukit_us_data_generation, ModaL_data_generation]
# fun_list = [AL_initial_data_generation, Emukit_us_data_generation, ModaL_data_generation]
# fun_list = [LHD_data_generation, LHD_data_replication]
ratio_list = [[150],[200],[250],[300],[350]]
# ratio_list = [[100], [200], [300], [400], [500], [1000]]
#set global variable noise
# seed = 10# for 1 repeat
noise = 0.15
case = 1
repeat = 10
if __name__ == '__main__':
    print(f'Running task {experiment["task_id"]}')
    sys.setrecursionlimit(1500)
    # id = experiment['task_id']# for 1 repeat
    ##############
    id = experiment['task_id'] // repeat # for 10 repeats
    repeat_time = experiment['task_id'] % repeat
    seed = 531 + repeat_time
    
    ##############
    path = os.getcwd() + f'/Database/Modelbase_{noise}_{repeat_time}/'
    model_id, fun_id, ratio_id = int_to_3d_coord(id, len(my_model_list), len(fun_list), len(ratio_list))
    last_digit_removed = re.sub(r'\d(?=[^\d]*$)', '', my_model_list[model_id])
    division = int(re.findall(r'\d+', last_digit_removed)[0])
    num_AL_initial = int(ratio_list[ratio_id][0] * division * 0.6)
    num_AL = ratio_list[ratio_id][0] * division - num_AL_initial
    my_model = my_model_list[model_id]
    fun = fun_list[fun_id]
    ratio = ratio_list[ratio_id]
    if id == 0:
        os.makedirs(path, exist_ok=True)
        #write the experiment information to a file
        with open(f'{path}/experiment_{experiment["experiment_date_time"]}.txt', 'w') as f:
            f.write(f'Experiment date and time: {experiment["experiment_date_time"]}\n')
            f.write(f'Noise level: {noise}\n')
            f.write(f'Random seed: {seed}\n')
            f.write(f'Noise Case: {case}\n')
        #LHD initial data generation
        for model in my_model_list:
            LHD_initial_data_generation(model)
            ############################## add 100k replication data ################################
            # last_digit_removed = re.sub(r'\d(?=[^\d]*$)', '', model)
            # division = int(re.findall(r'\d+', last_digit_removed)[0])
            # ratio_lhd = [20, 40, 50, 80, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200]
            # # LHD_data_replication(model,[ratio_4k])
            # # LHD_data_generation(model, ratio_lhd)
            # LHD_data_generation(model, ratio_lhd)
            # LHD_mean_replication(model, ratio_lhd)
            #########################################################################################
    print('done')
    fun(my_model, ratio)
