#!/usr/bin/env python
# coding: utf-8

import os
import sys
import autosklearn.regression
import pandas as pd
import sklearn.metrics
from datetime import datetime
import re


# #### measure resource consumption (start)
# # Get initial memory usage
# initial_memory = psutil.virtual_memory().used
# # Get initial CPU time
# initial_cpu_time = psutil.cpu_times().user + psutil.cpu_times().system

experiment = {
    'task_id': int(os.getenv('SLURM_ARRAY_TASK_ID')) ,
    # 'task_id': 0, #for test
    'experiment_date_time': datetime.now().strftime("%Y-%m-%d_%H%M%S")
}

#### load experiment setup
def load_experiment_csv():
    # Load the CSV file into a DataFrame
    Current_path = os.path.dirname(os.path.abspath(__file__))
    upper_path = os.getcwd()
    upper_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    df = pd.read_csv(upper_path + '/Experiments/experiment_setup.csv')

    # Get a row by its index
    row_index = int(experiment['task_id'])
    row = df.iloc[row_index]
    print(row)
    return row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5]#0 for dir ,1 for Model, 2 for Traindata, 3 for Test data, 4 for randomseed


#### main function
    
def run():
    # load simulation parameters from experiment list
    Repeat_dir,Model_dir, Traindata_name, Testdata_name,seed,_ = load_experiment_csv()
    # read last number in the Repeat_dir
    id = re.search(r'(\d+)$', Repeat_dir).group(1)
    seed = int(seed)
    upper_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # Database = Current_path + f'/Modelbase'
    #if dir only has one folder start with Modelbase, then use the folder as the database
    Database = 'Database' + '/' + Repeat_dir
    n = 0
    # find the number in name of folder
    middle_number = re.findall(r'_(\d+\.\d+|\d+)_', Repeat_dir)[0]
    noise = float(middle_number)
    path = Database + '/' + Model_dir

    #test data
    Testdata = pd.read_csv(path + '/' + Testdata_name)
    X_test = Testdata.drop(['Z_mod'], axis=1)
    y_test = Testdata['Z_mod']
    
    #train data
    Traindata = pd.read_csv(path + '/' + Traindata_name)
    X_train = Traindata.drop(['Z_mod'], axis=1)
    y_train = Traindata['Z_mod']
    n = Traindata.shape[0]
    #take the number out of the directory name
    model_input = Model_dir
    model_input = model_input.replace('Model','')
    #delete the last digit
    model_input = model_input[:-1]
    #convert the string to integer
    model_input = int(model_input)
    ratio = n/model_input
    #train model
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task = 200,
        per_run_time_limit = 30, # 20
        initial_configurations_via_metalearning=25,
        memory_limit=10240,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds':5},
        seed = seed,
        n_jobs=1
    )
    automl.fit(X_train, y_train)
    #predict
    y_hat = automl.predict(X_test)
    #save the R2 score
#     R2 = automl.score(X_test, y_test)  ########
    R2=sklearn.metrics.r2_score(y_test, y_hat)
    RMSE = sklearn.metrics.mean_squared_error(y_test, y_hat, squared=False)
#     print("Test R2 score:", )
    #save the log
    log = automl.sprint_statistics()


########################################## old version ########################################################
#     #create a new directory to save the results
#     result_path = path + '/AutoML'
#     if not os.path.exists(result_path):
#         os.makedirs(result_path)
#     #save the results

#     results = {
#         "Model_dir": Model_dir,
#         "Traindata_Name": Traindata_name,
#         "R2_score": R2,
# #         "Model_log": "string"
#     }
#     if os.path.exists(result_path+Traindata_name+".txt"):
#         os.remove(result_path+Traindata_name+".txt")
        
    # with open(result_path+Traindata_name+".txt", "a") as file:
    #     header = ",".join(results.keys()) + "\n"
    #     file.write(header)
    #     line = ",".join([str(value) for value in results.values()]) + "\n"
    #     # write in
    #     file.write(line)

################################################################# new output #################################################################
    #create a new directory to save the results
    result_path = upper_path + '/Experiments/' + f'/Results_{noise}_{seed}/' + Model_dir
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #save the results
    with open(result_path + '/' + f'R2_{noise}.csv', 'a') as f:
        #initialize the file
        if os.path.getsize(result_path + '/' + f'R2_{noise}.csv') == 0:
            f.write('file' + ',' + 'n' + ',' + 'ratio' + ',' + 'R2' + ',' + 'id' + '\n')
        f.write(Traindata_name + ','+  str(n) + ',' + str(ratio) + ',' + str(R2) + ',' + str(id) + '\n')
    with open(result_path + '/' + f'log_{noise}.txt', 'a') as f:
        f.write(Traindata_name + ' ' + log + '\n')
    #save the model
    result = automl.cv_results_
    result = pd.DataFrame(result)
    Modelpath = result_path + '/cv_results'
    if not os.path.exists(Modelpath):
        os.makedirs(Modelpath)
    result.to_csv(Modelpath + '/' + Traindata_name + '.csv')
    resultfolder = upper_path + '/Results'
    if not os.path.exists(resultfolder):
        os.makedirs(resultfolder)
    with open(resultfolder + f'/R2_{noise}_{seed}.csv' ,'a') as f:
        if os.path.getsize(resultfolder + f'/R2_{noise}_{seed}.csv') == 0:
            f.write('model' + ',' + 'file' + ',' + 'n' + ',' + 'ratio' + ',' + 'R2' + ',' + 'id' + '\n')
        f.write(Model_dir + ',' + Traindata_name + ',' + str(n) + ',' + str(ratio) + ',' + str(R2) + ',' + str(id) + '\n')
    with open(resultfolder + f'/RMSE_{noise}_{seed}.csv' ,'a') as f:
        if os.path.getsize(resultfolder + f'/RMSE_{noise}_{seed}.csv') == 0:
            f.write('model' + ',' + 'file' + ',' + 'n' + ',' + 'ratio' + ',' + 'RMSE' + ',' + 'id' + '\n')
        f.write(Model_dir + ',' + Traindata_name + ',' + str(n) + ',' + str(ratio) + ',' + str(RMSE) + ',' + str(id) + '\n')
    print('done!')

#########
if __name__ == '__main__':
    run()

#########
    
# yi.start()

# print("Start measurement...")
# this_job = jm.Job_metrics()
# this_job.start_measurement()

# TODO Wall-time (start, stop) output -> time series plotting of all tasks
# TODO histogram


# print("Stop measurement")
# this_job.stop_measurement()


# hours = this_job.wall_time['hours']
# minutes = this_job.wall_time['minutes']
# seconds = this_job.wall_time['seconds']

# print()
# print("[", experiment['slurm_task_id'], "]", "Simulation total time consumed:", hours, "h", minutes, "m", seconds, "s")

# print(this_job.cpu['avg_load'])
# print(this_job.cpu['freq'])
# print(this_job.cpu['temp'])
# print(this_job.time)

# yi.stop()
# yi.get_thread_stats().print_all()

# t = {'frequency': [
#         scpufreq(current=1529.4634296875001, min=1500.0, max=2250.0),  

# 'temperature': [{'nvme': [shwtemp(label='Composite', current=35.85, high=79.85, critical=86.85), 

#shwtemp(label='Sensor 1', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=45.85, high=65261.85, critical=65261.85), shwtemp(label='Composite', current=35.85, high=79.85, critical=86.85), shwtemp(label='Sensor 1', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=45.85, high=65261.85, critical=65261.85), shwtemp(label='Composite', current=35.85, high=79.85, critical=86.85), shwtemp(label='Sensor 1', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 3', current=34.85, high=65261.85, critical=65261.85)], 'i350bb': [shwtemp(label='loc1', current=43.0, high=120.0, critical=110.0)], 'k10temp': [shwtemp(label='Tctl', current=47.5, high=None, critical=None), shwtemp(label='Tccd8', current=39.5, high=None, critical=None), shwtemp(label='Tccd1', current=48.5, high=None, critical=None), shwtemp(label='Tccd2', current=41.5, high=None, critical=None), shwtemp(label='Tccd3', current=40.75, high=None, critical=None), shwtemp(label='Tccd4', current=42.0, high=None, critical=None), shwtemp(label='Tccd5', current=40.75, high=None, critical=None), shwtemp(label='Tccd6', current=42.5, high=None, critical=None), shwtemp(label='Tccd7', current=43.5, high=None, critical=None), shwtemp(label='Tctl', current=41.75, high=None, critical=None), shwtemp(label='Tccd8', current=40.25, high=None, critical=None), shwtemp(label='Tccd1', current=40.75, high=None, critical=None), shwtemp(label='Tccd2', current=41.75, high=None, critical=None), shwtemp(label='Tccd3', current=41.0, high=None, critical=None), shwtemp(label='Tccd4', current=41.5, high=None, critical=None), shwtemp(label='Tccd5', current=38.75, high=None, critical=None), shwtemp(label='Tccd6', current=40.0, high=None, critical=None), shwtemp(label='Tccd7', current=41.25, high=None, critical=None)]}, {'nvme': [shwtemp(label='Composite', current=35.85, high=79.85, critical=86.85), shwtemp(label='Sensor 1', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=45.85, high=65261.85, critical=65261.85), shwtemp(label='Composite', current=35.85, high=79.85, critical=86.85), shwtemp(label='Sensor 1', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=45.85, high=65261.85, critical=65261.85), shwtemp(label='Composite', current=35.85, high=79.85, critical=86.85), shwtemp(label='Sensor 1', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 3', current=34.85, high=65261.85, critical=65261.85)], 'i350bb': [shwtemp(label='loc1', current=43.0, high=120.0, critical=110.0)], 'k10temp': [shwtemp(label='Tctl', current=47.875, high=None, critical=None), shwtemp(label='Tccd8', current=39.75, high=None, critical=None), shwtemp(label='Tccd1', current=47.75, high=None, critical=None), shwtemp(label='Tccd2', current=42.25, high=None, critical=None), shwtemp(label='Tccd3', current=40.75, high=None, critical=None), shwtemp(label='Tccd4', current=42.25, high=None, critical=None), shwtemp(label='Tccd5', current=40.0, high=None, critical=None), shwtemp(label='Tccd6', current=42.5, high=None, critical=None), shwtemp(label='Tccd7', current=43.5, high=None, critical=None), shwtemp(label='Tctl', current=42.5, high=None, critical=None), shwtemp(label='Tccd8', current=41.0, high=None, critical=None), shwtemp(label='Tccd1', current=41.0, high=None, critical=None), shwtemp(label='Tccd2', current=41.5, high=None, critical=None), shwtemp(label='Tccd3', current=41.0, high=None, critical=None), shwtemp(label='Tccd4', current=41.75, high=None, critical=None), shwtemp(label='Tccd5', current=39.25, high=None, critical=None), shwtemp(label='Tccd6', current=39.75, high=None, critical=None), shwtemp(label='Tccd7', current=41.5, high=None, critical=None)]}, {'nvme': [shwtemp(label='Composite', current=35.85, high=79.85, critical=86.85), shwtemp(label='Sensor 1', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=45.85, high=65261.85, critical=65261.85), shwtemp(label='Composite', current=35.85, high=79.85, critical=86.85), shwtemp(label='Sensor 1', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=45.85, high=65261.85, critical=65261.85), shwtemp(label='Composite', current=35.85, high=79.85, critical=86.85), shwtemp(label='Sensor 1', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=35.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 3', current=34.85, high=65261.85, critical=65261.85)], 'i350bb': [shwtemp(label='loc1', current=43.0, high=120.0, critical=110.0)], 'k10temp': [shwtemp(label='Tctl', current=49.25, high=None, critical=None), shwtemp(label='Tccd8', current=40.25, high=None, critical=None), shwtemp(label='Tccd1', current=45.75, high=None, critical=None), shwtemp(label='Tccd2', current=42.25, high=None, critical=None), shwtemp(label='Tccd3', current=41.5, high=None, critical=None), shwtemp(label='Tccd4', current=43.0, high=None, critical=None), shwtemp(label='Tccd5', current=41.75, high=None, critical=None), shwtemp(label='Tccd6', current=43.5, high=None, critical=None), shwtemp(label='Tccd7', current=44.5, high=None, critical=None), shwtemp(label='Tctl', current=41.5, high=None, critical=None), shwtemp(label='Tccd8', current=39.25, high=None, critical=None), shwtemp(label='Tccd1', current=40.5, high=None, critical=None), shwtemp(label='Tccd2', current=41.0, high=None, critical=None), shwtemp(label='Tccd3', current=40.5, high=None, critical=None), shwtemp(label='Tccd4', current=41.0, high=None, critical=None), shwtemp(label='Tccd5', current=38.75, high=None, critical=None), shwtemp(label='Tccd6', current=39.75, high=None, critical=None), shwtemp(label='Tccd7', current=41.25, high=None, critical=None)]}], 

# 'avg_load': [(0.32275390625, 0.466796875, 0.40771484375), (0.69775390625, 0.54248046875, 0.4326171875), (1.9150390625, 1.04248046875, 0.630859375)]}


# #### time measurement (end)
# end_time = time.time()
# duration_seconds = end_time - start_time
# minutes = int(duration_seconds // 60)
# seconds = int(duration_seconds % 60)
# print()
# print("[", experiment['slurm_task_id'], "]", "Simulation total time consumed:", minutes, "m", seconds, "s", "(", duration_seconds, "s)")




# #### measure resource consumption (end)
# # Get final memory usage
# final_memory = psutil.virtual_memory().used
# # Get final CPU time
# final_cpu_time = psutil.cpu_times().user + psutil.cpu_times().system
# # Calculate memory consumption
# memory_consumption = final_memory - initial_memory
# # Calculate CPU consumption
# cpu_consumption = final_cpu_time - initial_cpu_time

# print("[", experiment['slurm_task_id'], "]", "Memory consumption:", memory_consumption/1024/1024, "Mbytes")
# print("[", experiment['slurm_task_id'], "]", "CPU consumption:", cpu_consumption, "seconds")
