#!/usr/bin/env python
# coding: utf-8

# Clear the old File

# In[1]:


#clear all the files in the Experiments folder
import os
path = os.getcwd()
upper_path = os.path.abspath(os.path.join(path, os.pardir))
folder_path = upper_path + "/Experiments/"
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        os.remove(folder_path + filename)
print("Clear finish")


# Experiments setup
# In[4]:


import os
import csv
import re
repeat = 10
line = 0
randomseed = 0###################################################### change here every run ######################################
randomseed = str(randomseed)
path = os.getcwd() + "/Database"
root_path = os.getcwd()
upper_path = os.path.abspath(os.path.join(root_path, "..")) 
# 指定路径
modelbase = []
dir_list = os.listdir(path)
for dir in dir_list:
    if dir.startswith("Modelbase"):
        modelbase.append(dir)
for mymodel in modelbase:
    experiments = []
    folder_path = path + "/" + mymodel
    # folder_path= folder_path +mymodel
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        #if filename is folder
        if os.path.isdir(folder_path + "/" + filename):
            model_path = folder_path + "/" + filename
            for datasetname in os.listdir(model_path):
                # 检查文件是否以 '.csv' 结尾
                if datasetname.endswith('.csv') and datasetname !="LHD_data_200000.csv" and datasetname !="LHD_data_100000.csv":
                    for i in range(repeat):
                        # 如果是，打印文件名
                        randomseed = str(i)
                        print(mymodel+","+filename+","+datasetname+","+"LHD_data_200000.csv"+ ',' + randomseed + "," + str(line))
                        experiments.append(mymodel+","+filename+","+datasetname+","+"LHD_data_200000.csv"+ ',' + randomseed + "," + str(line))
                        line = line + 1
    # 将列表保存为CSV文件
    with open(upper_path + '/Experiments/experiment_setup.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if line == 0:
            writer.writerow(["model", "method", "dataset", "LHD_data", "randomseed", "line"])
        for i in experiments:
            writer.writerow(i.split(','))
    file.close()
print("finish")

