#!/usr/bin/env python3
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import medfilt
import os
import csv
from datetime import datetime


filename = "test_output.csv"
initial_personality_means = [0.1,0.3,0.5,0.7,0.6]
weight = 0.0
alpha = 0
personality_buffer = np.empty((0,5),"float32")
ow=np.array([[1 / 3, 0, 2 / 3, 0, 0, 0, 0, 0],   #o_def
    [0.0, 0, 0.0, 0, 0.0, 0, 1.0, 0],           #c_def
    [2 / 3, 0, 1 / 3, 0, 0, 0, 0, 0],           #e_def
    [1 / 3, 0, 0, 0, 2 / 3, 0, 0, 0],            #a_def
    [0, 2 / 3, 0, 0, 0, 1 / 3, 0, 0]])          #n_def
h = []
def read_csv(filename):
        MFC_data = pd.read_csv(filename)
        time = MFC_data['time'].to_numpy()
        data = MFC_data.loc[:,'cell_1':].to_numpy()
        return time,data
def plot(X,Y,labels):
    fig, ax = plt.subplots(figsize=(10,6))
    for i in range(Y.shape[1]):
        y = Y[:,i]
        ax.plot(y,label=labels[i])
        ax.legend()
    plt.show()
class Robot:
    def __init__(self,mu,ow,weight=0.5,alpha=0.5):
        self.initial_personality = self.create_random_personality(mu)
        #self.acquired_personality = self.update_acquired_personality()
        self.personality = self.initial_personality
        self.last_personality = self.personality
        self.mfc_data = np.empty((0,1),"float32")
        self.time_data = np.empty((0,1),"float32")
        self.diff = np.empty((0,1),"float32")
        self.ow = ow
        self.order = 0
        self.answer = 0
        self.happiness = 0
        self.stability = 0
        self.health = [0,0,0,0] #h--,h-,h+,h++
        self.instability = [0,0,0,0] #i--,i-,i+,i++
        self.mean_mfc = 0.0
        self.mean_diff = 0.0
        self.std_mfc = 0.0
        self.std_diff = 0.0
        self.weight = weight
        self.alpha = alpha

    def create_random_personality(self,mu):
        sigma = 0.1
        if len(mu)==5:
            personality = [np.random.normal(i,sigma) for i in mu]
        else:
            print("error creating personalities....setting default personality ")
            personality = [0.5*5]
        return personality

    def get_mean_std(self):
        if len(self.mfc_data)>0:
            self.mean_mfc,self.std_mfc = np.mean(self.mfc_data[-100:]),np.std(self.mfc_data[-100:])
        if len(self.diff)>0:
            self.mean_diff,self.std_diff = np.mean(self.diff[-100:]),np.std(self.diff[-100:])

    def update_hs(self):
        if len(self.mfc_data) and len(self.diff):
            health_bins = np.array([-1e+10,self.mean_mfc-self.std_mfc,self.mean_mfc+1e-10,self.mean_mfc+self.std_mfc+2e-10,1e+10])
            stability_bins = np.array([-1e+10,self.mean_diff-self.std_diff,self.mean_diff+1e-10,self.mean_diff+self.std_diff+2e-10,1e+10])
            self.health[np.digitize(self.mfc_data[-1],health_bins)-1] += 1
            self.instability[-np.digitize(self.diff[-1],stability_bins)] += 1


    def update_order_answer(self,o=0,a=0):
        self.order = (o+1)/2
        self.answer = (a+1)/2

    def update_mfc_data(self,data,time):
        self.mfc_data = np.append(self.mfc_data,data)[-101:]
        self.time_data = np.append(self.time_data,time)[-101:]
        self.diff = np.abs(np.diff(self.mfc_data))

    def update_happiness_stability(self):

        if not all(value == 0 for value in self.health):
            self.happiness = (2 * self.health[3] + self.health[2]) / (
                        2 * self.health[3] + self.health[2] +
                        self.health[1] + 2 * self.health[0])
            self.stability = (2 * self.instability[3] + self.instability[2]) / (
                           2 * self.instability[3] + self.instability[2] +
                        self.instability[1] + 2 * self.instability[0])
        else:
            self.happiness = 0
            self.stability = 1
        #print(self.stability)


    def update_acquired_personality(self):
        vals = np.array([self.happiness,1-self.happiness,self.answer,1-self.answer,self.order,1-self.order,self.stability,1-self.stability,])
        #print(vals)
        self.acquired_personality = [np.sum(np.multiply(self.ow[i],vals)) for i in range(len(self.initial_personality))]
        return self.acquired_personality

    def update_final_personality(self):
        self.last_personality = self.personality
        self.personality = [self.initial_personality[i]*self.weight + self.acquired_personality[i]*(1-self.weight) for i in range(len(self.initial_personality))]
        return self.personality
    def smoothening(self):
        self.personality = [self.personality[i]*self.alpha + self.last_personality[i]*(1-self.alpha) for i in range(len(self.personality))]
        return self.personality

if __name__ == "__main__":
    time,data = read_csv(filename)
    robot = Robot(initial_personality_means,ow,weight,alpha)
    order = 0
    answer = 0.5
    #print(ow.shape)

    for i in range(len(time)):
        robot.update_mfc_data(data[i,2],time[i])
        if i>0:
            robot.get_mean_std()
            robot.update_order_answer(order,answer)
            robot.update_hs()
            robot.update_happiness_stability()
            personality = robot.update_acquired_personality()
            #personality = robot.update_final_personality()
            #personality = robot.smoothening()
            personality_buffer = np.append(personality_buffer,np.array([personality]),axis=0)
    #print(personality)
    #print(personality_buffer[0:10])
    plot(time,personality_buffer[:],["--","-","+","++","1"])
    #plt.plot(personality_buffer[:,-1])
    #plt.show()
