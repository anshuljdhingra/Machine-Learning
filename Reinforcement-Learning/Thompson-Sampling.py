# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:25:25 2017

@author: adhingra
"""
# Thompson Sampling - Reinforcement Learning Model

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Fitting & Implementing Thompson Sampling - Reinforcement Learning Model to the dataset
import random
N = 10000
d = 10
number_of_reward_1 = [0] * d
number_of_reward_0 = [0] * d
ad_selected = []
total_reward = 0
for n in range(0,N):
    max_random_draw = 0 
    ad = 0
    reward = 0
    for i in range(0,d):
        random_beta = random.betavariate(number_of_reward_1[i] +1, number_of_reward_0[i] +1)
        if random_beta > max_random_draw :
            max_random_draw = random_beta
            ad = i
    ad_selected.append(ad)        
    reward = dataset.values[n,ad]
    total_reward = total_reward + reward
    if reward == 1:
        number_of_reward_1[ad] = number_of_reward_1[ad] + 1
    else :
        number_of_reward_0[ad] = number_of_reward_0[ad] + 1
    
# visualising the results
plt.hist(x = ad_selected, color= 'pink')
plt.title('Ads Selection')
plt.xlabel('List of Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
            
