# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:17:30 2017

@author: adhingra
"""
# Upper Confidence Bound - Reinforcement Learning Model

# we need to select an ad out of 10 ads that has highest number of selections and total rewards
# we would be needing sum of selections for each ad and corresponding sum of rewards at all rounds.

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Fitting Upper Confidence Bound - Reinforcement Learning Model to the dataset
import math
N = 10000
d = 10
ad_selected = []
total_rewards = 0
sum_of_rewards = [0] * d
number_of_ad_selections = [0] * d
for n in range(0,N):
    max_upper_bound = 0 
    reward = 0
    ad = 0
    for i in range(0,d):
        if number_of_ad_selections[i] > 0:
            avg_reward = sum_of_rewards[i] / number_of_ad_selections[i]
            delta_i = math.sqrt(1.5 * (math.log(n+1) / number_of_ad_selections[i]))
            upper_bound  = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    number_of_ad_selections[ad] = number_of_ad_selections[ad] + 1
    reward = dataset.values[n,ad]
    total_rewards = total_rewards + reward
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    ad_selected.append(ad)
    
# visualising the ad selctions
plt.hist(ad_selected)
plt.title('Ad Selections')
plt.xlabel('List of Ads')
plt.ylabel('Each Ad selection')
plt.show()
