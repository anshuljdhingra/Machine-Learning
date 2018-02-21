# Thompson Sampling - Reinforcement Learning Model

# importing dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# implementing Thompson Sampling - Reinforcement Learning Model to the dataset
N = 10000
d = 10
number_of_reward_1 = integer(d)
number_of_reward_0 = integer(d)
total_reward = 0
ad_selected = integer(0)
for(n in 1:N) {
  max_random_draw = 0
  reward = 0
  ad = 0
  for(i in 1:d) {
    random_draw = rbeta(n =1, shape1 = number_of_reward_1[i] +1,
                        shape2 = number_of_reward_0[i] +1)
    if (random_draw > max_random_draw) {
        max_random_draw = random_draw
        ad = i
    }
    
  }
  reward = dataset[n,ad]
  total_reward = total_reward + reward
  ad_selected = append(ad_selected, ad)
  if (reward == 1) {
    number_of_reward_1[ad] = number_of_reward_1[ad] +1
  }else {
    number_of_reward_0[ad] = number_of_reward_0[ad] + 1
  }
}

# visualisation of the results
hist(x = ad_selected, col='purple', main = 'Ads Selection',
     xlab = 'List of Ads', ylab = 'No of times each ad was selected')
