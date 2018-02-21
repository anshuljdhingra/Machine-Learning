#Upper Confidence Bound - Reinforcement Learning Model

# importing dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing Upper Confidence Bound - Reinforcement Learning Model to the simulated dataset
N = 10000
d = 10
number_of_ad_selection = integer(d)
sum_of_rewards = integer(d)
total_rewards = 0
ad_selected = integer(0)
for(n in 1:N) {
  max_Upper_Confidence_Bound = 0 
  ad = 0
  reward = 0
  for( i in 1:d) {
    if (number_of_ad_selection[i] > 0) {
      avg_reward = sum_of_rewards[i] / number_of_ad_selection[i]
      delta_i = sqrt(1.5 * (log(n) / number_of_ad_selection[i]))
      upper_confidence_bound = avg_reward + delta_i
    } else {
      upper_confidence_bound = 1e400
    }
    if(upper_confidence_bound > max_Upper_Confidence_Bound) {
      max_Upper_Confidence_Bound = upper_confidence_bound
      ad = i
    }
  }
  ad_selected = append(ad_selected, ad)
  number_of_ad_selection[ad] = number_of_ad_selection[ad] + 1
  reward  = dataset[n,ad]
  total_rewards = total_rewards + reward
  sum_of_rewards[ad] = sum_of_rewards[ad] + reward
}

# visualising the results
hist(x = ad_selected, col = 'red' , main = 'Ad Selections',
     xlab = 'list of ads', ylab = 'total number of ads selected')


