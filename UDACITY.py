# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:51:13 2023

@author: Goldaraz
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.ticker as mtick



calendar = pd.read_csv(r"U:\Personal\Estudiaciones\Udacity\Data_Scientist\Project_1\calendar.csv")
calendar.head()
calendar.info()
calendar.columns

"In order to get a summary of the dates where there are more reservations and action on the market lets analyze the date column:"
calendar['month'] = pd.DatetimeIndex(calendar['date']).month
month_distribution = calendar.groupby('month').size().reset_index(name='count')
Monthly_reservations_plot=sns.barplot(data=month_distribution, x='month', y='count')


"This first plot gives us the amount of reservations done throught the year, which is not really giving us any deep insight into the dataset"
"A good approach could be to look into the availability and how it changes throughout the year, instead of the total reservations #"
availability_counts = calendar[calendar['available'] == 't'].groupby('month').size()
availability_counts.name = 'count_avg'
month_distribution = month_distribution.join(availability_counts, on='month').fillna(0)
month_distribution['Availability_rate'] = month_distribution['count_avg'] / month_distribution['count']
Av_rate_plot=sns.barplot(data=month_distribution, x=month_distribution.index + 1, y='Availability_rate', color='green').set_ylim(0, 1)


"Looking at the monthly prices for this distribution:"
calendar['price'] = calendar['price'].strip('$').astype(float)
mean_prices = calendar[~calendar['price'].isnull()].groupby('month')['price'].mean()
mean_prices.name = 'mean_price'
month_distribution = month_distribution.join(mean_prices, on='month')                    
Mean_prices_plot=sns.barplot(data=month_distribution, x='month', y='mean_price')

"Finally, in we can normalize the results to create a new parameter, which we will name combined_score_normalized which will give us a number between 0 and 1, where 1 is the most optimum time to book a reservation and 0 the least, taking into account both the price and the availability."
mean_price_normalized = (month_distribution['mean_price'] - month_distribution['mean_price'].min()) / (month_distribution['mean_price'].max() - month_distribution['mean_price'].min())
combined_score = (1 - month_distribution['Availability_rate']) * (1 - mean_price_normalized)
month_distribution['combined_score_normalized'] = combined_score


listings = pd.read_csv(r"U:\Personal\Estudiaciones\Udacity\Data_Scientist\Project_1\listings.csv")
listings.head()
listings.info()



reviews = pd.read_csv(r"U:\Personal\Estudiaciones\Udacity\Data_Scientist\Project_1\reviews.csv")
reviews.head()
reviews.info()

"Another interesting question is to take a look into the amount of positive reviews that are being given by the customers: "
rating_scores=listings[['review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']]
proportion = pd.DataFrame()  # Crear un DataFrame vacío

for i in range(50, 96, 5):
    negative = rating_scores['review_scores_rating'][rating_scores['review_scores_rating'] < i].count()
    proportion = proportion.append(pd.DataFrame([negative / rating_scores.shape[0]], [i]))
    
    
proportion.reset_index(inplace=True)
proportion.columns = ['Rating Score', 'Proportion']

"In the following plot it is observed the amount of negative rating scores in a cumulative way, and only when we set the threshold at 95 points score we see a significant negative proportion, otherwise its under 10%"
plt.figure(figsize=(8, 6))
sns.barplot(x='Rating Score', y='Proportion', data=proportion)
plt.xlabel('Total rating Score')
plt.ylabel('Proportion')
plt.ylim(0,1)
plt.title('Proportion of Negative Rating Scores')
plt.show()    

"Finally we can also take a look into the mean ratings for the other categories"
categories = ['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
              'review_scores_communication', 'review_scores_location', 'review_scores_value']
mean_rating_values=rating_scores.dropna().mean().drop('review_scores_rating')
mean_rating_values = mean_rating_values[:len(categories)]


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)

theta = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

ax.plot(theta, mean_rating_values, marker='o')
ax.fill(theta, mean_rating_values, alpha=0.25)

ax.set_xticks(theta)
ax.set_xticklabels(categories)
ax.set_rticks(np.arange(9, 11, 0.1))  # Establecer los ticks del eje radial
ax.set_rlim(9, 10)  # Establecer los límites del eje radial


plt.title('Mean Ratings')
plt.show()

