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
month_distribution=calendar.groupby(['month']).size().reset_index(name='count')
#month_distribution['month'] = month_distribution.index+1
sns.barplot(data=month_distribution, x="month", y="count")

"This first plot gives us the amount of reservations done throught the year, which is not really giving us any deep insight into the dataset"
"A good approach could be to look into the availability and how it changes throughout the year, instead of the total reservations #"
month_distribution['count_avg']=calendar[calendar['available'] == 't'] .groupby(['month']).size()
month_distribution['ava_rate']=month_distribution['count_avg']/month_distribution["count"]
sns.barplot(data=month_distribution, x="month", y="ava_rate",color="green").set_ylim(0,1)

"Looking at the monthly prices for this distribution:"
calendar.columns = calendar.columns.str.strip()
Avg_prices=calendar.groupby('month')['date'].mean().reset_index()
                    

listings = pd.read_csv(r"U:\Personal\Estudiaciones\Udacity\Data_Scientist\Project_1\listings.csv")
listings.head()
listings.info()

"Another interesting question is to take a look into the amount of positive reviews that are being given by the customers all along the "

reviews = pd.read_csv(r"U:\Personal\Estudiaciones\Udacity\Data_Scientist\Project_1\reviews.csv")
reviews.head()
reviews.info()


rating_scores=listings[['review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']]
proportion={}
for i in range(50,91,10):
    negative=rating_scores['review_scores_rating'][rating_scores['review_scores_rating'] < i].count()
    proportion=pd.DataFrame([negative/rating_scores.shape[0]],[i])
print(proportion)

mean_rating_values=rating_scores.dropna().mean().drop('review_scores_rating')
import plotly.express as px
df_plot = pd.DataFrame(dict(
    r=mean_rating_values,
    theta=['review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']))
fig = px.line_polar(df_plot, r='r', theta='theta', line_close=True)
fig.show()

