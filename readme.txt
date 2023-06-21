Writing a Data Scientist Blog Post:

This Readme is a brief introduction to the first project in the Data Scientist Nanodegree from Udacity;
The objective was to create a Blog Post with the findings from  Seattle's AirBNB dataset, after its corresponding preparation and analysis, as well as
the conclusions and ideas obtained from it.

The Data Analysis was done through Python, with the following libraries:
Pandas - Data Arrangement
Numpy - Mathematical Procedures
MatlabPlotLib , Seaborn - Data Plotting

Repository Files used for the Project:

calendar.csv - The largest archive on this Dataset, it cointains the reservation dates of every client and ID as well as the price in a time period
listings.csv - This archive contains plenty of information about the list of places, such as descriptions from the location, owner information and so on. 
reviews.csv - Descriptive archive with the reviews written from the customers. - NOT USED -
Udacity Project 1.py - Jupyter notebook with all the code for the analysis
Blogpost link: Link to a brief blog I've created to upload the blog idea.


Short Summary:

  Calendar Analysis:
    The code first loads the calendar data, which contains information about reservations, availability, and prices.
    It explores the data by checking its structure and displaying the first few rows.
    The date column is converted to a month format for further analysis.
    A bar plot is created to show the distribution of reservations throughout the year, highlighting the availability rate for each month.

  Price Analysis:
    The price column is processed to remove unnecessary characters and converted to a numerical format.
    The mean prices for each month are calculated.
    Another bar plot is created to display the average prices for different months.

  Combined Score Analysis:
    A combined score is calculated by multiplying the availability rate and mean price for each month.
    The combined score is normalized between 0 and 1.
    A bar plot is generated to visualize the normalized combined scores, indicating the best months for making reservations based on price and availability.

  Listing Ratings Analysis:
    The listings data is loaded, containing information about various rating scores.
    The code focuses on the review_scores_rating column and analyzes the proportion of negative rating scores based on different thresholds.
    A bar plot is created to show the cumulative proportion of negative rating scores.

  Mean Ratings Analysis:
    Mean ratings for different rating categories (accuracy, cleanliness, check-in, communication, location, and value) are calculated.
    A radar plot is generated to visualize the mean ratings, highlighting the strengths and weaknesses in each category.

Overall, the analysis provides insights into the availability, pricing, and ratings of Airbnb listings, helping users understand the best months for reservations and the overall quality of the listings.



