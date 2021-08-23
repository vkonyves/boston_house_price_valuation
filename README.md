# boston_house_price_valuation

This python module is estimating property prices in Boston.
It uses some (simple but reasonable) assumptions, and a trained and tested linear regression algorithm. 

Dependencies:
- numpy
- pandas
- scikit-learn (datasets, linear_model, metrics)

Keyword arguments:
rm -- Number of rooms in the property.
ptratio -- Number of students per teacher in the classroom for the school in the area.
chas -- True if the property is next to the Charles River, False otherwise.
larger_range -- True for a 95% prediction interval, False for a 68% interval.

Usage with example:
import boston_house_price_valuation as val

# get_dollar_estimate(rm , ptratio, chas=False, large_range=True):
val.get_dollar_estimate(3, 20, True, True)

Tested with Python 3.6.13 and 3.8.10, both in Ipython and Jupyter Notebook.
Created by Vera Konyves in August 2021, in the course of the Udemy course "Complete 2020 Data Science & Machine Learning Bootcamp". 
