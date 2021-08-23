# boston_house_price_valuation

This python module is estimating property prices in Boston.
It uses some (simple but reasonable) assumptions, and a trained linear regression algorithm. 

# Dependencies

- numpy
- pandas
- scikit-learn (datasets, linear_model, metrics)

# Keyword arguments for get_dollar_estimate:

- rm -- Number of rooms in the property.
- ptratio -- Number of students per teacher in the classroom for the school in the area.
- chas -- True if the property is next to the Charles River, False otherwise.
- larger_range -- True for a 95% prediction interval, False for a 68% interval.

# Usage with example:

import boston_house_price_valuation as val

val.get_dollar_estimate(4, 20, False, True)

# Created and Tested:

Created by Vera Konyves in August 2021, in the course of the Udemy course "Complete 2020 Data Science & Machine Learning Bootcamp". 
Tested with Python 3.6 and 3.8, both in Ipython and Jupyter Notebook.
