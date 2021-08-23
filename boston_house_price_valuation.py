import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#################################################
# 
# boston_house_price_valuation.py
# Using some (simple but reasonable) assumptions, we are estimating house prices, 
# with a trained and tested linear regression algorithm. 
# Created by Vera Konyves in August 2021, in the course of the Udemy course 
# "Complete 2020 Data Science & Machine Learning Bootcamp". 
#

# Collect Data
boston_dataset = load_boston()   
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names) 
# Check the Boston house prices dataset (collected in the 1970s; publicly available). 
# Based on previous analysis we drop the features INDUS and AGE which do not provide much unique info to the valuation model.  
features = data.drop(['INDUS', 'AGE'], axis=1)   

# We use a log transformation in order to get a better linear regression (with which we predict the prices). 
log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICE'])  

# Important features and their indices in the data set.
CRIME_idx = 0
ZN_idx = 1
CHAS_idx = 2
RM_idx = 4
PTRATIO_idx = 8 

# Scale factor to update the price in USD from that of the 1970s.
old_mean_price = np.median(boston_dataset.target)  
zillow_mean_price = 583.3
scale_factor = zillow_mean_price / old_mean_price

# Template (with average features) for our predictions
property_stats = features.mean().values.reshape(1, 11) 

# Linear regression model to calculate all of the "theta" coefficients. 
regr = LinearRegression().fit(features, target)   
fitted_vals = regr.predict(features)   

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)


# Estimate property log prices
def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river=False, 
                    high_confidence=True):  
        
    # Configure property
    property_stats[0][RM_idx] = nr_rooms  
    property_stats[0][PTRATIO_idx] = students_per_classroom
    
    if next_to_river:   
        property_stats[0][CHAS_idx] = 1
    else:   
        property_stats[0][CHAS_idx] = 0
    
    # Make prediction
    log_estimate = regr.predict(property_stats)[0][0]   

    # Calculate range (prediction interval)
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE   
        lower_bound  = log_estimate - 2*RMSE   
        interval = 95
    else:
        upper_bound = log_estimate + RMSE    
        lower_bound  = log_estimate - RMSE    
        interval = 68         
        
    return log_estimate, upper_bound, lower_bound, interval


# Estimate up-to-date dollar prices
def get_dollar_estimate(rm , ptratio, chas=False, large_range=True):
    """
    Estimate the price of a property in Boston.
    
    Keword arguments:
    rm -- number of rooms in the property.
    ptratio -- number of students per teacher in the classroom for the school in the area.
    chas -- True if the property is next to the Charles River, False otherwise.
    larger_range -- True for a 95% prediction interval, False for a 68% interval.
    
    """
    
    if rm < 1 or ptratio < 1:                       
        print('That is unrealistic. Try again.')
        return    
    
    log_est, upper, lower, conf = get_log_estimate(rm, students_per_classroom=ptratio,
                                                     next_to_river=chas, high_confidence=large_range)

    # Convert to today's dollars
    dollar_est = np.e**log_est * 1000 * scale_factor
    dollar_hi  = np.e**upper * 1000 * scale_factor
    dollar_low = np.e**lower * 1000 * scale_factor


    # Round dollar values to nearest 1000
    rounded_est = np.around(dollar_est, -3)  # to the nearest 1000 
    rounded_hi = np.around(dollar_hi, -3) 
    rounded_low = np.around(dollar_low, -3) 

    print(f'The estimated property price is {rounded_est} USD.')
    print(f'At {conf}% confidence the valuation range is')
    print(f'{rounded_low} USD at the lower end to {rounded_hi} USD at the high end.')
    
