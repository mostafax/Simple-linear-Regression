import numpy as np
import pandas as pd
import sklearn

#Caculte Solpe and interspection
def simple_linear_regression(input_feature, output):
    #Y=MX+b
    n =len(input_feature)
    x =input_feature
    y =output
    x_mean =x.mean()
    y_mean = y.mean()
    sum_xy = (y*x).sum()
    xy_by_n = (y.sum() * x.sum())/n
    x_square = (x ** 2).sum()
    xx_by_n = (x.sum() * x.sum()) / n

    # use the formula for the slope
    slope = (sum_xy - xy_by_n) / (x_square - xx_by_n)
    #slope = y2-y1/x2-x1

    # use the formula for the intercept
    intercept = y_mean - (slope * x_mean)
    #b = y-mx
    return(intercept, slope)



#Caculate Predected Value
def Predict(input_feature, intercept, slope):
    predected_value = intercept+(slope * input_feature)
    return predected_value

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predicted_values = intercept + (slope * input_feature)
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = output - predicted_values
    # square the residuals and add them up
    RSS = (residuals **2 ).sum()
    return(RSS)

#Givin the price caculte sq_ft
def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = (output - intercept)/slope
    return estimated_feature

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

DatasetTrain = pd.read_csv("kc_house_train_data.csv",dtype=dtype_dict)
test_data = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
Y = DatasetTrain.iloc[:,2].values #Price
X =DatasetTrain['sqft_living'].values #Size in feet
sqft_intercept, sqft_slope = simple_linear_regression(DatasetTrain['sqft_living'].values, DatasetTrain['price'].values)
s=simple_linear_regression(X,Y)
#print(s[0])
pre=Predict(X,s[0],s[1])
#print(pre[0:10])
rss_prices_on_sqft = get_residual_sum_of_squares(DatasetTrain['sqft_living'], DatasetTrain['price'], sqft_intercept, sqft_slope)
#print ('The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft))


# Compute RSS when using bedrooms on TEST data:
sqft_intercept, sqft_slope = simple_linear_regression(DatasetTrain['bedrooms'].values,
                                                      DatasetTrain['price'].values)
rss_prices_on_bedrooms = get_residual_sum_of_squares(test_data['bedrooms'].values,
                                                     test_data['price'].values,
                                                     sqft_intercept, sqft_slope)
print('The RSS of predicting Prices based on Bedrooms is : ' + str(rss_prices_on_bedrooms))



# Compute RSS when using squarfeet on TEST data:
sqft_intercept, sqft_slope = simple_linear_regression(DatasetTrain['sqft_living'].values,
                                                      DatasetTrain['price'].values)
rss_prices_on_sqft = get_residual_sum_of_squares(test_data['sqft_living'].values,
                                                 test_data['price'].values,
                                                 sqft_intercept, sqft_slope)
print('The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft))


if rss_prices_on_bedrooms < rss_prices_on_sqft:
    print(rss_prices_on_bedrooms+" is Smaller")
else:
    print(rss_prices_on_sqft+" is Smaller")
