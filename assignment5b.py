import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
"""
Name: Rayhan Chowdhury




The decision tree regressor for this data-set seems to outperform the linear regressor consistently. 
This reflects on the data-set, suggesting it may not be all that linear. 
This suggests to me that the data-set had more nonlinearities for the Decision Tree to capture. 
Ultimately, perhaps the target was not as linearly correlated with the input features when it came to Linear Regression. 

"""




#loading data sets
def load_dataset(dataset):
    """This method loads the data set, splits the testing and training between 75% and 25%, 
    and then returns the training data, training labels, testing data, testing labels"""
    pass

    raw_data = np.loadtxt(dataset, dtype = float, delimiter = ",") #dataset read and stored in variable
    
    labels = raw_data[:,-1] #dataset labels furthest right column of dataset specified
    data = raw_data[:,:-1].astype(float) #dataset specified of all columns except the last one 
    

    train_data,test_data,train_labels,test_labels = train_test_split(data, labels, train_size = .75, test_size = .25, shuffle=True) #training at 80% and testing at 20% split



    return train_data,test_data,train_labels,test_labels 


analyze_data = load_dataset("Concrete_Data.csv") #load dataset; dataset reference: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
train_data = np.array(analyze_data[0]) #returned train data as np array stored in variable
test_data = np.array(analyze_data[1]) #returned test data as np array stored in variable
trained_labels = np.array(analyze_data[2]) #returned trained labels/target as np array stored in variable
test_labels = np.array(analyze_data[3]) #returned test labels/target as np array stored in variable


print("           ")
print("           ")
print("THE TRAINING & TESTING DATA")
print("=============================")
print("Size of train data is:", train_data.size) #print the size of train data
print("Size of test data is:", test_data.size) #print the size of test data
print("Shape of train data is:", train_data.shape) #the shape of train data
print("Shape of test data is:", test_data.shape) #the shape of test data
print("Total number of features:", train_data.shape[1] + 1 ) #total number of features 
print("8 inputs; 1 output")



#Training linear regressor
linear_regressor = LinearRegression()
linear_regressor.fit(train_data,trained_labels)

#Training Decision Tree regressor
trees_regressor = DecisionTreeRegressor(criterion="mse",splitter="best",max_depth=100, random_state=0) 
trees_regressor.fit(train_data,trained_labels)

#Testing model 
predict_model = linear_regressor.predict(test_data)
predict_trees = trees_regressor.predict(test_data)

#Print coefficients and intercepts of the model
print("           ")
print("           ")

print("LINEAR REGRESSION RESULTS")
print("=============================")
print("Coefficients:", linear_regressor.coef_) #linear regressor coefficients
print("Intercepts:", linear_regressor.intercept_) #linear regressor intercept
print("Correlation(r):", np.corrcoef(predict_model,test_labels)[0,1]) #correlation r value
print("Residual Sums of Squares:", ((predict_model-test_labels)**2).sum()) #residual sums of squares 
print("           ")
print("           ")
print("DECISION TREE REGRESSION RESULTS")
print("=============================")
print("Correlation(r):", np.corrcoef(predict_trees,test_labels)[0,1]) #correlation r value
print("Residual Sums of Squares:", ((predict_trees-test_labels)**2).sum()) #residual sums of squares



