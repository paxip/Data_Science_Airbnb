
from sklearn import datasets
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tabular_data import Data_Preparation
import numpy as np
import pandas as pd



def splits_dataset(X,y):
    print(f"Number of samples in dataset: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    
    data_sets = [X_train, y_train, X_test, y_test, X_validation, y_validation]
        
def linear_regression_model(X_train, y_train, X_test, X_validation):
    model = SGDRegressor()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)








       
























if __name__ == '__main__':
    airbnb_df = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
    X,y = Data_Preparation.load_airbnb('Price_Night', airbnb_df)
    print(X)