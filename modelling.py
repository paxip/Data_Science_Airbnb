
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
    print("Number of samples in:")
    print(f"    Training: {len(y_train)}")
    print(f"    Testing: {len(y_test)}")
    
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    print("Number of samples in:")
    print(f"    Training: {len(y_train)}")
    print(f"    Validation: {len(y_validation)}")
    print(f"    Testing: {len(y_test)}")





       
























if __name__ == '__main__':
    listing_df = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
    X,y = Data_Preparation.load_airbnb('Price_Night', listing_df)
    print(X)