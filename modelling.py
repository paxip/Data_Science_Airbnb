
from sklearn import datasets
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tabular_data import Data_Preparation
from tabular_data import load_airbnb
import numpy as np
import pandas as pd

def splits_dataset(X,y):
    print(f"Number of samples in dataset: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("Number of samples in:")
    print(f"    Training: {len(y_train)}")
    print(f"    Testing: {len(y_test)}")
       
























if __name__ == '__main__':
       airbnb_df = Data_Preparation.clean_tabular_data()
       X,y = load_airbnb('Price_Night', airbnb_df)
       print(y.shape)