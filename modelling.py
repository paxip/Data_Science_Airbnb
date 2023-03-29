
from sklearn.linear_model import SGDRegressor
from sklearn import datasets
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tabular_data import Data_Preparation
from typing import Any
import joblib
import itertools
import numpy as np
import os
import os.path
import pandas as pd




def splits_dataset(X,y):
    print(f"Number of samples in dataset: {len(X)}") 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    return X_train, y_train, X_test, y_test, X_validation, y_validation
   
def linear_regression_model(data_sets):
    model.fit(data_sets[0], data_sets[1])

def evaluate_regression_model(y_test_pred, y_train_pred, data_sets):
    train_loss = mean_squared_error(data_sets[1], y_train_pred)
    test_loss = mean_squared_error(data_sets[3], y_test_pred)

    MSE_train = mean_squared_error(data_sets[1], y_train_pred)
    MSE_test = mean_squared_error(data_sets[3], y_test_pred)
    print(f"MSE for train set: {MSE_train} | "f"MSE for test set: {MSE_test}")

    RMSE_train = mean_squared_error(data_sets[1], y_train_pred, squared=False)
    RMSE_test = mean_squared_error(data_sets[3], y_test_pred, squared=False)
    print(f"RMSE for train set: {RMSE_train} | "f"RMSE for test set: {RMSE_test}")

    MAE_train = mean_absolute_error(data_sets[1], y_train_pred)
    MAE_test = mean_absolute_error(data_sets[3], y_test_pred)
    print(f"MAE for train set: {MAE_train} | "f"MAE for test set: {MAE_test}")

    R2_train = r2_score(data_sets[1], y_train_pred)
    R2_test = r2_score(data_sets[3], y_test_pred)
    print(f"R2 for train set: {R2_train} | "f"R2 for test set: {R2_test}")

def custom_tune_regression_model_hyperparameters(model_type, X_train, y_train, X_validation, y_validation, grid_dict):
    keys, values = zip(*grid_dict.items())
    iteration_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    RMSE_list = []

    for iteration in iteration_dicts:
            model = model_type(learning_rate=iteration['learning_rate'], max_iter=iteration['max_iter'], loss=iteration['loss'], fit_intercept=iteration['fit_intercept'], alpha=iteration['alpha'])
            #linear_regression_model(data_sets)
            model.fit(X_train, y_train)
            # y_train_pred = model.predict(X_train)
            # train_RMSE = mean_squared_error(y_train, y_train_pred, squared=False)
            y_validation_pred = model.predict(X_validation)
            validation_RMSE = mean_squared_error(y_validation, y_validation_pred, squared=False)
            validation_MAE = mean_absolute_error(y_validation, y_validation_pred)
            validation_R2 = r2_score(y_validation, y_validation_pred)
            RMSE_list.append(validation_RMSE)
            print(f"This iteration RMSE is {validation_RMSE}")
            if validation_RMSE <= min(RMSE_list):
                best_model = model
                best_iteration_parameters = iteration
                best_validation_RMSE = validation_RMSE
                best_validation_MAE = validation_MAE
                best_validation_R2 = validation_R2
                print(f"The best RMSE is {best_validation_RMSE}")
                performance_metrics = {'validation RMSE': best_validation_RMSE, 'validation MAE': best_validation_MAE, 'validation R2': best_validation_R2}
    return best_model, performance_metrics, best_iteration_parameters

def tune_regression_model_hyperparametes():
    parameters = {'loss': ['squared_error', 'huber', 'epsilon_insensitive'], 'alpha': [0.00005,0.0001, 0.0002,], 'max_iter': [500, 1000, 1500]}
    grid_LR = GridSearchCV(estimator=model, param_grid=parameters, cv=2, refit=True)
    grid_LR.fit(data_sets[0], data_sets[1])
    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_LR.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_LR.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_LR.best_params_)

def create_directories(model_path, reg_path):
    path = os.path.join(model_path, reg_path)  
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print("Directory '%s' can not be created")
    return path
   
def save_model(fp, filename):
    try:
        with open(os.path.join(fp, filename), "w") as outfile:
            joblib.dump(model, filename)
            print('model.joblib file created.')
            return True
        
    except Exception as e:
        print(e)
        return False
  
     
     
     



    
    




     
        








if __name__ == '__main__':
    airbnb_df = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
    X,y = Data_Preparation.load_airbnb('Price_Night', airbnb_df)
    # print(X)
    splits_dataset(X,y)
    X_train, y_train, X_test, y_test, X_validation, y_validation = splits_dataset(X,y)
    data_sets = [X_train, y_train, X_test, y_test, X_validation, y_validation]
    model = SGDRegressor()
    
    # linear_regression_model(data_sets)
    # y_train_pred = model.predict(data_sets[0])
    # y_validation_pred = model.predict(data_sets[4])
    # y_test_pred = model.predict(data_sets[2])
    
    # evaluate_regression_model(y_test_pred, y_train_pred, data_sets)
    # grid_dict = {'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'max_iter': [500, 1000, 1500, 2000, 2500, 3000], 'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 'fit_intercept' : [True, False], 'alpha': [0.00005,0.0001, 0.00015, 0.0002]}


    # best_model, performance_metrics, best_iteration_parameters = custom_tune_regression_model_hyperparameters(SGDRegressor, X_train, y_train, X_validation, y_validation, grid_dict)
    # print(f"The best model is {best_model}")
    # print(f"The best_parameters are {best_iteration_parameters}")
    # print(f"The best performance_metrics are {performance_metrics}")

    # tune_regression_model_hyperparametes()

    create_directories('/Users/apple/Documents/GitHub/Data_Science_Airbnb/model', 'regression')
    
    save_model('/Users/apple/Documents/GitHub/Data_Science_Airbnb/model/regression', '/Users/apple/Documents/GitHub/Data_Science_Airbnb/model/regression/model.joblib')
  
    

    




