from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from tabular_data import Data_Preparation
from typing import Any
import joblib
import json
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
   
def linear_regression_model(model, data_sets):
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

def regression_model_performance(model, data_sets):
    y_validation_pred = model.predict(data_sets[4])
    RMSE = mean_squared_error(data_sets[5], y_validation_pred, squared=False)
    MAE = mean_absolute_error(data_sets[5], y_validation_pred)
    R2 = r2_score(data_sets[5], y_validation_pred)
    return RMSE, MAE, R2

def save_model(model, parameters, metrics, folder):
    os.makedirs(folder)
    filepaths = []
    filenames = ['model.joblib','hyperparameters.json', 'metrics.json']
    for file in filenames:
        filepath = os.path.join(folder, file)
        filepaths.append(filepath)
    model_fp, hyperparams_fp, metrics_fp = filepaths
    
    joblib.dump(model, model_fp)
    
    with open(hyperparams_fp, 'w') as file:
        json.dump(parameters, file)

    with open(metrics_fp, 'w') as file:
        json.dump(metrics, file)

def tune_regression_model_hyperparameters(model, parameters, data_sets):
    hyperparameter_tuning = GridSearchCV(estimator=model, param_grid=parameters, cv=2, refit=True)
    linear_regression_model(hyperparameter_tuning, data_sets)
    # best_model.fit(data_sets[0], data_sets[1])
    metrics = regression_model_performance(hyperparameter_tuning, data_sets)
    best_estimator = hyperparameter_tuning.best_estimator_
    best_parameters = hyperparameter_tuning.best_params_
    print(f' The best_estimator is {best_estimator}.')
    print(f'The best parameters for this estimator are {best_parameters}.')
    print(f'The metrics for this model are {metrics}')
    model_name = type(model).__name__
    save_model(best_estimator, best_parameters, metrics, folder=(f'Data_Science_Airbnb/models/regression/{model_name}'))
    return metrics, best_estimator, best_parameters




def evaluate_all_models():
    sgdr_parameters = {'loss': ['squared_error', 'huber', 'epsilon_insensitive'], 'alpha': [0.00005,0.0001, 0.0002,], 'max_iter': [500, 1000, 1500]}
    dtr_parameters = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'splitter': ['best', 'random'], 'max_depth': [None, 2, 5]}
    rfr_parameters = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200], 'criterion' : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'], 'max_depth': [1, 8, 16, 32, 64]}
        # 'min_samples_split': [1, 2, 4, 8], 'min_samples_leaf':  [1, 1.5, 2], 'max_features': ['sqrt', 'log2', None], 'warm_start': [True, False]}
    gbr_parameters = {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'], 'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01], 'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200]}
        # 'subsample': [0.0, 0,5, 1.0], 'criterion': ['friedman_mse', 'squared_error'], 'min_samples_leaf': [0.5, 1], 'minimum_weight_fraction': [0.0, 0.25, 0.5]}

    models_parameters = {SGDRegressor(): sgdr_parameters, DecisionTreeRegressor(): dtr_parameters, RandomForestRegressor(): rfr_parameters, GradientBoostingRegressor(): gbr_parameters}
    for model, parameters in models_parameters.items():
        tune_regression_model_hyperparameters(model, parameters, data_sets)


    

  
    
    
  
  

if __name__ == '__main__':
    airbnb_df = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
    X,y = Data_Preparation.load_airbnb('Price_Night', airbnb_df)
    X = scale(X)
    # print(X)
    X_train, y_train, X_test, y_test, X_validation, y_validation = splits_dataset(X,y)
    data_sets = [X_train, y_train, X_test, y_test, X_validation, y_validation]
    
    evaluate_all_models()
    # RMSE, MAE, R2 = regression_model_performance(model, data_sets)
    # metrics = RMSE, MAE, R2
   







    
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

    # tune_regression_model_hyperparameters()
    #parameters = {'loss': ['squared_error', 'huber', 'epsilon_insensitive'], 'alpha': [0.00005,0.0001, 0.0002,], 'max_iter': [500, 1000, 1500]}
    
    # metrics = tune_regression_model_hyperparameters()
    # save_model(model, parameters, metrics, 'models/regression/linear_regression')

    
    
    










  
    

    




