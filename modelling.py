from sklearn import datasets
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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



def get_regression_metrics(y_train, y_train_pred):
    # RMSE = mean_squared_error(y_train, y_train_pred, squared = False)
    R2 = r2_score(y_train, y_train_pred)
    return R2

def get_classification_metrics(X, y, model):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    F1 = f1_score(y, y_pred, average='macro')
    return accuracy, precision, recall, F1
    


def custom_tune_regression_model_hyperparameters(grid_dict, model_type=SGDRegressor):
    keys, values = zip(*grid_dict.items())
    iteration_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    R2_list = []

    for iteration in iteration_dicts:
            model = model_type()
            model.set_params(**iteration)
            # model = model_type(learning_rate=iteration['learning_rate'], max_iter=iteration['max_iter'], loss=iteration['loss'], fit_intercept=iteration['fit_intercept'], alpha=iteration['alpha'])
            model.fit(X_train, y_train)
            y_validation_pred = model.predict(X_validation)
            validation_R2 = r2_score(y_validation, y_validation_pred)
            validation_RMSE = mean_squared_error(y_validation, y_validation_pred, squared=False)
            R2_list.append(validation_R2)
            
            if validation_R2 <= min(R2_list):
                best_model = model
                best_iteration_parameters = iteration
                best_validation_R2 = validation_R2
                performance_metrics = {'R2': best_validation_R2, 'RMSE' : validation_RMSE}
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



def tune_regression_model_hyperparameters(model, parameters):
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=2, refit=True)
    grid_search.fit(X_train, y_train)
    
    
    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_

    y_train_pred = best_model.predict(X_train)
    metrics = get_regression_metrics(y_train, y_train_pred)
    
    model_name = type(model).__name__
    save_model(best_model, best_parameters, metrics, folder=(f'models/regression/{model_name}'))
    return best_model

def evaluate_all_models():
    sgdr_parameters = {'loss': ['squared_error', 'huber', 'epsilon_insensitive'], 'alpha': [0.00005,0.0001, 0.0002,], 'max_iter': [1000, 1500, 2000]}
    dtr_parameters = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'splitter': ['best', 'random'], 'max_depth': [None, 2, 5]}
    rfr_parameters = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200], 'criterion' : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'], 'max_depth': [1, 8, 16, 32, 64]}
        # 'min_samples_split': [1, 2, 4, 8], 'min_samples_leaf':  [1, 1.5, 2], 'max_features': ['sqrt', 'log2', None], 'warm_start': [True, False]}
    gbr_parameters = {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'], 'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01], 'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200]}
        # 'subsample': [0.0, 0,5, 1.0], 'criterion': ['friedman_mse', 'squared_error'], 'min_samples_leaf': [0.5, 1], 'minimum_weight_fraction': [0.0, 0.25, 0.5]}

    models_parameters = {SGDRegressor(): sgdr_parameters, DecisionTreeRegressor(): dtr_parameters, RandomForestRegressor(): rfr_parameters, GradientBoostingRegressor(): gbr_parameters}
    for model, parameters in models_parameters.items():
        best_model = tune_regression_model_hyperparameters(model, parameters)
        best_models.append(best_model)    
    return best_models
    
def find_best_model():
    R2_scores = []
    best_models = evaluate_all_models()
    print(best_models)
    for best_model in best_models:
        y_test_pred = best_model.predict(X_test)
        R2 = get_regression_metrics(y_test, y_test_pred)
        R2_scores.append(R2)
    print(R2_scores)

    if R2 <= 1:
        print(f'The best_model is {best_model} with an R2 score of {R2}')
        model_name = type(best_model).__name__
        # joblib.load((f'Data_Science_Airbnb/models/regression/{model_name}/model.joblib'))
        # json.load(f'Data_Science_Airbnb/models/regression/{model_name}/hyperparameters.json')
        # json.load(f'Data_Science_Airbnb/models/regression/{model_name}/metrics.json')

    return best_model


    

    
     

if __name__ == '__main__':
    # airbnb_df = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
    # X,y = Data_Preparation.load_airbnb('Price_Night', airbnb_df)
    # X = X.select_dtypes(include =['float64', 'int64'])
    # X = scale(X)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    
    # best_models = []
    # find_best_model()
   

    # grid_dict = {'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 
    #             'max_iter': [500, 1000, 1500, 2000, 2500, 3000], 
    #             'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 
    #             'fit_intercept' : [True, False], 
    #             'alpha': [0.00005,0.0001, 0.00015, 0.0002]}


    # Keep hashtag best_model, performance_metrics, best_iteration_parameters = custom_tune_regression_model_hyperparameters(SGDRegressor, X_train, y_train, X_validation, y_validation, grid_dict)


    # 1. Pass a lisitngs data frame to include 'Category' since it is non numerical data.
    # 2. Import load_airbnb function like you did for Price_Night but loading in dataset with "Category" as the label.
    # 3. Find a method to convert 'Category' into numerical data so that it can be used for training the data.
    # Use sklearn to train a logistic regression model to predict the category from the tabular data.


    airbnb_df = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
    airbnb_df['Category'].astype('category')
    X,y = Data_Preparation.load_airbnb('Category', airbnb_df)
    X = X.select_dtypes(include =['float64', 'int64'])
    X = scale(X)

    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

    best_models = []
    best_model = find_best_model()



    


    
    
    
    
    
   


  
    


    
    
    










  
    

    




