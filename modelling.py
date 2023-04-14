from sklearn import datasets
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier
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

def save_model(model, parameters, metrics, folder):
    os.makedirs(folder, exist_ok=True)
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

def get_regression_metrics(y_train, y_train_pred):
    R2 = r2_score(y_train, y_train_pred)
    R2_scores = {'R2_score': R2}
    return R2_scores

def get_classification_metrics(X, y, model):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    F1 = f1_score(y, y_pred, average='macro')
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1': F1}
    return metrics

def tune_classification_model_hyperparameters(model, parameters):
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=2, refit=True, error_score='raise')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_

    # y_train_pred = best_model.predict(X_train)
    metrics = get_classification_metrics(X_validation, y_validation, best_model)
    return best_model, best_parameters, metrics

def tune_regression_model_hyperparameters(model, parameters):
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=2, refit=True)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_

    y_validation_pred = best_model.predict(X_validation)
    metrics = get_regression_metrics(y_validation, y_validation_pred)
    
    model_name = type(model).__name__
    save_model(best_model, best_parameters, metrics, folder=(f'models/regression/{model_name}'))
    return best_model

def evaluate_all_models(task_folder):
    if task_folder == 'models/regression':
        sgdr_parameters = {'loss': ['squared_error', 'huber', 'epsilon_insensitive'], 'alpha': [0.00005,0.0001, 0.0002,], 'max_iter': [1000, 1500, 2000]}
        dtr_parameters = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'splitter': ['best', 'random'], 'max_depth': [None, 2, 5]}
        rfr_parameters = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200], 'criterion' : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'], 'max_depth': [1, 8, 16, 32, 64]}
        gbr_parameters = {'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'], 'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01], 'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200]}

        models_parameters = {SGDRegressor(): sgdr_parameters, DecisionTreeRegressor(): dtr_parameters, RandomForestRegressor(): rfr_parameters, GradientBoostingRegressor(): gbr_parameters}
        for model, parameters in models_parameters.items():
            best_model = tune_regression_model_hyperparameters(model, parameters)
            best_regression_models.append(best_model)    
        return best_regression_models
    
    elif task_folder == 'models/classification':
        LR_parameters = {'solver': ['lbfgs', 'newton-cg', 'newton-cholesky','saga'],'max_iter': [100, 200, 300],'verbose': [0, 1]}
        DT_clf_parameters = {'criterion': ['gini', 'entropy', 'log_loss'], 'splitter': ['best', 'random'],'max_depth': [1, 8, 16]}
        RF_clf_parameters = {'n_estimators': [100, 200, 300],'criterion': ['gini', 'entropy', 'log_loss'],'max_depth': [1, 8, 16], 'min_samples_split': [2, 4, 6],'oob_score': [True, False]}
        GB_clf_parameters = {'learning_rate' : [0.1, 0.5, 1], 'n_estimators': [4, 8, 16], 'criterion': ['friedman_mse', 'squared_error']}
        models_parameters = {LogisticRegression(): LR_parameters, DecisionTreeClassifier(): DT_clf_parameters, RandomForestClassifier(): RF_clf_parameters, GradientBoostingClassifier(): GB_clf_parameters}
    
        for model, parameters in models_parameters.items():
            best_model, best_parameters, metrics = tune_classification_model_hyperparameters(model, parameters)
            model_name = type(model).__name__
            save_model(best_model, best_parameters, metrics, folder=(f'models/classification/{model_name}'))
            best_classification_models.append(best_model)    
        return best_classification_models
    
def find_best_model(task_folder):
    if task_folder == 'models/regression':
        
        best_regression_models = evaluate_all_models()
        for best_model in best_regression_models:
            y_test_pred = best_model.predict(X_test)
            scores = get_regression_metrics(y_test, y_test_pred)
            R2 = scores.get('R2_score')
            print(R2)

        if R2 <= 1:
            print(f'The best_model is {best_model} with an R2 score of {R2}')

    elif task_folder == 'models/classification':
        best_classification_models = evaluate_all_models('models/classification')
        print(best_classification_models)
        for best_model in best_classification_models:
            scores = get_classification_metrics(X_test, y_test, best_model)
            accuracy = scores.get('accuracy')
            print(accuracy)
            
        if accuracy <= 1:
            print(f'The best_model is {best_model} with an accuracy score of {accuracy}')

    model_name = type(best_model).__name__ 
    path = (f'/Users/apple/Documents/GitHub/Data_Science_Airbnb/{task_folder}/{model_name}')
    os.chdir(path)

    load_model = joblib.load('model.joblib')
    load_params = json.loads('hyperparameters.json')
    load_metrics = json.loads('metrics.json')

    return load_model, load_params, load_metrics
    
       
    # with open (f'/Users/apple/Documents/GitHub/Data_Science_Airbnb/{task_folder}/{model_name}', 'r') as fp:
    #     model = joblib.load((f'Data_Science_Airbnb/{task_folder}/{model_name}/model.joblib'))
    #     parameters = json.load(f'Data_Science_Airbnb/{task_folder}/{model_name}/hyperparameters.json')
    #     metrics = json.load(f'Data_Science_Airbnb/{task_folder}/{model_name}/metrics.json')

    # return model, parameters, metrics




            

if __name__ == '__main__':
    # airbnb_df = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
    # X,y = Data_Preparation.load_airbnb('Price_Night', airbnb_df)
    # X = X.select_dtypes(include =['float64', 'int64'])
    # X = scale(X)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    
    best_regression_models = []
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    

    best_classification_models = []

    
    find_best_model('models/classification')





    


    
    
    
    
    
   


  
    


    
    
    










  
    

    




