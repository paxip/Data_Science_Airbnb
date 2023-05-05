import itertools
import json
import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
import yaml

from collections import OrderedDict as OrderedDict
from datetime import datetime
from tabular_data import Data_Preparation
from torchmetrics import R2Score
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv('/Users/apple/Documents/GitHub/Data_Science_Airbnb/airbnb_datasets/clean_tabular_data.csv')
        self.X, self.y = Data_Preparation.load_airbnb('Price_Night', self.data)
        self.X = self.X.select_dtypes(include =['float64', 'int64'])
    
    def __getitem__(self, index):
        return (torch.tensor(self.X.iloc[index]), torch.tensor(self.y.iloc[index]))
    
    def __len__(self):
        return len(self.X)



class NN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_layer_width = config['hidden_layer_width']
        depth = config['depth']
        print(hidden_layer_width)
        print(depth)
        input_nodes = 9
        output_nodes = 1
        self.linear_layers = []
        self.input_layer = torch.nn.Linear(input_nodes, hidden_layer_width)
        
        for hidden_layer in range(depth -1):
            self.linear_layers.append(torch.nn.Linear(hidden_layer_width, hidden_layer_width))
            self.linear_layers.append(torch.nn.ReLU())
        
        self.linear_layers = torch.nn.Sequential(*self.linear_layers)
        print(self.linear_layers)

        self.output_layer = torch.nn.Linear(hidden_layer_width, output_nodes)
        
    def forward(self, X):
        X = self.input_layer(X)
        X = self.linear_layers(X)
        X = self.output_layer(X)
        return X


def train(model, config, epochs=10):

    if config['optimiser'] == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight decay'], momentum=config['momentum'])

    elif config['optimiser'] == 'Adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], amsgrad=config['amsgrad'])
    
    elif config['optimiser'] == 'Adadelta':
        optimiser = torch.optim.Adadelta(model.parameters(), lr=config['learning_rate'], rho=config['rho'])

    writer = SummaryWriter()
    batch_index_1 = 0
    batch_index_2 = 0
    
    training_start_time = time.time()
    prediction_times = []

    for epoch in range(epochs):
        for batch in train_loader:
            X, y = batch
            X = X.to(torch.float32)
            y = y.to(torch.float32)
            y = torch.unsqueeze(y, 1)
            prediction_start_time = time.time()
            y_prediction = model(X)
            prediction_end_time = time.time()
            prediction_duration = prediction_end_time - prediction_start_time
            prediction_times.append(prediction_duration)
            loss = F.mse_loss(y_prediction, y) 
            R2_train = R2Score()
            R2_train = R2_train(y_prediction, y)
            RMSE_train = torch.sqrt(loss)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            # writer.add_scalar('train loss', loss.item(), batch_index_1)
            # writer.add_scalar('train accuracy', R2_train, batch_index_1)  
            batch_index_1 += 1   

        training_end_time = time.time() 

    with torch.no_grad():
        for batch in validation_loader:
            X, y = batch
            X = X.to(torch.float32)
            y = y.to(torch.float32)
            y = torch.unsqueeze(y, 1)
            y_validation = model(X)
            loss_validation = F.mse_loss(y_validation, y)
            R2_validation = R2Score()
            R2_validation = R2_validation(y_validation, y)
            RMSE_validation = torch.sqrt(loss_validation)
            # writer.add_scalar('validation loss', loss_validation.item(), batch_index_2)
            # writer.add_scalar('validation accuracy', R2_validation, batch_index_2)
            batch_index_2 += 1
    
    training_duration = training_end_time - training_start_time
    inference_latency = sum(prediction_times)/len(prediction_times)

    return R2_train, RMSE_train, R2_validation, RMSE_validation, training_duration, inference_latency

def get_nn_config():
    with open('/Users/apple/Documents/GitHub/Data_Science_Airbnb/nn_config.yaml', 'r') as stream:
        try:
            config =yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
        return config

def get_metrics(model, config):
    R2_train, RMSE_train, R2_validation, RMSE_validation, training_duration, inference_latency = train(model, config)
    performance_metrics = {'R2_train': R2_train.item(), 'RMSE_train': RMSE_train, 'R2_validation': R2_validation.item(), 'RMSE_validation': RMSE_validation, 'training duration': training_duration, 'inference_latency': inference_latency}
    return performance_metrics


def save_model(model, performance_metrics, config):
    if not isinstance(model, torch.nn.Module):
        print('This model is not a Pytorch module.')
    
    else:
        folder = os.path.join('Data_Science_Airbnb/neural_networks/regression', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        try:
            os.makedirs(folder)
        except OSError as e:
            print(e)

        file_name = 'model.pt'
        filepath = os.path.join(folder, file_name)
        state_dictionary = model.state_dict()
        torch.save(state_dictionary, filepath)

        with open(f"{folder}/metrics.json", 'w') as fp:
            json.dump(performance_metrics, fp)  

        with open(f"{folder}/hyperparameters.json", 'w') as fp:
            json.dump(config, fp)  


def generate_nn_configs():
    Adam_hyperparameters = {'optimiser': ['Adam'], 'learning_rate': [0.001, 0.0001], 'amsgrad': [True, False], 'hidden_layer_width': [5], 'depth': [3]}
    Adadelta_hyperparameters = {'optimiser': ['Adadelta'], 'learning_rate': [1.0, 0.001, 0.0001], 'rho': [0.3, 0.9], 'hidden_layer_width': [5], 'depth': [3]}
    SGD_hyperparameters = {'optimiser': ['SGD'], 'learning_rate': [1.0, 0.001, 0.0001], 'weight decay': [0.01, 0.02], 'momentum': [0.1], 'hidden_layer_width': [5], 'depth': [3]}

    optimiser_list = [Adam_hyperparameters, Adadelta_hyperparameters, SGD_hyperparameters]
    nn_configs = []

    for optimiser in optimiser_list:
        keys, values = zip(*optimiser.items())
        hyperparameters_dict= [dict(zip(keys, v)) for v in itertools.product(*values)]
        nn_configs.append(hyperparameters_dict)

    return nn_configs


def convert_all_params_to_yaml(nn_configs, yaml_file):
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(nn_configs, f, sort_keys=False, default_flow_style=False)


def find_best_nn():
    nn_configs = generate_nn_configs()
    R2_list = []
    convert_all_params_to_yaml(nn_configs, '/Users/apple/Documents/GitHub/Data_Science_Airbnb/nn_config.yaml')
    get_nn_config()
    for nn_config in nn_configs:
        for config in nn_config:
            model = NN(config)
            trained_model = train(model, config)
            performance_metrics = get_metrics(model, config)
            R2 = performance_metrics['R2_validation']
            R2_list.append(R2)

    
    print(R2_list)
    for R2 in R2_list:
        if R2 <= min(R2_list):
        best_model = trained_model
        save_model(best_model, performance_metrics, config)
        
        













# Save R2 value and    
   

    #  Save config used in the hyperparameters.json file for each model trained.
    # Save best model in a folder.(call save model)
    #  Return model, metrics, hyperparameters. 

    

        




#   save hyperparameters in json file.
#   Calculate RMSE loss and R2 score for training, test and validation.

   
  
if __name__ == '__main__':
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_set, test_set, validation_set = random_split(dataset, [0.7, 0.15, 0.15])
    batch_size = 4
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

   
    # # train(model, config)

    find_best_nn()
    
 
    
    
    
    
   
    
    
    


        
     





# for epoch in range(epochs):
#         for batch in train_loader:
#             X, y = batch
#             X = X.to(torch.float32)
#             y = y.to(torch.float32)
#             y = torch.unsqueeze(y, 1)
#             y_prediction = model(X)
#             train_loss = F.mse_loss(y_prediction, y)
#             train_loss.backward()
#             print(f'Train loss is {train_loss.item()}')
#             R2_train = R2Score()
#             R2_train = R2_train(y_prediction, y)
#             optimiser.step()
#             optimiser.zero_grad()
#             writer.add_scalar('train loss', train_loss.item(), batch_index)
#             writer.add_scalar('train accuracy', R2_train, batch_index)       
                    
#         for batch in validation_loader:
#             X, y = batch
#             X = X.to(torch.float32)
#             y = y.to(torch.float32)
#             y = torch.unsqueeze(y, 1)
#             y_validation = model(X)
#             validation_loss = F.mse_loss(y_validation, y)
#             validation_loss.backward()
#             print(f'Validation loss is {validation_loss.item()}')
#             R2_validation = R2Score()
#             R2_validation = R2_validation(y_validation, y)
#             optimiser.step()
#             optimiser.zero_grad()
#             writer.add_scalar('validation loss', validation_loss.item(), batch_index)
#             writer.add_scalar('validation accuracy', R2_validation, batch_index)

#             batch_index += 1
