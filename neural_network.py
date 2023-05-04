import datetime
import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
import yaml

from collections import OrderedDict as OrderedDict
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
        optimiser = torch.optim.SGD(model.parameters(), lr=config['learning rate'], weight_decay=config['weight decay'], momentum=config['momentum'])
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
            writer.add_scalar('train loss', loss.item(), batch_index_1)
            writer.add_scalar('train accuracy', R2_train, batch_index_1)  
            batch_index_1 += 1   

        training_end_time = time.time() 
                    
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
            writer.add_scalar('validation loss', loss_validation.item(), batch_index_2)
            writer.add_scalar('validation accuracy', R2_validation, batch_index_2)
            batch_index_2 += 1
    
    training_duration = training_end_time - training_start_time
    inference_latency = sum(prediction_times)/len(prediction_times)

    return R2_train, RMSE_train, R2_validation, RMSE_validation, training_duration, inference_latency

def get_nn_config():
    with open('/Users/apple/Documents/GitHub/Data_Science_Airbnb/nn_config.yaml', 'r') as stream:
        try:
            config =yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as e:
            print(e)   
        return config

def save_model(folder):
    if not isinstance(model, torch.nn.Module):
        print('This model is not a Pytorch module.')
    
    else:
        os.makedirs(folder, exist_ok=True)
        path = folder
        file_name = 'model.pt'
        with open(os.path.join(path, file_name), 'w') as fp:

#   Check if model is pytorch model.
#   If yes, save torch model in a file called model.pt - use video for this part.
#   save hyperparameters in json file.
#   Calculate RMSE loss and R2 score for training, test and validation.
#   Time taken to train the model under a key called training_duration.
#   Time taken to make a prediction under a key called inference_latency.






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

    
  
if __name__ == '__main__':
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_set, test_set, validation_set = random_split(dataset, [0.7, 0.15, 0.15])
    batch_size = 4
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    config = get_nn_config()
    model = NN(config)
    train(model, config)

 
    
    
    
    
   
    
    
    


        
     





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
