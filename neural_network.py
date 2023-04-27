
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
        batch_index = 0

    for epoch in range(epochs):
        for batch in train_loader:
            X, y = batch
            X = X.to(torch.float32)
            y = y.to(torch.float32)
            y = torch.unsqueeze(y, 1)
            y_prediction = model(X)
            train_loss = F.mse_loss(y_prediction, y)
            train_loss.backward()
            print(f'Train loss is {train_loss.item()}')
            R2_train = R2Score()
            R2_train = R2_train(y_prediction, y)
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('train loss', train_loss.item(), batch_index)
            writer.add_scalar('train accuracy', R2_train, batch_index)       
                    
        for batch in validation_loader:
            X, y = batch
            X = X.to(torch.float32)
            y = y.to(torch.float32)
            y = torch.unsqueeze(y, 1)
            y_validation = model(X)
            validation_loss = F.mse_loss(y_validation, y)
            validation_loss.backward()
            print(f'Validation loss is {validation_loss.item()}')
            R2_validation = R2Score()
            R2_validation = R2_validation(y_validation, y)
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('validation loss', validation_loss.item(), batch_index)
            writer.add_scalar('validation accuracy', R2_validation, batch_index)

            batch_index += 1

def get_nn_config():
    with open('/Users/apple/Documents/GitHub/Data_Science_Airbnb/nn_config.yaml', 'r') as stream:
        try:
            config =yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as e:
            print(e)
        
        return config


  
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

 
    
    
    
    
   
    
    
    


        
     






